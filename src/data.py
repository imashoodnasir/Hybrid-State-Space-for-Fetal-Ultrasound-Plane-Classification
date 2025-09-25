import cv2, os, numpy as np, pandas as pd
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.exposure import equalize_adapthist

def anisotropic_diffusion(img, niter=5, k=20.0, lam=0.2):
    img = img.astype(np.float32)
    for _ in range(niter):
        nablaN = np.roll(img, -1, axis=0) - img
        nablaS = np.roll(img,  1, axis=0) - img
        nablaE = np.roll(img, -1, axis=1) - img
        nablaW = np.roll(img,  1, axis=1) - img
        cN = np.exp(-(nablaN/k)**2); cS = np.exp(-(nablaS/k)**2)
        cE = np.exp(-(nablaE/k)**2); cW = np.exp(-(nablaW/k)**2)
        img = img + lam*(cN*nablaN + cS*nablaS + cE*nablaE + cW*nablaW)
    return img

def zscore(img):
    m, s = img.mean(), img.std()
    if s < 1e-6: s = 1.0
    return (img - m) / s

def clahe_gray(img):
    img8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return equalize_adapthist(img8, clip_limit=0.01).astype(np.float32)

def affine_align(img, angle=0.0, scale=1.0):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

def build_transforms(size=224, aug="strong", is_mask=False):
    if is_mask:
        return A.Compose([A.Resize(size, size, interpolation=cv2.INTER_NEAREST)])
    aug_list = [A.Resize(size, size)]
    if aug == "strong":
        aug_list += [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.7, border_mode=cv2.BORDER_REFLECT_101),
            A.CLAHE(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.CoarseDropout(max_holes=4, max_height=int(size*0.1), max_width=int(size*0.1), p=0.2),
        ]
    elif aug == "mild":
        aug_list += [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=10, p=0.5, border_mode=cv2.BORDER_REFLECT_101),
        ]
    aug_list += [A.Normalize(mean=0.0, std=1.0), ToTensorV2()]
    return A.Compose(aug_list)

def preprocess_gray(img, do_diff=True, do_affine=True):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = zscore(img)
    if do_diff:
        img = anisotropic_diffusion(img, niter=3)
    if do_affine:
        img = affine_align(img, angle=0.0, scale=1.0)
    img = clahe_gray(img)
    return img

class FPDB(Dataset):
    def __init__(self, csv_path, size=224, aug="strong"):
        df = pd.read_csv(csv_path)
        self.paths = df['filepath'].tolist()
        self.labels = df['label'].astype(int).tolist()
        self.tfm = build_transforms(size, aug=aug)
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = cv2.imread(self.paths[i], cv2.IMREAD_GRAYSCALE)
        img = preprocess_gray(img)
        img = self.tfm(image=img)['image'].unsqueeze(0).repeat(3,1,1)
        return img, self.labels[i]

class HeadLarge(Dataset):
    def __init__(self, csv_path, size=224, aug="strong"):
        df = pd.read_csv(csv_path)
        self.paths = df['filepath'].tolist()
        self.labels = df['label'].astype(int).tolist()
        self.circ   = df['circumference'].astype(float).tolist()
        self.tfm = build_transforms(size, aug=aug)
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = cv2.imread(self.paths[i], cv2.IMREAD_GRAYSCALE)
        img = preprocess_gray(img)
        img = self.tfm(image=img)['image'].unsqueeze(0).repeat(3,1,1)
        return img, self.labels[i], np.float32(self.circ[i])

class HC18(Dataset):
    def __init__(self, csv_path, size=224, aug="mild"):
        df = pd.read_csv(csv_path)
        self.imgs = df['filepath_img'].tolist()
        self.masks= df['filepath_mask'].tolist()
        self.circ = df['circumference'].astype(float).tolist()
        self.tfm_img  = build_transforms(size, aug=aug)
        self.tfm_mask = build_transforms(size, aug=aug, is_mask=True)
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        img  = cv2.imread(self.imgs[i], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.masks[i], cv2.IMREAD_GRAYSCALE)
        img  = preprocess_gray(img, do_diff=True, do_affine=True)
        mask = (mask>127).astype(np.uint8)
        img  = self.tfm_img(image=img)['image'].unsqueeze(0).repeat(3,1,1)
        mask = self.tfm_mask(image=mask)['image'].long()
        return img, mask, np.float32(self.circ[i])

def make_loader(name, csv, size, aug, batch, workers, shuffle=True):
    if name == "FPDB":
        ds = FPDB(csv, size, aug)
    elif name == "HEAD_LARGE":
        ds = HeadLarge(csv, size, aug)
    elif name == "HC18":
        ds = HC18(csv, size, aug)
    else:
        raise ValueError(f"Unknown dataset {name}")
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers, pin_memory=True)
