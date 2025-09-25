import torch, numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss

def classification_metrics(logits, labels, num_classes):
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    preds = probs.argmax(axis=1)
    y = labels.cpu().numpy()
    acc = accuracy_score(y, preds)
    macro_f1 = f1_score(y, preds, average="macro")
    try:
        auroc = roc_auc_score(y, probs, multi_class="ovr")
    except Exception:
        auroc = float("nan")
    nll = log_loss(y, probs, labels=list(range(num_classes)))
    return {"acc": acc, "macro_f1": macro_f1, "auroc": auroc, "nll": nll}

def expected_calibration_error(probs, labels, n_bins=15):
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    y = labels
    bins = np.linspace(0, 1, n_bins+1)
    ece = 0.0; N = len(y)
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i+1])
        if m.sum() == 0: continue
        acc = (preds[m] == y[m]).mean()
        conf_mean = conf[m].mean()
        ece += (m.sum()/N) * abs(acc - conf_mean)
    return ece

def dice_iou(pred_logits, gt_mask, thresh=0.5):
    p = (torch.sigmoid(pred_logits) > thresh).float()
    inter = (p*gt_mask).sum(dim=(1,2))
    dice = (2*inter) / (p.sum(dim=(1,2)) + gt_mask.sum(dim=(1,2)).clamp_min(1))
    union = (p+gt_mask).clamp_max(1).sum(dim=(1,2)).clamp_min(1)
    iou = inter/union
    return dice.mean().item(), iou.mean().item()

def hd95(pred_logits, gt_mask, spacing=(1.0,1.0)):
    import scipy.ndimage as ndi
    p = (torch.sigmoid(pred_logits) > 0.5).float().cpu().numpy()
    g = gt_mask.cpu().numpy().astype(np.float32)
    dists = []
    for i in range(p.shape[0]):
        A, B = p[i], g[i]
        dtA = ndi.distance_transform_edt(1 - A, sampling=spacing)
        dtB = ndi.distance_transform_edt(1 - B, sampling=spacing)
        sA = (A - ndi.binary_erosion(A)).astype(bool)
        sB = (B - ndi.binary_erosion(B)).astype(bool)
        dAB = dtB[sA]
        dBA = dtA[sB]
        if dAB.size == 0 or dBA.size == 0:
            dists.append(0.0)
        else:
            dists.append(np.percentile(np.concatenate([dAB, dBA]), 95))
    return float(np.mean(dists))

def regression_metrics(pred, target):
    p = pred.detach().cpu().numpy()
    t = target.detach().cpu().numpy()
    mae = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t)**2)))
    corr = float(np.corrcoef(p, t)[0,1]) if len(p)>1 else 0.0
    return {"mae": mae, "rmse": rmse, "corr": corr}
