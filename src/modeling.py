import torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=512):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        n_h = img_size // patch_size
        n_w = img_size // patch_size
        self.num_patches = n_h * n_w
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.randn(1, 1 + self.num_patches, embed_dim) * 0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        B = x.shape[0]
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos
        return x

class SSMBlock(nn.Module):
    def __init__(self, dim, kernel_size=15, mlp_ratio=4):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*mlp_ratio),
            nn.GELU(),
            nn.Linear(dim*mlp_ratio, dim),
        )
        self.gamma = nn.Parameter(torch.ones(1))
    def forward(self, x):
        y = x.transpose(1, 2)
        y = self.dwconv(y).transpose(1, 2)
        y = self.norm(y)
        y = self.mlp(y)
        return x + self.gamma * y

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.drop1 = nn.Dropout(proj_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*mlp_ratio), nn.GELU(), nn.Dropout(proj_drop),
            nn.Linear(dim*mlp_ratio, dim), nn.Dropout(proj_drop),
        )
    def forward(self, x):
        q = k = self.norm1(x)
        attn_out, _ = self.attn(q, k, x)
        x = x + self.drop1(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x

class FusionGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Linear(dim, dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta  = nn.Parameter(torch.tensor(0.5))
    def forward(self, f_ssm, f_vit):
        gate = torch.sigmoid(self.g(f_ssm))
        fused = gate * f_ssm + (1 - gate) * f_vit
        return fused + self.alpha * f_ssm + self.beta * f_vit

class ClassificationHead(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)
    def forward(self, x_cls):
        return self.fc(x_cls)

class RegressionHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
    def forward(self, x_cls):
        return self.mlp(x_cls).squeeze(-1)

class SegmentationHead(nn.Module):
    def __init__(self, dim, img_size=224, patch=16):
        super().__init__()
        self.h = img_size // patch
        self.w = img_size // patch
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim//2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(dim//2, 1, 1)
        )
    def forward(self, tokens):
        x = tokens[:,1:,:]
        B, N, D = x.shape
        x = x.transpose(1,2).reshape(B, D, self.h, self.w)
        x = self.conv(x)
        x = torch.nn.functional.interpolate(x, scale_factor=16, mode='bilinear', align_corners=False)
        return x.squeeze(1)

class HybridModel(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, dim=512,
                 ssm_depth=4, vit_depth=6, heads=8, mlp_ratio=4,
                 num_classes=12, use_fusion_gate=True,
                 tasks=("cls",)):
        super().__init__()
        self.pe = PatchEmbed(img_size, patch_size, in_chans, dim)
        self.ssm = nn.ModuleList([SSMBlock(dim, kernel_size=15, mlp_ratio=mlp_ratio) for _ in range(ssm_depth)])
        self.vit = nn.ModuleList([TransformerBlock(dim, heads, mlp_ratio) for _ in range(vit_depth)])
        self.fuse = FusionGate(dim) if use_fusion_gate else None
        self.tasks = set(tasks)
        if "cls" in self.tasks: self.cls_head = ClassificationHead(dim, num_classes)
        if "reg" in self.tasks: self.reg_head = RegressionHead(dim)
        if "seg" in self.tasks: self.seg_head = SegmentationHead(dim, img_size, patch_size)
        self.learn_sigma = nn.ParameterDict()
        for t in self.tasks:
            self.learn_sigma[f"sig_{t}"] = nn.Parameter(torch.tensor(0.2))

    def forward(self, x):
        tok = self.pe(x)
        s = tok
        for blk in self.ssm: s = blk(s)
        v = tok
        for blk in self.vit: v = blk(v)
        s_cls, v_cls = s[:,0], v[:,0]
        if self.fuse is not None:
            f_cls = self.fuse(s_cls, v_cls)
            f_tok = self.fuse(s, v)
        else:
            f_cls = 0.5*(s_cls + v_cls)
            f_tok = 0.5*(s + v)
        out = {}
        if "cls" in self.tasks: out["logits"] = self.cls_head(f_cls)
        if "reg" in self.tasks: out["circ"]   = self.reg_head(f_cls)
        if "seg" in self.tasks: out["mask"]   = self.seg_head(f_tok)
        return out
