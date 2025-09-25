import torch, torch.nn as nn, torch.nn.functional as F

def class_weighted_ce(logits, target, num_classes):
    with torch.no_grad():
        counts = torch.bincount(target, minlength=num_classes).float().clamp_min(1)
        w = counts.sum() / counts
        w = w / w.mean()
    return F.cross_entropy(logits, target, weight=w.to(logits.device))

def dice_loss(pred_logits, target_mask, eps=1e-6):
    pred = torch.sigmoid(pred_logits)
    num = 2*(pred*target_mask).sum(dim=(1,2))
    den = pred.sum(dim=(1,2)) + target_mask.sum(dim=(1,2)) + eps
    return (1 - (num/den)).mean()

def smooth_l1_loss(pred, target):
    return F.smooth_l1_loss(pred, target)

def entropy_reg(logits, lam=1e-3):
    p = F.softmax(logits, dim=-1)
    ent = -(p * torch.log(p.clamp_min(1e-8))).sum(dim=-1).mean()
    return -lam * ent

def ellipse_params_from_mask(mask):
    B, H, W = mask.shape
    device = mask.device
    y = torch.arange(H, device=device).float().view(1,H,1).expand(B,H,W)
    x = torch.arange(W, device=device).float().view(1,1,W).expand(B,H,W)
    m00 = mask.sum(dim=(1,2)).clamp_min(1.0)
    xbar = (x*mask).sum(dim=(1,2))/m00
    ybar = (y*mask).sum(dim=(1,2))/m00
    x_c = x - xbar.view(B,1,1)
    y_c = y - ybar.view(B,1,1)
    mu20 = ( (x_c**2)*mask ).sum(dim=(1,2))/m00
    mu02 = ( (y_c**2)*mask ).sum(dim=(1,2))/m00
    mu11 = ( (x_c*y_c)*mask ).sum(dim=(1,2))/m00
    theta = 0.5*torch.atan2(2*mu11, (mu20 - mu02 + 1e-8))
    tr = mu20 + mu02
    det= mu20*mu02 - mu11**2
    lam1 = (tr/2) + torch.sqrt((tr/2)**2 - det + 1e-8)
    lam2 = (tr/2) - torch.sqrt((tr/2)**2 - det + 1e-8)
    a = torch.sqrt(lam1).clamp_min(1e-6)
    b = torch.sqrt(lam2).clamp_min(1e-6)
    return a, b, theta

def shape_consistency_loss(pred_mask_logits, gt_mask):
    pred_bin = (torch.sigmoid(pred_mask_logits) > 0.5).float()
    a_p, b_p, th_p = ellipse_params_from_mask(pred_bin)
    a_g, b_g, th_g = ellipse_params_from_mask(gt_mask.float())
    return ((a_p - a_g).abs() + (b_p - b_g).abs() + (th_p - th_g).abs()).mean()

def multi_task_loss(outputs, batch, tasks, sigmas, num_classes, cfg):
    total = 0.0
    logs = {}
    if "cls" in tasks:
        logits = outputs["logits"]
        ce = class_weighted_ce(logits, batch["label"], num_classes)
        ce += entropy_reg(logits, cfg["loss"].get("entropy_lambda", 1e-3))
        logs["L_cls"] = ce.item()
        total += (1/(2*sigmas["sig_cls"]**2))*ce + torch.log(sigmas["sig_cls"])
    if "seg" in tasks:
        dl = dice_loss(outputs["mask"], batch["mask"].float())
        logs["L_dice"] = dl.item()
        total += (1/(2*sigmas["sig_seg"]**2))*dl + torch.log(sigmas["sig_seg"])
        if cfg["loss"].get("use_shape_consistency", False):
            sl = shape_consistency_loss(outputs["mask"], batch["mask"])
            logs["L_shape"] = sl.item()
            total += (1/(2*sigmas["sig_seg"]**2))*sl
    if "reg" in tasks:
        hub = smooth_l1_loss(outputs["circ"], batch["circ"])
        logs["L_reg"] = hub.item()
        total += (1/(2*sigmas["sig_reg"]**2))*hub + torch.log(sigmas["sig_reg"])
    return total, logs
