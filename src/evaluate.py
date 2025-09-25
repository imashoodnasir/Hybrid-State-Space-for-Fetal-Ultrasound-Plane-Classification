import os, argparse, torch
from src.utils import load_yaml
from src.data import make_loader
from src.modeling import HybridModel
from src.metrics import classification_metrics, expected_calibration_error, dice_iou, hd95, regression_metrics

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--use_temp_scaler", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_loader = make_loader(cfg["dataset"], cfg["splits_csv"].replace(".csv","_test.csv"),
                              cfg["image_size"], "mild", cfg["train"]["batch_size"], cfg["train"]["num_workers"], shuffle=False)

    tasks = ("cls",) if cfg["task"]=="classification" else (
            ("cls","reg") if cfg["task"]=="classification_regression" else
            ("seg","reg") if cfg["task"]=="segmentation_regression" else ("cls","seg","reg"))

    model = HybridModel(img_size=cfg["image_size"], patch_size=cfg["patch_size"], in_chans=3,
                        dim=cfg["model"]["d_model"], ssm_depth=cfg["model"]["ssm_depth"],
                        vit_depth=cfg["model"]["vit_depth"], heads=cfg["model"]["heads"],
                        mlp_ratio=cfg["model"]["mlp_ratio"], num_classes=cfg["classes"],
                        use_fusion_gate=cfg["model"]["use_fusion_gate"], tasks=tasks).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"]); model.eval()

    scalerT = None
    if args.use_temp_scaler and "cls" in tasks:
        ts_path = os.path.join(os.path.dirname(args.ckpt), "temp_scaler.pth")
        if os.path.exists(ts_path):
            from src.calibration import TemperatureScaler
            scalerT = TemperatureScaler().to(device)
            scalerT.load_state_dict(torch.load(ts_path, map_location=device)["T"])

    cls_logits, cls_labels = [], []
    seg_logits, seg_masks  = [], []
    reg_preds, reg_gts     = [], []

    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            out = model(x)
            if "cls" in tasks:
                logits = out["logits"]
                if scalerT is not None:
                    logits = scalerT(logits)
                cls_logits.append(logits.cpu()); cls_labels.append(batch[1].cpu())
            if "seg" in tasks:
                seg_logits.append(out["mask"].cpu()); seg_masks.append(batch[1].cpu())
            if "reg" in tasks:
                y = batch[2] if "seg" in tasks else batch[2]
                reg_preds.append(out["circ"].cpu()); reg_gts.append(y.cpu())

    scores = {}
    if "cls" in tasks and len(cls_logits)>0:
        logits = torch.cat(cls_logits); labels = torch.cat(cls_labels)
        cm = classification_metrics(logits, labels, cfg["classes"])
        import numpy as np
        probs = torch.softmax(logits, dim=-1).numpy()
        ece = expected_calibration_error(probs, labels.numpy())
        cm["ece"] = ece
        scores.update(cm)
    if "seg" in tasks and len(seg_logits)>0:
        d, i = dice_iou(torch.cat(seg_logits), torch.cat(seg_masks))
        h = hd95(torch.cat(seg_logits), torch.cat(seg_masks))
        scores["dice"] = d; scores["iou"] = i; scores["hd95"] = h
    if "reg" in tasks and len(reg_preds)>0:
        rm = regression_metrics(torch.cat(reg_preds), torch.cat(reg_gts))
        scores.update(rm)
    print("[TEST]", scores)

if __name__ == "__main__":
    main()
