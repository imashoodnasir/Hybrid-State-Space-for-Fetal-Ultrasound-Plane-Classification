import os, argparse, torch
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler

from src.utils import set_seed, ensure_dir, load_yaml, save_json, AverageMeter
from src.data import make_loader
from src.modeling import HybridModel
from src.losses import multi_task_loss
from src.metrics import classification_metrics, expected_calibration_error, dice_iou, hd95, regression_metrics
from src.calibration import fit_temperature

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    return ap.parse_args()

def build_tasks(task_name):
    if task_name == "classification": return ("cls",)
    if task_name == "classification_regression": return ("cls","reg")
    if task_name == "segmentation_regression": return ("seg","reg")
    if task_name == "all": return ("cls","seg","reg")
    raise ValueError(task_name)

def main():
    args = parse_args()
    cfg  = load_yaml(args.config)
    set_seed(cfg.get("seed",42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_dir(cfg["save_dir"])

    train_loader = make_loader(cfg["dataset"], cfg["splits_csv"].replace(".csv","_train.csv"),
                               cfg["image_size"], cfg["train"]["aug"], cfg["train"]["batch_size"], cfg["train"]["num_workers"], shuffle=True)
    val_loader   = make_loader(cfg["dataset"], cfg["splits_csv"].replace(".csv","_val.csv"),
                               cfg["image_size"], "mild", cfg["train"]["batch_size"], cfg["train"]["num_workers"], shuffle=False)
    test_loader  = make_loader(cfg["dataset"], cfg["splits_csv"].replace(".csv","_test.csv"),
                               cfg["image_size"], "mild", cfg["train"]["batch_size"], cfg["train"]["num_workers"], shuffle=False)

    tasks = build_tasks(cfg["task"])
    model = HybridModel(img_size=cfg["image_size"], patch_size=cfg["patch_size"], in_chans=3,
                        dim=cfg["model"]["d_model"], ssm_depth=cfg["model"]["ssm_depth"],
                        vit_depth=cfg["model"]["vit_depth"], heads=cfg["model"]["heads"],
                        mlp_ratio=cfg["model"]["mlp_ratio"], num_classes=cfg["classes"],
                        use_fusion_gate=cfg["model"]["use_fusion_gate"], tasks=tasks).to(device)

    opt = Adam(model.parameters(), lr=cfg["optimizer"]["lr"], weight_decay=cfg["optimizer"]["weight_decay"])
    scaler = GradScaler()
    best_key = "macro_f1" if "cls" in tasks else ("dice" if "seg" in tasks else "rmse")
    best_score = -1e9 if best_key in ("macro_f1","dice") else 1e9
    epochs = cfg["train"]["epochs"]

    for epoch in range(1, epochs+1):
        model.train()
        loss_meter = AverageMeter()
        for batch in train_loader:
            x = batch[0].to(device)
            bdict = {}
            if "cls" in tasks: bdict["label"] = batch[1].to(device)
            if "reg" in tasks and len(batch)>=3: bdict["circ"] = batch[2].to(device)
            if "seg" in tasks:
                bdict["mask"] = batch[1].to(device)
                if "reg" in tasks: bdict["circ"] = batch[2].to(device)

            opt.zero_grad()
            with autocast():
                out = model(x)
                loss, _logs = multi_task_loss(out, bdict, tasks, model.learn_sigma, cfg["classes"], cfg)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            loss_meter.update(loss.item(), x.size(0))

        # Validation
        model.eval()
        cls_logits, cls_labels = [], []
        seg_logits, seg_masks  = [], []
        reg_preds, reg_gts     = [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                out = model(x)
                if "cls" in tasks:
                    cls_logits.append(out["logits"].cpu()); cls_labels.append(batch[1].cpu())
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
            scores.update(cm); scores["ece"] = ece
        if "seg" in tasks and len(seg_logits)>0:
            d, i = dice_iou(torch.cat(seg_logits), torch.cat(seg_masks))
            scores["dice"] = d; scores["iou"] = i
        if "reg" in tasks and len(reg_preds)>0:
            rm = regression_metrics(torch.cat(reg_preds), torch.cat(reg_gts))
            scores.update(rm)

        cur = scores.get(best_key, 0.0)
        better = (cur > best_score) if best_key in ("macro_f1","dice") else (cur < best_score)
        if better:
            best_score = cur
            torch.save({"model": model.state_dict(), "cfg": cfg}, os.path.join(cfg["save_dir"], "best.pth"))

        print(f"Epoch {epoch}/{epochs} | train_loss={loss_meter.avg:.4f} | val={scores}")

    if cfg["eval"].get("calibrate_temperature", False) and "cls" in tasks:
        ckpt = torch.load(os.path.join(cfg["save_dir"], "best.pth"), map_location=device)
        model.load_state_dict(ckpt["model"]); model.eval()
        scalerT, nll = fit_temperature(model, val_loader, device, cfg["classes"])
        torch.save({"T": scalerT.state_dict()}, os.path.join(cfg["save_dir"], "temp_scaler.pth"))
        print(f"[Calibration] Val NLL after TS: {nll:.4f}")

if __name__ == "__main__":
    main()
