import argparse, torch
from src.utils import load_yaml
from src.data import make_loader
from src.modeling import HybridModel
from src.metrics import classification_metrics

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_cfg", type=str, required=True)
    ap.add_argument("--test_cfg", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    return ap.parse_args()

def main():
    args = parse_args()
    tr = load_yaml(args.train_cfg)
    te = load_yaml(args.test_cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tasks = ("cls",)

    model = HybridModel(img_size=te["image_size"], patch_size=te["patch_size"], in_chans=3,
                        dim=tr["model"]["d_model"], ssm_depth=tr["model"]["ssm_depth"],
                        vit_depth=tr["model"]["vit_depth"], heads=tr["model"]["heads"],
                        mlp_ratio=tr["model"]["mlp_ratio"], num_classes=te["classes"],
                        use_fusion_gate=tr["model"]["use_fusion_gate"], tasks=tasks).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"]); model.eval()

    test_loader  = make_loader(te["dataset"], te["splits_csv"].replace(".csv","_test.csv"),
                               te["image_size"], "mild", te["train"]["batch_size"], te["train"]["num_workers"], shuffle=False)

    logits_list, labels_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            out = model(x)
            logits_list.append(out["logits"].cpu()); labels_list.append(y.cpu())

    logits = torch.cat(logits_list); labels = torch.cat(labels_list)
    cm = classification_metrics(logits, labels, te["classes"])
    print("[CROSS-DATASET]", cm)

if __name__ == "__main__":
    main()
