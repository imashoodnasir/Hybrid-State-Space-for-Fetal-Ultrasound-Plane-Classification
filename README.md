# Hybrid State-Space + Vision Transformer for Fetal Ultrasound (Python)

This repository reproduces the paper’s pipeline: preprocessing → Hybrid SSM+ViT with gated fusion → multi-task heads (classification / segmentation / circumference regression) → calibration (temperature scaling) → evaluation, cross-dataset tests, and ablations.

## 1) Setup
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

## 2) Data
Expected CSVs (patient-wise splits):
- `splits/fpdb_splits_train.csv`, `_val.csv`, `_test.csv` with columns: `filepath,label,patient_id`
- `splits/head_large_splits_*.csv` with: `filepath,label,patient_id,circumference`
- `splits/hc18_splits_*.csv` with: `filepath_img,filepath_mask,patient_id,circumference`

Place datasets under `data/` and update `configs/*.yaml` if paths differ.

## 3) Train
```bash
bash scripts/train_fpdb.sh
bash scripts/train_head_large.sh
bash scripts/train_hc18.sh
```

## 4) Evaluate and Calibrate
```bash
# (after FPDB training)
bash scripts/calibrate.sh
# or
python -m src.evaluate --config configs/planes_fpdb.yaml --ckpt runs/fpdb/best.pth --use_temp_scaler
```

## 5) Cross-Dataset
```bash
python -m src.cross_dataset --train_cfg configs/planes_fpdb.yaml --test_cfg configs/head_large.yaml --ckpt runs/fpdb/best.pth
```

## 6) Ablations
Edit YAML toggles (e.g., disable `use_fusion_gate`, set `task` to isolate heads) and run `scripts/ablations.sh`.

## Notes
- The SSM block here uses a depthwise 1D token-mixing proxy for long-range modeling; you can swap in S4 for research experiments.
- HC18 enables shape-consistency loss by default.
- Temperature scaling is saved to `temp_scaler.pth` and used in `evaluate.py --use_temp_scaler`.
- Ensure **patient-level** split integrity to avoid leakage.
