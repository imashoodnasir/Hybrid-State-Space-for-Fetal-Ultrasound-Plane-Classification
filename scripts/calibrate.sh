python -m src.train --config configs/planes_fpdb.yaml
python -m src.evaluate --config configs/planes_fpdb.yaml --ckpt runs/fpdb/best.pth --use_temp_scaler
