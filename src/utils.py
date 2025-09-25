import os, random, json, math, time
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_json(d, path):
    with open(path, "w") as f:
        json.dump(d, f, indent=2)

def load_yaml(path):
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def cosine_anneal_lr(optimizer, base_lr, step, total_steps):
    lr = 0.5 * base_lr * (1 + math.cos(math.pi * step / total_steps))
    for pg in optimizer.param_groups:
        pg['lr'] = lr

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum = 0.0; self.cnt = 0
    def update(self, val, n=1):
        self.sum += val * n; self.cnt += n
    @property
    def avg(self): 
        return self.sum / max(1, self.cnt)

def timer():
    t0 = time.time()
    def elapsed():
        return time.time() - t0
    return elapsed
