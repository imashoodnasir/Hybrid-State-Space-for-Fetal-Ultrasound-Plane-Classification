import torch, torch.nn as nn
from sklearn.metrics import log_loss

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_T = nn.Parameter(torch.zeros(1))
    def forward(self, logits):
        T = torch.exp(self.log_T)
        return logits / T

def fit_temperature(model, val_loader, device, num_classes):
    model.eval()
    scaler = TemperatureScaler().to(device)
    optimizer = torch.optim.LBFGS(scaler.parameters(), lr=0.1, max_iter=50)

    logits_list, labels_list = [], []
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            out = model(x)
            logits_list.append(out["logits"].detach())
            labels_list.append(y.detach())
    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    def closure():
        optimizer.zero_grad()
        scaled = scaler(logits)
        loss = torch.nn.functional.cross_entropy(scaled, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        scaled = scaler(logits)
        probs = torch.softmax(scaled, dim=-1).cpu().numpy()
        nll = log_loss(labels.cpu().numpy(), probs, labels=list(range(num_classes)))
    return scaler, float(nll)
