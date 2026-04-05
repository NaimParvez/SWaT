"""
lstm_model.py
=============
Baseline LSTM autoencoder with MC Dropout for uncertainty quantification.
Two modes:
  1. Standard:  dropout OFF at inference → point reconstruction
  2. MC Dropout: dropout ON at inference, T=20 passes → mean + std
"""

import torch
import torch.nn as nn
import numpy as np

# ── Model ──────────────────────────────────────────────────────────────

class LSTMAutoencoder(nn.Module):
    """
    LSTM encoder-decoder for anomaly detection on SWaT.
    Architecture: LSTM(64) → LSTM(16) → LSTM(16) → LSTM(64)
    Dropout applied after each LSTM layer — kept ON at inference
    for MC Dropout uncertainty estimation.
    """

    def __init__(self, n_features=51, hidden=64, latent=16,
                 dropout=0.2, window=20):
        super().__init__()
        self.n_features = n_features
        self.window     = window
        self.latent     = latent

        # Encoder
        self.enc1 = nn.LSTM(n_features, hidden, batch_first=True)
        self.enc2 = nn.LSTM(hidden, latent, batch_first=True)

        # Decoder
        self.dec1 = nn.LSTM(latent, latent, batch_first=True)
        self.dec2 = nn.LSTM(latent, hidden, batch_first=True)
        self.out  = nn.Linear(hidden, n_features)

        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        x: [batch, window, n_features]
        returns: reconstruction [batch, window, n_features]
        """
        # Encode
        e1, _ = self.enc1(x)
        e1 = self.drop(e1)
        e2, _ = self.enc2(e1)
        e2 = self.drop(e2)

        # Decode — use encoder output as decoder input directly
        d1, _ = self.dec1(e2)
        d1 = self.drop(d1)
        d2, _ = self.dec2(d1)
        d2 = self.drop(d2)

        return self.out(d2)


def enable_mc_dropout(model):
    """Set all Dropout layers to training mode (MC Dropout)."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def reconstruction_error(x, x_hat):
    """Per-sample RMSE over window and features. Returns [batch]."""
    return torch.sqrt(((x - x_hat) ** 2).mean(dim=[1, 2]))


# ── Training ───────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for x, y, _ in loader:
        # Train ONLY on normal windows
        normal_mask = (y == 0)
        if normal_mask.sum() == 0:
            continue
        x = x[normal_mask].to(device)
        optimizer.zero_grad()
        x_hat = model(x)
        loss = nn.MSELoss()(x_hat, x)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)


def val_epoch(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y, _ in loader:
            normal_mask = (y == 0)
            if normal_mask.sum() == 0:
                continue
            x = x[normal_mask].to(device)
            x_hat = model(x)
            loss = nn.MSELoss()(x_hat, x)
            total_loss += loss.item() * len(x)
    return total_loss / max(1, len(loader.dataset))


# ── Threshold ──────────────────────────────────────────────────────────

def compute_threshold(model, train_loader, device, percentile=95):
    """
    Fit anomaly threshold on training reconstruction errors.
    Uses normal samples only. Returns scalar threshold value.
    """
    model.eval()
    errors = []
    with torch.no_grad():
        for x, y, _ in train_loader:
            normal_mask = (y == 0)
            if normal_mask.sum() == 0:
                continue
            x = x[normal_mask].to(device)
            x_hat = model(x)
            err = reconstruction_error(x, x_hat)
            errors.extend(err.cpu().numpy())
    threshold = np.percentile(errors, percentile)
    print(f"  Threshold (p{percentile}): {threshold:.4f}")
    print(f"  Error range: [{min(errors):.4f}, {max(errors):.4f}]")
    return threshold


# ── MC Dropout inference ───────────────────────────────────────────────

@torch.no_grad()
def mc_predict(model, x, T=20, device='cpu'):
    """
    Run T stochastic forward passes with MC Dropout.

    Parameters
    ----------
    model : LSTMAutoencoder (dropout enabled)
    x     : tensor [batch, window, features]
    T     : number of MC passes
    device: torch device

    Returns
    -------
    mean_error : [batch]  — anomaly score
    std_error  : [batch]  — epistemic uncertainty
    all_errors : [T, batch] — raw errors for SHAP distribution
    """
    model.eval()
    enable_mc_dropout(model)   # keep dropout ON

    x = x.to(device)
    all_errors = []

    for _ in range(T):
        x_hat = model(x)
        err = reconstruction_error(x, x_hat)
        all_errors.append(err.cpu())

    all_errors  = torch.stack(all_errors)   # [T, batch]
    mean_error  = all_errors.mean(dim=0)    # [batch]
    std_error   = all_errors.std(dim=0)     # [batch]

    return mean_error, std_error, all_errors


# ── Full evaluation ────────────────────────────────────────────────────

from sklearn.metrics import (roc_auc_score, f1_score,
                              precision_score, recall_score,
                              average_precision_score)

def evaluate(model, loader, threshold, device, T=20, use_mc=True):
    """
    Evaluate anomaly detection performance on a DataLoader.
    Returns dict of metrics.
    """
    all_labels  = []
    all_scores  = []
    all_stds    = []

    for x, y, _ in loader:
        if use_mc:
            mean_err, std_err, _ = mc_predict(model, x, T=T, device=device)
        else:
            model.eval()
            with torch.no_grad():
                x_hat = model(x.to(device))
                mean_err = reconstruction_error(x.to(device), x_hat).cpu()
            std_err = torch.zeros_like(mean_err)

        all_scores.extend(mean_err.numpy())
        all_stds.extend(std_err.numpy())
        all_labels.extend(y.numpy())

    scores = np.array(all_scores)
    stds   = np.array(all_stds)
    labels = np.array(all_labels)
    preds  = (scores > threshold).astype(int)

    metrics = {
        'auroc':     roc_auc_score(labels, scores),
        'auprc':     average_precision_score(labels, scores),
        'f1':        f1_score(labels, preds, zero_division=0),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall':    recall_score(labels, preds, zero_division=0),
        'mean_uncertainty_normal': stds[labels == 0].mean(),
        'mean_uncertainty_attack': stds[labels == 1].mean(),
    }
    return metrics, scores, stds, labels