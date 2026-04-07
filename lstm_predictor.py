"""
lstm_predictor.py
=================
Prediction-based LSTM for SWaT anomaly detection.
Predicts x(t+1) from window x(t-19:t).
Per-sensor prediction error gives localized signal for SHAP.

Key difference from autoencoder:
  AE:   input window → reconstruct SAME window → global error scalar
  Pred: input window → predict NEXT timestep  → per-sensor error [51]
  
SHAP on prediction model: which input sensors caused the 
prediction of sensor i to be wrong? → clean localization signal.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from swat_loader import load_swat, ATTACK_MAP

# ── Dataset ──────────────────────────────────────────────────────────

class SWaTPredictor(Dataset):
    """
    Returns (X_window, y_next, label) where:
      X_window: [window, 51] — input window
      y_next:   [51]         — true values at t+1
      label:    float        — 0 normal, 1 attack (of the NEXT timestep)
    """
    def __init__(self, df, features, window_size=20):
        self.X      = df[features].values.astype(np.float32)
        self.labels = df['label'].values.astype(np.float32)
        self.w      = window_size

    def __len__(self):
        return len(self.X) - self.w  # need t+1

    def __getitem__(self, idx):
        x_win  = torch.tensor(self.X[idx:idx + self.w])       # [w, 51]
        y_next = torch.tensor(self.X[idx + self.w])           # [51]
        label  = torch.tensor(float(
            self.labels[idx:idx + self.w + 1].max()))         # scalar
        return x_win, y_next, label


# ── Model ─────────────────────────────────────────────────────────────

class LSTMPredictor(nn.Module):
    """
    Two-layer LSTM → Linear → predict all 51 sensors at t+1.
    Dropout after each LSTM layer (MC Dropout compatible).
    """
    def __init__(self, n_features=51, hidden=64, dropout=0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, hidden, batch_first=True)
        self.lstm2 = nn.LSTM(hidden, hidden, batch_first=True)
        self.drop  = nn.Dropout(p=dropout)
        self.head  = nn.Linear(hidden, n_features)

    def forward(self, x):
        # x: [batch, window, 51]
        h1, _ = self.lstm1(x)
        h1     = self.drop(h1)
        h2, _ = self.lstm2(h1)
        h2     = self.drop(h2)
        # Use only the LAST hidden state for prediction
        out    = self.head(h2[:, -1, :])  # [batch, 51]
        return out


def enable_mc_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


# ── Per-sensor error ──────────────────────────────────────────────────

def per_sensor_error(y_true, y_pred):
    """Squared error per sensor. Returns [batch, 51]."""
    return (y_true - y_pred) ** 2


def anomaly_score(y_true, y_pred):
    """Mean squared error across sensors. Returns [batch]."""
    return per_sensor_error(y_true, y_pred).mean(dim=1)


# ── Training ──────────────────────────────────────────────────────────

def train_predictor(
    normal_csv='/kaggle/input/datasets/naimparvezz/swat-dataset/SWaT_Dataset_Normal_v1.csv',
    attack_csv='/kaggle/input/datasets/naimparvezz/swat-dataset/SWaT_Dataset_Attack_v0.csv',
    epochs=40, batch=512, lr=1e-3, device='cuda',
    save_path='lstm_predictor_best.pt'
):
    DEVICE = device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {DEVICE}")

    # Load data
    train_df, val_df, test_df, scaler, feats = load_swat(
        normal_csv=normal_csv,
        attack_csv=attack_csv,
        verbose=False
    )

    train_ds = SWaTPredictor(train_df, feats, window_size=20)
    val_ds   = SWaTPredictor(val_df,   feats, window_size=20)
    test_ds  = SWaTPredictor(test_df,  feats, window_size=20)

    train_loader = DataLoader(train_ds, batch_size=batch,
                              shuffle=True,  num_workers=2,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch,
                              shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=batch,
                              shuffle=False, num_workers=2)

    model = LSTMPredictor(n_features=51, hidden=64, dropout=0.2).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr,
                             weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=4, factor=0.5, verbose=False)

    best_val  = float('inf')
    best_state = None

    print(f"\n{'Epoch':>6} {'Train MSE':>12} {'Val MSE':>12}")
    print("─" * 35)

    for epoch in range(1, epochs + 1):
        # Train on normal only
        model.train()
        tr_loss = 0
        for x, y_next, label in train_loader:
            normal = label == 0
            if normal.sum() == 0:
                continue
            x      = x[normal].to(DEVICE)
            y_next = y_next[normal].to(DEVICE)
            opt.zero_grad()
            pred = model(x)
            loss = nn.MSELoss()(pred, y_next)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * len(x)
        tr_loss /= len(train_loader.dataset)

        # Validate on normal only
        model.eval()
        va_loss = 0
        with torch.no_grad():
            for x, y_next, label in val_loader:
                normal = label == 0
                if normal.sum() == 0:
                    continue
                x      = x[normal].to(DEVICE)
                y_next = y_next[normal].to(DEVICE)
                pred   = model(x)
                loss   = nn.MSELoss()(pred, y_next)
                va_loss += loss.item() * len(x)
        va_loss /= max(1, sum(
            (label == 0).sum().item()
            for _, _, label in val_loader))

        sched.step(va_loss)
        marker = " ✓" if va_loss < best_val else ""
        print(f"{epoch:>6} {tr_loss:>12.5f} {va_loss:>12.5f}{marker}")

        if va_loss < best_val:
            best_val   = va_loss
            best_state = {k: v.clone()
                          for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save({'state_dict': best_state,
                'feats':      feats,
                'scaler':     scaler}, save_path)
    print(f"\nBest val MSE: {best_val:.5f} → saved {save_path}")

    # ── Threshold on train set ────────────────────────────────────────
    print("\nComputing per-sensor thresholds on training set...")
    model.eval()
    all_errors = []  # [N, 51]
    with torch.no_grad():
        for x, y_next, label in train_loader:
            normal = label == 0
            if normal.sum() == 0:
                continue
            x      = x[normal].to(DEVICE)
            y_next = y_next[normal].to(DEVICE)
            pred   = model(x)
            err    = per_sensor_error(y_next, pred)  # [batch, 51]
            all_errors.append(err.cpu().numpy())

    all_errors = np.vstack(all_errors)  # [N, 51]
    # Per-sensor threshold at 99th percentile
    thresh_per_sensor = np.percentile(all_errors, 99, axis=0)  # [51]
    # Global threshold (mean of per-sensor)
    thresh_global     = np.percentile(
        all_errors.mean(axis=1), 99)

    print(f"  Global threshold (p99): {thresh_global:.5f}")
    print(f"  Per-sensor thresholds: min={thresh_per_sensor.min():.5f},"
          f" max={thresh_per_sensor.max():.5f}")

    # ── Val threshold search ──────────────────────────────────────────
    print("\nSearching optimal threshold on val set...")
    val_scores, val_labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y_next, label in val_loader:
            x      = x.to(DEVICE)
            y_next = y_next.to(DEVICE)
            pred   = model(x)
            score  = anomaly_score(y_next, pred).cpu().numpy()
            val_scores.extend(score)
            val_labels.extend(label.numpy())

    val_scores = np.array(val_scores)
    val_labels = np.array(val_labels)

    best_f1, best_thresh = 0, thresh_global
    for p in np.linspace(50, 99.9, 200):
        t   = np.percentile(val_scores, p)
        prd = (val_scores > t).astype(int)
        f1  = f1_score(val_labels, prd, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    print(f"  Best val threshold: {best_thresh:.5f} (F1={best_f1:.3f})")

    # ── Test evaluation ───────────────────────────────────────────────
    print("\nEvaluating on test set...")
    test_scores, test_labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y_next, label in test_loader:
            x      = x.to(DEVICE)
            y_next = y_next.to(DEVICE)
            pred   = model(x)
            score  = anomaly_score(y_next, pred).cpu().numpy()
            test_scores.extend(score)
            test_labels.extend(label.numpy())

    test_scores = np.array(test_scores)
    test_labels = np.array(test_labels)
    test_preds  = (test_scores > best_thresh).astype(int)

    print(f"\n{'Metric':<20} {'Value':>10}")
    print("─" * 32)
    print(f"{'AUROC':<20} {roc_auc_score(test_labels, test_scores):>10.4f}")
    print(f"{'F1 (val-thresh)':<20} {f1_score(test_labels, test_preds, zero_division=0):>10.4f}")
    print(f"{'Precision':<20} {precision_score(test_labels, test_preds, zero_division=0):>10.4f}")
    print(f"{'Recall':<20} {recall_score(test_labels, test_preds, zero_division=0):>10.4f}")

    import json
    with open('optimal_threshold_pred.json', 'w') as f:
        json.dump({
            'threshold': float(best_thresh),
            'val_f1':    float(best_f1),
            'thresh_per_sensor': thresh_per_sensor.tolist(),
            'feats':     list(feats),
        }, f)
    print("\n✅ Saved optimal_threshold_pred.json")
    print("   Next: python shap_predictor.py")

    return model, feats, scaler, best_thresh, thresh_per_sensor


if __name__ == '__main__':
    train_predictor(
        normal_csv='/kaggle/input/datasets/naimparvezz/swat-dataset/SWaT_Dataset_Normal_v1.csv',
        attack_csv='/kaggle/input/datasets/naimparvezz/swat-dataset/SWaT_Dataset_Attack_v0.csv',
        epochs=40, batch=512, lr=1e-3,
        device='cuda',
        save_path='lstm_predictor_best.pt'
    )