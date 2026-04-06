# threshold_search.py
# Run once after training. Uses saved lstm_best.pt + val set.

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from swat_loader import load_swat, SWaTDataset
from lstm_model import LSTMAutoencoder, mc_predict, reconstruction_error

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load data
train_df, val_df, test_df, scaler, feats = load_swat(
    normal_csv='SWaT_Dataset_Normal_v1.csv',
    attack_csv='SWaT_Dataset_Attack_v0.csv',
    verbose=False
)

val_ds   = SWaTDataset(val_df,  feats, window_size=20)
test_ds  = SWaTDataset(test_df, feats, window_size=20)

val_loader  = DataLoader(val_ds,  batch_size=512, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

# Load model
model = LSTMAutoencoder(n_features=51).to(DEVICE)
model.load_state_dict(torch.load('lstm_best.pt', map_location=DEVICE))
model.eval()

# Collect val scores
val_scores, val_labels = [], []
for x, y, _ in val_loader:
    mean_err, _, _ = mc_predict(model, x, T=20, device=DEVICE)
    val_scores.extend(mean_err.numpy())
    val_labels.extend(y.numpy())

val_scores = np.array(val_scores)
val_labels = np.array(val_labels)

# Search for best threshold on val
print("── Val threshold search ──────────────────────────")
print(f"{'Threshold':>12} {'F1':>8} {'Prec':>8} {'Rec':>8}")
print("─" * 42)

best_f1, best_thresh = 0, 0
candidates = np.percentile(val_scores, np.arange(50, 100, 0.5))
for thresh in candidates:
    preds = (val_scores > thresh).astype(int)
    f1    = f1_score(val_labels, preds, zero_division=0)
    if f1 > best_f1:
        best_f1     = f1
        best_thresh = thresh

# Show a few around the best
for thresh in np.percentile(val_scores, [70, 75, 80, 85, 90, 95]):
    preds = (val_scores > thresh).astype(int)
    f1    = f1_score(val_labels, preds, zero_division=0)
    prec  = precision_score(val_labels, preds, zero_division=0)
    rec   = recall_score(val_labels, preds, zero_division=0)
    marker = " ← best" if abs(thresh - best_thresh) < 0.001 else ""
    print(f"{thresh:>12.4f} {f1:>8.3f} {prec:>8.3f} {rec:>8.3f}{marker}")

print(f"\nBest val threshold: {best_thresh:.4f}  (F1={best_f1:.3f})")

# Apply best threshold to TEST set
test_scores, test_labels = [], []
for x, y, _ in test_loader:
    mean_err, std_err, _ = mc_predict(model, x, T=20, device=DEVICE)
    test_scores.extend(mean_err.numpy())
    test_labels.extend(y.numpy())

test_scores = np.array(test_scores)
test_labels = np.array(test_labels)
test_preds  = (test_scores > best_thresh).astype(int)

from sklearn.metrics import roc_auc_score, average_precision_score
print("\n── Final test results (val-optimized threshold) ─────────")
print(f"  AUROC:     {roc_auc_score(test_labels, test_scores):.4f}")
print(f"  AUPRC:     {average_precision_score(test_labels, test_scores):.4f}")
print(f"  F1:        {f1_score(test_labels, test_preds, zero_division=0):.4f}")
print(f"  Precision: {precision_score(test_labels, test_preds, zero_division=0):.4f}")
print(f"  Recall:    {recall_score(test_labels, test_preds, zero_division=0):.4f}")
print(f"  Threshold: {best_thresh:.4f} (selected on val set)")
print("\n✅ Use this threshold for all subsequent SHAP experiments.")

# Save threshold
import json
with open('optimal_threshold.json', 'w') as f:
    json.dump({'threshold': float(best_thresh),
               'val_f1':    float(best_f1),
               'method':    'val_set_f1_maximization'}, f)
print("   Saved to optimal_threshold.json")