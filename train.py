"""
train.py — Run this to train the baseline LSTM and verify it works.
"""

import torch
from torch.utils.data import DataLoader
from swat_loader import load_swat, SWaTDataset, ALL_FEATURES
from lstm_model import (LSTMAutoencoder, train_epoch, val_epoch,
                         compute_threshold, evaluate)

# ── Config ──────────────────────────────────────────────────────────
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS     = 30
BATCH      = 256
LR         = 1e-3
WINDOW     = 20
T_MC       = 20

print(f"Device: {DEVICE}")

# ── Load data ────────────────────────────────────────────────────────
train_df, val_df, test_df, scaler, feats = load_swat(
    normal_csv='/kaggle/input/datasets/naimparvezz/swat-dataset/SWat_Dataset/SWaT_Dataset_Normal_v1.csv',
    attack_csv='/kaggle/input/datasets/naimparvezz/swat-dataset/SWat_Dataset/SWaT_Dataset_Attack_v0.csv',
    verbose=False
)

train_ds = SWaTDataset(train_df, feats, WINDOW)
val_ds   = SWaTDataset(val_df,   feats, WINDOW)
test_ds  = SWaTDataset(test_df,  feats, WINDOW)

train_loader = DataLoader(train_ds, batch_size=BATCH,
                          shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH,
                          shuffle=False, num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=BATCH,
                          shuffle=False, num_workers=2)

print(f"Train windows: {len(train_ds):,}")
print(f"Val windows:   {len(val_ds):,}")
print(f"Test windows:  {len(test_ds):,}")

# ── Model ────────────────────────────────────────────────────────────
model = LSTMAutoencoder(
    n_features=len(feats),
    hidden=64, latent=16,
    dropout=0.2, window=WINDOW
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=3, factor=0.5, verbose=True)

# ── Training loop ────────────────────────────────────────────────────
best_val_loss = float('inf')
best_state    = None

print(f"\n{'Epoch':>6} {'Train':>10} {'Val':>10}")
print("─" * 30)

for epoch in range(1, EPOCHS + 1):
    tr_loss = train_epoch(model, train_loader, optimizer, DEVICE)
    va_loss = val_epoch(model, val_loader, DEVICE)
    scheduler.step(va_loss)

    marker = " ✓" if va_loss < best_val_loss else ""
    print(f"{epoch:>6} {tr_loss:>10.4f} {va_loss:>10.4f}{marker}")

    if va_loss < best_val_loss:
        best_val_loss = va_loss
        best_state = {k: v.clone() for k, v in
                      model.state_dict().items()}

# ── Restore best and evaluate ────────────────────────────────────────
model.load_state_dict(best_state)
torch.save(best_state, 'lstm_best.pt')
print(f"\nBest val loss: {best_val_loss:.4f} — saved to lstm_best.pt")

# ── Threshold on training set ────────────────────────────────────────
print("\nComputing threshold on training set...")
threshold = compute_threshold(model, train_loader, DEVICE, percentile=95)

# ── Test evaluation ──────────────────────────────────────────────────
print("\nEvaluating on test set (standard mode)...")
metrics_std, _, _, _ = evaluate(
    model, test_loader, threshold, DEVICE, use_mc=False)

print("\nEvaluating on test set (MC Dropout, T=20)...")
metrics_mc, scores, stds, labels = evaluate(
    model, test_loader, threshold, DEVICE, T=T_MC, use_mc=True)

print("\n" + "=" * 50)
print("RESULTS")
print("=" * 50)
print(f"{'Metric':<30} {'Standard':>10} {'MC (T=20)':>10}")
print("─" * 50)
for k in ['auroc', 'auprc', 'f1', 'precision', 'recall']:
    print(f"{k:<30} {metrics_std[k]:>10.4f} {metrics_mc[k]:>10.4f}")

print(f"\n{'Mean uncertainty (normal)':<30} "
      f"{'—':>10} {metrics_mc['mean_uncertainty_normal']:>10.4f}")
print(f"{'Mean uncertainty (attack)':<30} "
      f"{'—':>10} {metrics_mc['mean_uncertainty_attack']:>10.4f}")

print("\n✅ Done. Check AUROC > 0.80 before proceeding to SHAP.")