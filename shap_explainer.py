"""
shap_explainer.py
=================
Uncertainty-aware SHAP for SWaT anomaly localization.

For each attack in ATTACK_MAP (test set only):
  1. Extract the middle window of the attack period
  2. Run KernelSHAP T=20 times (one per MC Dropout pass)
  3. Collect φᵢ distribution per sensor
  4. Compute mean(φᵢ) ± std(φᵢ)
  5. Check: argmax(|mean φᵢ|) == attacked_sensor?

Output: results/shap_results.json + results/shap_results.csv
"""

import torch
import numpy as np
import pandas as pd
import shap
import json
import os
from torch.utils.data import DataLoader
from swat_loader import load_swat, SWaTDataset, ATTACK_MAP, ALL_FEATURES
from lstm_model import LSTMAutoencoder, enable_mc_dropout

os.makedirs('results', exist_ok=True)

# ── Config ──────────────────────────────────────────────────────────
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
T        = 20        # MC Dropout passes
N_BG     = 50        # KernelSHAP background samples (keep small = faster)
WINDOW   = 20

print(f"Device: {DEVICE}")
print(f"MC passes: T={T}")

# ── Load data and model ──────────────────────────────────────────────
print("\nLoading data...")
train_df, val_df, test_df, scaler, feats = load_swat(
    normal_csv='SWaT_Dataset_Normal_v1.csv',
    attack_csv='SWaT_Dataset_Attack_v0.csv',
    verbose=False
)

model = LSTMAutoencoder(n_features=51).to(DEVICE)
model.load_state_dict(torch.load('lstm_best.pt', map_location=DEVICE))
model.eval()
print("Model loaded.")

# ── Build background dataset ─────────────────────────────────────────
# Use N_BG random normal windows from training set as SHAP background
print(f"\nBuilding background ({N_BG} normal train windows)...")

train_ds = SWaTDataset(train_df, feats, WINDOW)
bg_indices = np.random.choice(
    [i for i in range(len(train_ds))], size=N_BG, replace=False)
bg_windows = torch.stack([train_ds[i][0] for i in bg_indices])
# Shape: [N_BG, WINDOW, 51]
# Flatten for KernelSHAP: [N_BG, WINDOW*51]
bg_flat = bg_windows.numpy().reshape(N_BG, -1)
print(f"Background shape: {bg_flat.shape}")

# ── SHAP wrapper function ────────────────────────────────────────────
def make_shap_fn(model, device, use_mc=True):
    """
    Returns a function f(X_flat) → anomaly_scores
    that KernelSHAP can call.
    X_flat: [n_samples, WINDOW*51]
    """
    def predict_fn(X_flat):
        X = torch.tensor(
            X_flat.reshape(-1, WINDOW, 51),
            dtype=torch.float32
        ).to(device)

        if use_mc:
            enable_mc_dropout(model)
        else:
            model.eval()

        with torch.no_grad():
            x_hat = model(X)
            # Reconstruction error per sample
            err = torch.sqrt(((X - x_hat)**2).mean(dim=[1, 2]))

        return err.cpu().numpy()

    return predict_fn

# ── Get one window for a given attack ───────────────────────────────
def get_attack_window(attack_id, test_df, feats, window_size=WINDOW):
    """
    Extract the middle window of an attack period from test_df.
    Returns tensor [1, window, features] or None if not in test set.
    """
    info = ATTACK_MAP.get(attack_id)
    if info is None:
        return None, None

    # Find rows in this attack window
    mask = ((test_df['Timestamp'] >= info['start']) &
            (test_df['Timestamp'] <= info['end']) &
            (test_df['label'] == 1))
    attack_rows = test_df[mask]

    if len(attack_rows) < window_size:
        return None, None

    # Take the middle of the attack period
    mid_idx = len(attack_rows) // 2
    mid_ts  = attack_rows.iloc[mid_idx]['Timestamp']

    # Find this timestamp in test_df
    ts_idx = test_df[test_df['Timestamp'] == mid_ts].index
    if len(ts_idx) == 0:
        return None, None

    # Get integer position in test_df
    pos = test_df.index.get_loc(ts_idx[0])
    start_pos = max(0, pos - window_size + 1)

    window_data = test_df.iloc[start_pos:start_pos + window_size]
    if len(window_data) < window_size:
        return None, None

    X = torch.tensor(
        window_data[feats].values, dtype=torch.float32
    ).unsqueeze(0)  # [1, window, 51]

    return X, mid_ts


# ── Main SHAP loop ───────────────────────────────────────────────────
results = []
skipped = []

# Only process attacks in the TEST set (attacks 13-41)
test_attack_ids = [aid for aid in ATTACK_MAP
                   if ATTACK_MAP[aid]['start'] > val_df['Timestamp'].max()]

print(f"\nAttacks to explain: {test_attack_ids}")
print(f"Total: {len(test_attack_ids)} attacks × T={T} passes "
      f"= {len(test_attack_ids)*T} SHAP calls\n")

for aid in test_attack_ids:
    info = ATTACK_MAP[aid]
    primary_sensor = info['primary']
    attack_type    = info['type']
    category       = info['category']

    print(f"Attack {aid:2d} | {primary_sensor:<10} | {attack_type:<12}", end=" ")

    # Get the attack window
    X_window, mid_ts = get_attack_window(aid, test_df, feats, WINDOW)
    if X_window is None:
        print("→ SKIPPED (not in test set or too short)")
        skipped.append(aid)
        continue

    # Flatten for SHAP: [1, WINDOW*51]
    X_flat = X_window.numpy().reshape(1, -1)

    # ── Run SHAP T times (one per MC pass) ──────────────────────────
    all_shap_values = []  # will be [T, 51] — one SHAP per sensor per pass

    for t in range(T):
        # Each call uses a different dropout mask
        predict_fn = make_shap_fn(model, DEVICE, use_mc=True)
        explainer  = shap.KernelExplainer(predict_fn, bg_flat)

        # nsamples controls accuracy vs speed trade-off
        # 'auto' = 2*n_features+2048, too slow for 1020 features
        # Use 256 for reasonable accuracy
        shap_vals = explainer.shap_values(X_flat, nsamples=256, silent=True)
        # shap_vals: [1, WINDOW*51] — one value per (timestep, feature)

        # Aggregate over the time dimension: sum |SHAP| per sensor
        # Reshape to [WINDOW, 51], then sum over time axis
        shap_2d = shap_vals[0].reshape(WINDOW, 51)
        # Per-sensor importance: sum of absolute SHAP over all timesteps
        sensor_shap = shap_2d.sum(axis=0)  # [51]
        all_shap_values.append(sensor_shap)

    # ── Compute uncertainty-aware SHAP statistics ────────────────────
    all_shap_values = np.array(all_shap_values)  # [T, 51]
    mean_shap = all_shap_values.mean(axis=0)      # [51]
    std_shap  = all_shap_values.std(axis=0)       # [51]
    ci_low    = np.percentile(all_shap_values, 5,  axis=0)  # [51]
    ci_high   = np.percentile(all_shap_values, 95, axis=0)  # [51]

    # ── Localization evaluation ──────────────────────────────────────
    # Use absolute mean SHAP for ranking
    abs_mean  = np.abs(mean_shap)
    ranking   = np.argsort(-abs_mean)  # descending

    feature_names = feats  # list of 51 sensor names
    top1_sensor   = feature_names[ranking[0]]
    top3_sensors  = [feature_names[i] for i in ranking[:3]]
    top5_sensors  = [feature_names[i] for i in ranking[:5]]

    # Normalize sensor name for comparison (both should be like 'LIT101')
    top1_correct = (top1_sensor == primary_sensor)
    top3_correct = (primary_sensor in top3_sensors)
    top5_correct = (primary_sensor in top5_sensors)

    # MRR: find rank of primary sensor (1-indexed)
    primary_idx = feature_names.index(primary_sensor) \
                  if primary_sensor in feature_names else -1
    if primary_idx >= 0:
        rank = np.where(ranking == primary_idx)[0][0] + 1
        mrr  = 1.0 / rank
    else:
        rank = -1
        mrr  = 0.0

    # Standard SHAP (no MC): use first pass as baseline
    std_shap_vals = all_shap_values[0]  # first pass = "standard SHAP"
    std_ranking   = np.argsort(-np.abs(std_shap_vals))
    std_top1      = feature_names[std_ranking[0]]
    std_top1_correct = (std_top1 == primary_sensor)

    result = {
        'attack_id':       aid,
        'primary_sensor':  primary_sensor,
        'attack_type':     attack_type,
        'category':        category,
        'timestamp':       str(mid_ts),
        # UQ-SHAP localization
        'uq_top1':         top1_sensor,
        'uq_top1_correct': top1_correct,
        'uq_top3_correct': top3_correct,
        'uq_top5_correct': top5_correct,
        'uq_rank':         int(rank),
        'uq_mrr':          float(mrr),
        # Standard SHAP localization (first pass = no UQ)
        'std_top1':        std_top1,
        'std_top1_correct':std_top1_correct,
        # Uncertainty metrics
        'mean_shap_primary': float(mean_shap[primary_idx])
                             if primary_idx >= 0 else None,
        'std_shap_primary':  float(std_shap[primary_idx])
                             if primary_idx >= 0 else None,
        'mean_shap_top1':    float(mean_shap[ranking[0]]),
        'std_shap_top1':     float(std_shap[ranking[0]]),
        # Full arrays for later analysis
        'mean_shap_all': mean_shap.tolist(),
        'std_shap_all':  std_shap.tolist(),
    }
    results.append(result)

    status = "✅" if top1_correct else "❌"
    std_status = "✅" if std_top1_correct else "❌"
    print(f"→ UQ:{status} ({top1_sensor}) | Std:{std_status} ({std_top1})"
          f" | rank={rank} | std={std_shap[primary_idx if primary_idx>=0 else 0]:.3f}")

# ── Summary statistics ───────────────────────────────────────────────
print("\n" + "=" * 65)
print("LOCALIZATION RESULTS")
print("=" * 65)

if results:
    n = len(results)
    uq_top1  = sum(r['uq_top1_correct']  for r in results) / n
    uq_top3  = sum(r['uq_top3_correct']  for r in results) / n
    uq_top5  = sum(r['uq_top5_correct']  for r in results) / n
    uq_mrr   = sum(r['uq_mrr']           for r in results) / n
    std_top1 = sum(r['std_top1_correct'] for r in results) / n

    print(f"\n  Attacks evaluated: {n} / {len(test_attack_ids)}")
    print(f"  Skipped:           {len(skipped)}")
    print(f"\n  {'Metric':<30} {'Standard SHAP':>14} {'UQ-SHAP (ours)':>14}")
    print(f"  {'─'*30} {'─'*14} {'─'*14}")
    print(f"  {'Top-1 localization':<30} {std_top1:>13.1%} {uq_top1:>13.1%}")
    print(f"  {'Top-3 localization':<30} {'—':>14} {uq_top3:>13.1%}")
    print(f"  {'Top-5 localization':<30} {'—':>14} {uq_top5:>13.1%}")
    print(f"  {'MRR':<30} {'—':>14} {uq_mrr:>13.3f}")

    print(f"\n  Baseline (Homaei 2026 Table 9): 45.2% Top-1")
    diff = uq_top1 - 0.452
    print(f"  Your UQ-SHAP vs baseline:        {diff:+.1%}")

# ── Save results ─────────────────────────────────────────────────────
# Full JSON
with open('results/shap_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Summary CSV
summary_rows = []
for r in results:
    summary_rows.append({
        'attack_id':       r['attack_id'],
        'primary_sensor':  r['primary_sensor'],
        'attack_type':     r['attack_type'],
        'category':        r['category'],
        'std_top1':        r['std_top1'],
        'std_correct':     r['std_top1_correct'],
        'uq_top1':         r['uq_top1'],
        'uq_correct':      r['uq_top1_correct'],
        'uq_top3':         r['uq_top3_correct'],
        'uq_rank':         r['uq_rank'],
        'uq_mrr':          r['uq_mrr'],
        'std_primary':     r['std_shap_primary'],
        'std_top1_val':    r['std_shap_top1'],
    })

pd.DataFrame(summary_rows).to_csv('results/shap_results.csv', index=False)
print(f"\n✅ Results saved to results/shap_results.json")
print(f"✅ Summary saved to results/shap_results.csv")
print(f"\nNext: python visualize_shap.py to generate paper figures.")