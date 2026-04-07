"""
shap_predictor.py
=================
UQ-SHAP localization using prediction-based LSTM.

Key difference from shap_explainer_v2.py:
  The model outputs predicted x(t+1) — a 51-dim vector.
  SHAP function: f(X_window) = anomaly_score = MSE(y_true, y_pred)
  
  Because the model predicts per-sensor, SHAP can attribute
  which INPUT sensors caused the prediction error to spike.
  This gives clean localization signal.
"""

import torch
import numpy as np
import pandas as pd
import shap
import json
import os
from swat_loader import load_swat, ATTACK_MAP
from lstm_predictor import LSTMPredictor, SWaTPredictor, enable_mc_dropout

os.makedirs('results', exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
T      = 20
WINDOW = 20

# ── Load ──────────────────────────────────────────────────────────────
train_df, val_df, test_df, scaler, feats = load_swat(
    normal_csv='/kaggle/input/datasets/naimparvezz/swat-dataset/SWaT_Dataset_Normal_v1.csv',
    attack_csv='/kaggle/input/datasets/naimparvezz/swat-dataset/SWaT_Dataset_Attack_v0.csv',
    verbose=False
)

ckpt  = torch.load('lstm_predictor_best.pt', map_location=DEVICE)
model = LSTMPredictor(n_features=51).to(DEVICE)
model.load_state_dict(ckpt['state_dict'])
model.eval()
print("Prediction model loaded.")

thresh_info = json.load(open('optimal_threshold_pred.json'))
threshold   = thresh_info['threshold']

# ── Mean baseline (one timestep × 51 sensors) ─────────────────────────
# For prediction model: background is one window of mean values
train_mean  = train_df[feats].mean().values.astype(np.float32)  # [51]
bg_window   = np.tile(train_mean, (WINDOW, 1))                   # [20, 51]
bg_flat     = bg_window.reshape(1, -1)                            # [1, 1020]

# ── SHAP predict fn ───────────────────────────────────────────────────
def make_predict_fn(model, device, y_true_next, use_mc=True):
    """
    f(X_flat) → anomaly_score scalar
    y_true_next: [51] — the actual next timestep values
    """
    y_true = torch.tensor(y_true_next, dtype=torch.float32).to(device)

    def predict_fn(X_flat):
        X    = torch.tensor(
            X_flat.reshape(-1, WINDOW, 51),
            dtype=torch.float32).to(device)
        if use_mc:
            enable_mc_dropout(model)
        else:
            model.eval()
        with torch.no_grad():
            pred  = model(X)                            # [batch, 51]
            y_rep = y_true.unsqueeze(0).expand_as(pred) # [batch, 51]
            score = ((y_rep - pred)**2).mean(dim=1)      # [batch]
        return score.cpu().numpy()

    return predict_fn


# ── Max-error window selection ────────────────────────────────────────
def get_max_error_window_pred(attack_id, test_df, feats, model,
                               device, window_size=WINDOW):
    info = ATTACK_MAP.get(attack_id)
    if info is None:
        return None, None, None, None

    mask = ((test_df['Timestamp'] >= info['start']) &
            (test_df['Timestamp'] <= info['end']) &
            (test_df['label'] == 1))
    rows = test_df[mask].reset_index(drop=True)

    if len(rows) < window_size + 1:
        return None, None, None, None

    best_err, best_X, best_y, best_ts = -1, None, None, None
    n_windows = len(rows) - window_size
    positions = np.linspace(0, n_windows - 1,
                            min(50, n_windows), dtype=int)

    model.eval()
    with torch.no_grad():
        for pos in positions:
            window = rows.iloc[pos:pos + window_size]
            y_next = rows.iloc[pos + window_size]
            if len(window) < window_size:
                continue
            X = torch.tensor(
                window[feats].values,
                dtype=torch.float32).unsqueeze(0).to(device)
            Y = torch.tensor(
                y_next[feats].values.astype(np.float32)).to(device)
            pred  = model(X)
            err   = ((Y - pred[0])**2).mean().item()
            if err > best_err:
                best_err = err
                best_X   = X.cpu()
                best_y   = Y.cpu().numpy()
                best_ts  = window.iloc[-1]['Timestamp']

    return best_X, best_y, best_ts, best_err


# ── Main SHAP loop ────────────────────────────────────────────────────
test_attack_ids = sorted([
    aid for aid in ATTACK_MAP
    if ATTACK_MAP[aid]['start'] > val_df['Timestamp'].max()
])
print(f"\n{len(test_attack_ids)} attacks | T={T}\n")

results = []

for aid in test_attack_ids:
    info           = ATTACK_MAP[aid]
    primary_sensor = info['primary']

    print(f"Attack {aid:2d} | {primary_sensor:<10} | {info['type']:<12}",
          end=" | ")

    X_win, y_next, ts, max_err = get_max_error_window_pred(
        aid, test_df, feats, model, DEVICE, WINDOW)

    if X_win is None:
        print("SKIPPED")
        continue

    X_flat = X_win.numpy().reshape(1, -1)

    # Run SHAP T times
    all_shap = []
    for t in range(T):
        fn         = make_predict_fn(model, DEVICE, y_next, use_mc=True)
        explainer  = shap.KernelExplainer(fn, bg_flat)
        shap_vals  = explainer.shap_values(X_flat, nsamples=256,
                                            silent=True)
        # shap_vals: [1, 1020] → [WINDOW, 51] → sum last 5 timesteps
        shap_2d    = shap_vals[0].reshape(WINDOW, 51)
        sensor_sh  = shap_2d[-5:].sum(axis=0)  # [51]
        all_shap.append(sensor_sh)

    all_shap  = np.array(all_shap)         # [T, 51]
    mean_shap = all_shap.mean(axis=0)
    std_shap  = all_shap.std(axis=0)

    abs_mean = np.abs(mean_shap)
    ranking  = np.argsort(-abs_mean)
    feats_l  = list(feats)

    top1 = feats_l[ranking[0]]
    top3 = [feats_l[i] for i in ranking[:3]]
    top5 = [feats_l[i] for i in ranking[:5]]

    pidx         = feats_l.index(primary_sensor) \
                   if primary_sensor in feats_l else -1
    top1_correct = (top1 == primary_sensor)
    top3_correct = (primary_sensor in top3)
    top5_correct = (primary_sensor in top5)
    rank         = int(np.where(ranking == pidx)[0][0]) + 1 \
                   if pidx >= 0 else -1
    mrr          = 1.0 / rank if rank > 0 else 0.0

    std_ranking  = np.argsort(-np.abs(all_shap[0]))
    std_top1     = feats_l[std_ranking[0]]
    std_correct  = (std_top1 == primary_sensor)

    uq = "✅" if top1_correct else "❌"
    st = "✅" if std_correct  else "❌"
    print(f"err={max_err:.3f} | UQ:{uq}({top1}) Std:{st}({std_top1})"
          f" rank={rank}")

    results.append({
        'attack_id':        aid,
        'primary_sensor':   primary_sensor,
        'attack_type':      info['type'],
        'category':         info['category'],
        'max_err':          max_err,
        'uq_top1':          top1,
        'uq_top1_correct':  top1_correct,
        'uq_top3_correct':  top3_correct,
        'uq_top5_correct':  top5_correct,
        'uq_rank':          rank,
        'uq_mrr':           mrr,
        'std_top1':         std_top1,
        'std_top1_correct': std_correct,
        'mean_shap_primary':float(mean_shap[pidx]) if pidx >= 0 else None,
        'std_shap_primary': float(std_shap[pidx])  if pidx >= 0 else None,
        'mean_shap_all':    mean_shap.tolist(),
        'std_shap_all':     std_shap.tolist(),
    })

# ── Summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("RESULTS — Prediction LSTM + UQ-SHAP")
print("=" * 65)
n = len(results)
if n:
    uq1  = sum(r['uq_top1_correct']  for r in results) / n
    uq3  = sum(r['uq_top3_correct']  for r in results) / n
    uq5  = sum(r['uq_top5_correct']  for r in results) / n
    umrr = sum(r['uq_mrr']           for r in results) / n
    s1   = sum(r['std_top1_correct'] for r in results) / n

    print(f"\n  {'Metric':<28} {'Std SHAP':>12} {'UQ-SHAP':>12}")
    print(f"  {'─'*28} {'─'*12} {'─'*12}")
    print(f"  {'Top-1':<28} {s1:>11.1%} {uq1:>11.1%}")
    print(f"  {'Top-3':<28} {'—':>12} {uq3:>11.1%}")
    print(f"  {'Top-5':<28} {'—':>12} {uq5:>11.1%}")
    print(f"  {'MRR':<28} {'—':>12} {umrr:>11.3f}")
    print(f"\n  Baseline (Homaei 2026): 45.2% Top-1")
    print(f"  Gap:                    {uq1-0.452:+.1%}")

    for atype in ['spoof', 'actuator', 'multi']:
        sub = [r for r in results if r['attack_type'] == atype]
        if sub:
            acc = sum(r['uq_top1_correct'] for r in sub) / len(sub)
            print(f"  {atype:<12}: {acc:.1%} ({len(sub)})")

with open('results/shap_pred_results.json', 'w') as f:
    json.dump(results, f, indent=2)
rows = [{k: v for k, v in r.items()
         if k not in ('mean_shap_all','std_shap_all')} for r in results]
pd.DataFrame(rows).to_csv('results/shap_pred_results.csv', index=False)
print("\n✅ Saved results/shap_pred_results.json + .csv")