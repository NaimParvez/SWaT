"""
Unsupervised SWaT training script (LSTM Autoencoder + MC Dropout uncertainty).

What this script provides:
1) DataLoader creation via SWaTDataset from code.py
2) LSTM autoencoder trained only on the normal training split
3) MC Dropout inference for uncertainty estimation
4) Thresholding and anomaly metrics on validation/test
5) CSV export of per-sample error/uncertainty/label
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import random
import sys
import sysconfig
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _ensure_stdlib_code_module_loaded() -> None:
    """Prevent local swat_loader.py from shadowing stdlib code during torch import."""
    mod = sys.modules.get("swat_loader")
    if mod is not None and getattr(mod, "__file__", "") and "\\lib\\swat_loader.py" in str(mod.__file__).lower():
        return

    stdlib_dir = sysconfig.get_paths().get("stdlib")
    if not stdlib_dir:
        return

    code_path = pathlib.Path(stdlib_dir) / "swat_loader.py"
    if not code_path.exists():
        return

    spec = importlib.util.spec_from_file_location("swat_loader", code_path)
    if spec is None or spec.loader is None:
        return

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["swat_loader"] = module


_ensure_stdlib_code_module_loaded()

import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader


def _load_local_swat_module():
    script_dir = pathlib.Path(__file__).resolve().parent
    module_path = script_dir / "swat_loader.py"
    spec = importlib.util.spec_from_file_location("swat_loader_local", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load SWaT module from: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_swat = _load_local_swat_module()
SWaTDataset = _swat.SWaTDataset
load_swat = _swat.load_swat


@dataclass
class Config:
    normal_csv: str = "SWaT_Dataset_Normal_v1.csv"
    attack_csv: str = "SWaT_Dataset_Attack_v0.csv"
    batch_size: int = 256
    epochs: int = 12
    lr: float = 1e-3
    weight_decay: float = 1e-5
    window_size: int = 20
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    seed: int = 42
    num_workers: int = 0
    mc_passes: int = 30
    threshold_percentile: float = 95.0
    save_path: str = "lstm_autoencoder.pt"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_swat(batch):
    """Keep metadata as a list to avoid default-collate issues with objects."""
    x, y, meta = zip(*batch)
    return torch.stack(x), torch.stack(y), list(meta)


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.latent_dropout = nn.Dropout(dropout)
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.output_proj = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        _, (h_n, c_n) = self.encoder(x)
        context = self.latent_dropout(h_n[-1])

        # Repeat latent context across timesteps and decode to input space.
        dec_in = context.unsqueeze(1).repeat(1, x.size(1), 1)
        dec_out, _ = self.decoder(dec_in, (h_n, c_n))
        recon = self.output_proj(dec_out)
        return recon


def mc_dropout_inference(
    model: nn.Module,
    x: torch.Tensor,
    T: int = 30,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform T stochastic forward passes with dropout enabled.

    Returns
    -------
    mean_reconstruction : torch.Tensor [B, W, F]
    std_reconstruction  : torch.Tensor [B, W, F]
    """
    was_training = model.training
    model.train()  # Intentionally keep dropout active for MC Dropout.

    preds = []
    with torch.no_grad():
        for _ in range(T):
            preds.append(model(x).unsqueeze(0))

    stacked = torch.cat(preds, dim=0)  # [T, B, W, F]
    mean_reconstruction = stacked.mean(dim=0)
    std_reconstruction = stacked.std(dim=0, unbiased=False)

    if not was_training:
        model.eval()

    return mean_reconstruction, std_reconstruction


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mc_passes: int = 30,
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    mean_recon_error : float
    metrics          : dict with AUROC/AUPRC (from reconstruction error)
    errors_np        : per-sample reconstruction error
    uncertainty_np   : per-sample uncertainty (mean reconstruction std)
    labels_np        : per-sample labels
    """
    model.eval()
    total_error_sum = 0.0
    total_samples = 0
    all_errors: List[np.ndarray] = []
    all_uncertainty: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for x, y, _ in loader:
        x = x.to(device)
        y_np = y.cpu().numpy()

        mean_recon, std_recon = mc_dropout_inference(model, x, T=mc_passes)
        per_sample_error = ((x - mean_recon) ** 2).mean(dim=(1, 2))
        per_sample_uncertainty = std_recon.mean(dim=(1, 2))

        bs = x.size(0)
        total_error_sum += per_sample_error.sum().item()
        total_samples += bs

        all_errors.append(per_sample_error.detach().cpu().numpy())
        all_uncertainty.append(per_sample_uncertainty.detach().cpu().numpy())
        all_labels.append(y_np)

    errors_np = np.concatenate(all_errors) if all_errors else np.array([])
    uncertainty_np = np.concatenate(all_uncertainty) if all_uncertainty else np.array([])
    labels_np = np.concatenate(all_labels) if all_labels else np.array([])

    metrics: Dict[str, float] = {}
    if labels_np.size and len(np.unique(labels_np)) > 1:
        metrics["auroc"] = float(roc_auc_score(labels_np, errors_np))
        metrics["auprc"] = float(average_precision_score(labels_np, errors_np))
    else:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")

    mean_recon_error = total_error_sum / max(total_samples, 1)
    return mean_recon_error, metrics, errors_np, uncertainty_np, labels_np


def validation_threshold(errors: np.ndarray, percentile: float = 95.0) -> float:
    """Threshold = percentile(validation_error), computed on validation only."""
    errors = np.asarray(errors).reshape(-1)
    if errors.size == 0:
        return 0.0
    p = float(np.clip(percentile, 0.0, 100.0))
    return float(np.percentile(errors, p))


def normalize_errors(errors: np.ndarray, mean_error: float, std_error: float) -> np.ndarray:
    """Z-normalize reconstruction errors with train-set statistics."""
    errors = np.asarray(errors).reshape(-1)
    return (errors - mean_error) / (std_error + 1e-8)


def print_percentile_f1_grid(errors: np.ndarray, labels: np.ndarray) -> None:
    """Optional quick comparison across common percentile thresholds."""
    errors = np.asarray(errors).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if errors.size == 0 or labels.size == 0:
        return

    print("Validation F1 by percentile threshold:")
    for p in [90, 95, 97, 99]:
        thr = float(np.percentile(errors, p))
        pred = (errors > thr).astype(np.int64)
        f1 = f1_score(labels, pred, zero_division=0)
        print(f"  p{p}: threshold={thr:.6f}, F1={f1:.4f}")


def log_uncertainty_stats(
    split_name: str,
    errors: np.ndarray,
    uncertainty: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Print uncertainty summary and error-uncertainty Spearman correlation."""
    normal_mask = labels == 0
    attack_mask = labels == 1

    normal_unc = uncertainty[normal_mask] if normal_mask.any() else np.array([])
    attack_unc = uncertainty[attack_mask] if attack_mask.any() else np.array([])

    normal_mean = float(normal_unc.mean()) if normal_unc.size else float("nan")
    attack_mean = float(attack_unc.mean()) if attack_unc.size else float("nan")

    print(
        f"{split_name} uncertainty mean | "
        f"normal={normal_mean:.6f}, attack={attack_mean:.6f}"
    )

    if errors.size > 1 and uncertainty.size > 1:
        corr, _ = spearmanr(errors, uncertainty)
        print(f"{split_name} Uncertainty-Error Correlation: {corr:.4f}")
    else:
        print(f"{split_name} Uncertainty-Error Correlation: nan")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x, _, _ in loader:
        x = x.to(device)

        optimizer.zero_grad(set_to_none=True)
        recon = model(x)
        loss = ((x - recon) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

    return total_loss / max(total_samples, 1)


def build_dataloaders(cfg: Config):
    train_df, val_df, test_df, _, features = load_swat(
        normal_csv=cfg.normal_csv,
        attack_csv=cfg.attack_csv,
        window_size=cfg.window_size,
        verbose=True,
    )

    print("\n[Info] Unsupervised mode: training uses normal train split only.")

    train_ds = SWaTDataset(train_df, features, window_size=cfg.window_size, label_strategy="last")
    val_ds = SWaTDataset(val_df, features, window_size=cfg.window_size, label_strategy="last")
    test_ds = SWaTDataset(test_df, features, window_size=cfg.window_size, label_strategy="last")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_swat,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_swat,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_swat,
    )

    return train_loader, val_loader, test_loader, features


def main(cfg: Config) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mc_passes = max(1, int(cfg.mc_passes))

    print(f"\nUsing device: {device}")
    print(f"MC Dropout passes: {mc_passes}")

    train_loader, val_loader, test_loader, features = build_dataloaders(cfg)
    print(
        f"\nDataloaders ready: "
        f"train={len(train_loader.dataset):,}, "
        f"val={len(val_loader.dataset):,}, "
        f"test={len(test_loader.dataset):,}, "
        f"features={len(features)}"
    )

    model = LSTMAutoencoder(
        input_size=len(features),
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    best_val_auprc = -1.0
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_metrics, _, _, _ = evaluate(
            model,
            val_loader,
            device,
            mc_passes=mc_passes,
        )

        val_auroc = val_metrics.get("auroc", float("nan"))
        val_auprc = val_metrics.get("auprc", float("nan"))

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"train_loss={train_loss:.6f} | val_recon_error={val_loss:.6f} | "
            f"val_auroc={val_auroc:.4f} | val_auprc={val_auprc:.4f}"
        )

        score = val_auprc
        if np.isnan(score):
            score = -1.0
        if score > best_val_auprc:
            best_val_auprc = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "features": features,
            "window_size": cfg.window_size,
            "hidden_size": cfg.hidden_size,
            "num_layers": cfg.num_layers,
            "dropout": cfg.dropout,
            "mc_passes": mc_passes,
        },
        cfg.save_path,
    )
    print(f"\nSaved best model to: {cfg.save_path}")

    # Compute train reconstruction-error statistics for cross-split normalization.
    _, _, train_errors_raw, _, _ = evaluate(
        model,
        train_loader,
        device,
        mc_passes=mc_passes,
    )
    mean_train_error = float(np.mean(train_errors_raw)) if train_errors_raw.size else 0.0
    std_train_error = float(np.std(train_errors_raw)) if train_errors_raw.size else 1.0
    print(
        f"Train error stats for normalization: "
        f"mean={mean_train_error:.6f}, std={std_train_error:.6f}"
    )

    # Validation threshold from reconstruction error.
    _, _, val_errors_raw, val_uncertainty, val_labels = evaluate(
        model,
        val_loader,
        device,
        mc_passes=mc_passes,
    )
    val_errors = normalize_errors(val_errors_raw, mean_train_error, std_train_error)
    threshold = validation_threshold(
        val_errors,
        percentile=cfg.threshold_percentile,
    )

    val_metrics = {}
    if val_labels.size and len(np.unique(val_labels)) > 1:
        val_metrics["auroc"] = float(roc_auc_score(val_labels, val_errors))
        val_metrics["auprc"] = float(average_precision_score(val_labels, val_errors))
    else:
        val_metrics["auroc"] = float("nan")
        val_metrics["auprc"] = float("nan")

    print(
        f"Threshold ({cfg.threshold_percentile:.1f}th percentile): {threshold:.6f}\n"
        f"Mean val error: {val_errors.mean():.6f}\n"
        f"Std val error: {val_errors.std():.6f}"
    )
    print_percentile_f1_grid(val_errors, val_labels)

    val_pred = (val_errors > threshold).astype(np.int64)
    val_f1 = f1_score(val_labels, val_pred, zero_division=0)
    log_uncertainty_stats("Validation", val_errors, val_uncertainty, val_labels)

    # Final test metrics.
    test_loss, _, test_errors_raw, test_uncertainty, test_labels = evaluate(
        model,
        test_loader,
        device,
        mc_passes=mc_passes,
    )

    test_errors = normalize_errors(test_errors_raw, mean_train_error, std_train_error)
    test_metrics = {}
    if test_labels.size and len(np.unique(test_labels)) > 1:
        test_metrics["auroc"] = float(roc_auc_score(test_labels, test_errors))
        test_metrics["auprc"] = float(average_precision_score(test_labels, test_errors))
    else:
        test_metrics["auroc"] = float("nan")
        test_metrics["auprc"] = float("nan")

    test_pred = (test_errors > threshold).astype(np.int64)
    test_f1 = f1_score(test_labels, test_pred, zero_division=0)
    log_uncertainty_stats("Test", test_errors, test_uncertainty, test_labels)

    results_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "split": "val",
                    "reconstruction_error": val_errors,
                    "uncertainty": val_uncertainty,
                    "label": val_labels.astype(int),
                }
            ),
            pd.DataFrame(
                {
                    "split": "test",
                    "reconstruction_error": test_errors,
                    "uncertainty": test_uncertainty,
                    "label": test_labels.astype(int),
                }
            ),
        ],
        ignore_index=True,
    )
    results_df.to_csv("results_uncertainty.csv", index=False)
    print("Saved uncertainty outputs to: results_uncertainty.csv")

    print("\nFinal metrics")
    print(
        "Validation: "
        f"threshold={threshold:.6f}, "
        f"AUROC={val_metrics['auroc']:.4f}, "
        f"AUPRC={val_metrics['auprc']:.4f}, "
        f"F1={val_f1:.4f}"
    )
    print(
        "Test: "
        f"recon_error={test_loss:.6f}, "
        f"AUROC={test_metrics['auroc']:.4f}, "
        f"AUPRC={test_metrics['auprc']:.4f}, "
        f"F1@val_thr={test_f1:.4f}"
    )


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Unsupervised SWaT LSTM Autoencoder trainer")
    p.add_argument("--normal-csv", default="SWaT_Dataset_Normal_v1.csv")
    p.add_argument("--attack-csv", default="SWaT_Dataset_Attack_v0.csv")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--window-size", type=int, default=20)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--mc-passes", type=int, default=30)
    p.add_argument("--threshold-percentile", type=float, default=95.0)
    p.add_argument("--save-path", default="lstm_autoencoder.pt")

    a = p.parse_args()
    return Config(
        normal_csv=a.normal_csv,
        attack_csv=a.attack_csv,
        batch_size=a.batch_size,
        epochs=a.epochs,
        lr=a.lr,
        weight_decay=a.weight_decay,
        window_size=a.window_size,
        hidden_size=a.hidden_size,
        num_layers=a.num_layers,
        dropout=a.dropout,
        seed=a.seed,
        num_workers=a.num_workers,
        mc_passes=a.mc_passes,
        threshold_percentile=a.threshold_percentile,
        save_path=a.save_path,
    )


if __name__ == "__main__":
    main(parse_args())
