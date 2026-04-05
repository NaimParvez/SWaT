"""
swat_loader.py
==============
Complete data loading, preprocessing, and feature engineering pipeline
for the SWaT A1/A2 Dec 2015 dataset (official iTrust version).

Usage:
    from swat_loader import load_swat, SWaTDataset, ATTACK_MAP, verify_loader
    train_df, val_df, test_df, scaler = load_swat(
        normal_csv='SWaT_Dataset_Normal_v1.csv',
        attack_csv='SWaT_Dataset_Attack_v0.csv'
    )

References:
    Goh et al. (2017). A dataset to support research in the design of
    secure water treatment systems. CRITIS 2016. DOI: 10.1007/978-3-319-71368-7_8
    
    Homaei et al. (2026). Causal digital twins for cyber-physical security.
    MLWA 23, 100824. DOI: 10.1016/j.mlwa.2025.100824
    (Temporal split protocol: train days 1-5, val days 6-7, test days 8-11)
"""

import pandas as pd
import numpy as np
import importlib.util
import pathlib
import sys
import sysconfig


def _ensure_stdlib_code_module_loaded() -> None:
    """
    Avoid shadowing Python's stdlib `code` module when this file is named
    `swat_loader.py`. This prevents circular imports triggered by torch -> pdb -> code.
    """
    if 'swat_loader' in sys.modules:
        return

    stdlib_dir = sysconfig.get_paths().get('stdlib')
    if not stdlib_dir:
        return

    code_path = pathlib.Path(stdlib_dir) / 'swat_loader.py'
    if not code_path.exists():
        return

    spec = importlib.util.spec_from_file_location('swat_loader', code_path)
    if spec is None or spec.loader is None:
        return

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules['swat_loader'] = module


_ensure_stdlib_code_module_loaded()

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# 1. CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

WINDOW_SIZE = 20  # timesteps per window (validated by Perez Fiadzeawu 2025)

# Temporal split — matches Homaei et al. 2026 for fair comparison
# Normal data: Dec 22-28. Attack data: Dec 28 - Jan 2.
STABILIZATION_CUTOFF = pd.Timestamp('2015-12-22 21:00:00')
# 5 hours from start removed per Goh et al. 2017 (tank stabilization)

TRAIN_END = pd.Timestamp('2015-12-27 23:59:59')   # ~6 days normal
VAL_END   = pd.Timestamp('2015-12-28 23:59:59')   # first attack day as val
# TEST      = everything after VAL_END (attack days 2-4)

# 25 continuous sensors — Z-score normalize
CONTINUOUS_SENSORS = [
    'FIT101', 'LIT101',                                    # P1
    'AIT201', 'AIT202', 'AIT203', 'FIT201',               # P2
    'DPIT301', 'FIT301', 'LIT301',                        # P3
    'AIT401', 'AIT402', 'FIT401', 'LIT401',               # P4
    'AIT501', 'AIT502', 'AIT503', 'AIT504',               # P5
    'FIT501', 'FIT502', 'FIT503', 'FIT504',
    'PIT501', 'PIT502', 'PIT503',
    'FIT601',                                              # P6
]  # 25 sensors

# 26 binary actuators — keep as-is (0/1/2), do NOT normalize
BINARY_ACTUATORS = [
    'MV101', 'P101', 'P102',                              # P1
    'MV201', 'P201', 'P202', 'P203', 'P204', 'P205', 'P206',  # P2
    'MV301', 'MV302', 'MV303', 'MV304', 'P301', 'P302',  # P3
    'P401', 'P402', 'P403', 'P404', 'UV401',              # P4
    'P501', 'P502',                                        # P5
    'P601', 'P602', 'P603',                                # P6
]  # 26 actuators

ALL_FEATURES = CONTINUOUS_SENSORS + BINARY_ACTUATORS  # 51 total

# Sensor type lookup (from SWaT equipment list PDF)
SENSOR_TYPES = {
    'FIT': 'flow',    'LIT': 'level', 'AIT': 'analyser',
    'PIT': 'pressure','DPIT': 'diff_pressure',
    'MV':  'valve',   'P':   'pump',  'UV':  'uv_filter',
}

# Stage mapping
STAGE_MAP = {
    'P1': ['FIT101','LIT101','MV101','P101','P102'],
    'P2': ['AIT201','AIT202','AIT203','FIT201','MV201',
           'P201','P202','P203','P204','P205','P206'],
    'P3': ['DPIT301','FIT301','LIT301','MV301','MV302',
           'MV303','MV304','P301','P302'],
    'P4': ['AIT401','AIT402','FIT401','LIT401',
           'P401','P402','P403','P404','UV401'],
    'P5': ['AIT501','AIT502','AIT503','AIT504',
           'FIT501','FIT502','FIT503','FIT504',
           'P501','P502','PIT501','PIT502','PIT503'],
    'P6': ['FIT601','P601','P602','P603'],
}

# ══════════════════════════════════════════════════════════════════════════════
# 2. OFFICIAL ATTACK MAP (from iTrust list_of_attack.csv)
#    Format: attack_id -> {start, end, primary_sensor, all_sensors, type}
#    Timestamps converted to UTC-aware pandas Timestamps.
#    Note: Attack list uses local Singapore time (UTC+8). 
#    The CSV dataset timestamps are also local time — no conversion needed.
# ══════════════════════════════════════════════════════════════════════════════

ATTACK_MAP = {
    1:  {'start': '2015-12-28 10:29:14', 'end': '2015-12-28 10:44:53',
         'primary': 'MV101',   'all': ['MV101'],
         'type': 'actuator',   'category': 'SSSP'},
    2:  {'start': '2015-12-28 10:51:08', 'end': '2015-12-28 10:58:30',
         'primary': 'P102',    'all': ['P102'],
         'type': 'actuator',   'category': 'SSSP'},
    3:  {'start': '2015-12-28 11:22:00', 'end': '2015-12-28 11:28:22',
         'primary': 'LIT101',  'all': ['LIT101'],
         'type': 'spoof',      'category': 'SSSP'},
    4:  {'start': '2015-12-28 11:47:39', 'end': '2015-12-28 11:54:08',
         'primary': 'MV504',   'all': ['MV504'],
         'type': 'actuator',   'category': 'SSSP'},
    # 5: No physical impact — skip
    6:  {'start': '2015-12-28 12:00:55', 'end': '2015-12-28 12:04:10',
         'primary': 'AIT202',  'all': ['AIT202'],
         'type': 'spoof',      'category': 'SSSP'},
    7:  {'start': '2015-12-28 12:08:25', 'end': '2015-12-28 12:15:33',
         'primary': 'LIT301',  'all': ['LIT301'],
         'type': 'spoof',      'category': 'SSSP'},
    8:  {'start': '2015-12-28 13:10:10', 'end': '2015-12-28 13:26:13',
         'primary': 'DPIT301', 'all': ['DPIT301'],
         'type': 'spoof',      'category': 'SSSP'},
    # 9: No physical impact — skip
    10: {'start': '2015-12-28 14:16:20', 'end': '2015-12-28 14:19:00',
         'primary': 'FIT401',  'all': ['FIT401'],
         'type': 'spoof',      'category': 'SSSP'},
    11: {'start': '2015-12-28 14:19:00', 'end': '2015-12-28 14:28:20',
         'primary': 'FIT401',  'all': ['FIT401'],
         'type': 'spoof',      'category': 'SSSP'},
    # 12: No physical impact — skip
    13: {'start': '2015-12-29 11:11:25', 'end': '2015-12-29 11:15:17',
         'primary': 'MV304',   'all': ['MV304'],
         'type': 'actuator',   'category': 'SSSP'},
    14: {'start': '2015-12-29 11:35:40', 'end': '2015-12-29 11:42:50',
         'primary': 'MV303',   'all': ['MV303'],
         'type': 'actuator',   'category': 'SSSP'},
    # 15: No physical impact — skip
    16: {'start': '2015-12-29 11:57:25', 'end': '2015-12-29 12:02:00',
         'primary': 'LIT301',  'all': ['LIT301'],
         'type': 'spoof',      'category': 'SSSP'},
    17: {'start': '2015-12-29 14:38:12', 'end': '2015-12-29 14:50:08',
         'primary': 'MV303',   'all': ['MV303'],
         'type': 'actuator',   'category': 'SSSP'},
    # 18: No physical impact — skip
    19: {'start': '2015-12-29 18:10:43', 'end': '2015-12-29 18:15:01',
         'primary': 'AIT504',  'all': ['AIT504'],
         'type': 'spoof',      'category': 'SSSP'},
    20: {'start': '2015-12-29 18:15:43', 'end': '2015-12-29 18:22:17',
         'primary': 'AIT504',  'all': ['AIT504'],
         'type': 'spoof',      'category': 'SSSP'},
    21: {'start': '2015-12-29 18:30:00', 'end': '2015-12-29 18:42:00',
         'primary': 'MV101',   'all': ['MV101', 'LIT101'],
         'type': 'multi',      'category': 'SSMP'},
    22: {'start': '2015-12-29 22:55:18', 'end': '2015-12-29 23:03:00',
         'primary': 'UV401',   'all': ['UV401', 'AIT502', 'P501'],
         'type': 'multi',      'category': 'MSMP'},
    23: {'start': '2015-12-30 01:42:34', 'end': '2015-12-30 01:54:10',
         'primary': 'P602',    'all': ['P602', 'DPIT301', 'MV302'],
         'type': 'multi',      'category': 'MSMP'},
    24: {'start': '2015-12-30 09:51:08', 'end': '2015-12-30 09:56:28',
         'primary': 'P203',    'all': ['P203', 'P205'],
         'type': 'multi',      'category': 'SSMP'},
    25: {'start': '2015-12-30 10:01:50', 'end': '2015-12-30 10:12:01',
         'primary': 'LIT401',  'all': ['LIT401', 'P401'],
         'type': 'multi',      'category': 'SSMP'},
    26: {'start': '2015-12-30 17:04:56', 'end': '2015-12-30 17:29:00',
         'primary': 'P101',    'all': ['P101', 'LIT301'],
         'type': 'multi',      'category': 'MSMP'},
    27: {'start': '2015-12-31 01:17:08', 'end': '2015-12-31 01:45:18',
         'primary': 'P302',    'all': ['P302', 'LIT401'],
         'type': 'multi',      'category': 'MSMP'},
    28: {'start': '2015-12-31 01:45:19', 'end': '2015-12-31 11:15:27',
         'primary': 'P302',    'all': ['P302'],
         'type': 'actuator',   'category': 'SSSP'},
    29: {'start': '2015-12-31 15:32:00', 'end': '2015-12-31 15:34:00',
         'primary': 'P201',    'all': ['P201', 'P203', 'P205'],
         'type': 'multi',      'category': 'SSMP'},
    30: {'start': '2015-12-31 15:47:40', 'end': '2015-12-31 16:07:10',
         'primary': 'LIT101',  'all': ['LIT101', 'P101', 'MV201'],
         'type': 'multi',      'category': 'MSMP'},
    31: {'start': '2015-12-31 22:05:34', 'end': '2015-12-31 22:11:40',
         'primary': 'LIT401',  'all': ['LIT401'],
         'type': 'spoof',      'category': 'SSSP'},
    32: {'start': '2016-01-01 10:36:00', 'end': '2016-01-01 10:46:00',
         'primary': 'LIT301',  'all': ['LIT301'],
         'type': 'spoof',      'category': 'SSSP'},
    33: {'start': '2016-01-01 14:21:12', 'end': '2016-01-01 14:28:35',
         'primary': 'LIT101',  'all': ['LIT101'],
         'type': 'spoof',      'category': 'SSSP'},
    34: {'start': '2016-01-01 17:12:40', 'end': '2016-01-01 17:14:20',
         'primary': 'P101',    'all': ['P101'],
         'type': 'actuator',   'category': 'SSSP'},
    35: {'start': '2016-01-01 17:18:56', 'end': '2016-01-01 17:26:56',
         'primary': 'P101',    'all': ['P101', 'P102'],
         'type': 'multi',      'category': 'SSMP'},
    36: {'start': '2016-01-01 22:16:01', 'end': '2016-01-01 22:25:00',
         'primary': 'LIT101',  'all': ['LIT101'],
         'type': 'spoof',      'category': 'SSSP'},
    37: {'start': '2016-01-02 11:17:02', 'end': '2016-01-02 11:24:50',
         'primary': 'P501',    'all': ['P501', 'FIT502'],
         'type': 'multi',      'category': 'SSMP'},
    38: {'start': '2016-01-02 11:31:38', 'end': '2016-01-02 11:36:18',
         'primary': 'AIT402',  'all': ['AIT402', 'AIT502'],
         'type': 'multi',      'category': 'SSMP'},
    39: {'start': '2016-01-02 11:43:48', 'end': '2016-01-02 11:50:28',
         'primary': 'FIT401',  'all': ['FIT401', 'AIT502'],
         'type': 'multi',      'category': 'MSSP'},
    40: {'start': '2016-01-02 11:51:42', 'end': '2016-01-02 11:56:38',
         'primary': 'FIT401',  'all': ['FIT401'],
         'type': 'spoof',      'category': 'SSSP'},
    41: {'start': '2016-01-02 13:13:02', 'end': '2016-01-02 13:40:56',
         'primary': 'LIT301',  'all': ['LIT301'],
         'type': 'spoof',      'category': 'SSSP'},
}

# Convert timestamps
for aid in ATTACK_MAP:
    ATTACK_MAP[aid]['start'] = pd.Timestamp(ATTACK_MAP[aid]['start'])
    ATTACK_MAP[aid]['end']   = pd.Timestamp(ATTACK_MAP[aid]['end'])


# ══════════════════════════════════════════════════════════════════════════════
# 3. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _load_csv(path: str) -> pd.DataFrame:
    """Load a SWaT CSV file and standardize column names."""
    def _candidate_datetime_series(frame: pd.DataFrame) -> pd.Series:
        cols = [str(c).strip() for c in frame.columns]
        frame.columns = cols
        lower = {c: c.lower() for c in cols}

        # Preferred path: explicit timestamp-like column names.
        ts_cols = [
            c for c in cols
            if ('timestamp' in lower[c]) or (lower[c] in {'time', 'datetime', 'date_time'})
        ]
        if ts_cols:
            return frame[ts_cols[0]].astype(str).str.strip()

        # Common SWaT variant: separate Date + Time columns.
        date_cols = [c for c in cols if 'date' in lower[c]]
        time_cols = [
            c for c in cols
            if ('time' in lower[c]) and ('date' not in lower[c])
        ]
        if date_cols and time_cols:
            return (
                frame[date_cols[0]].astype(str).str.strip() + ' ' +
                frame[time_cols[0]].astype(str).str.strip()
            )

        # Fallback: infer by parse success on early columns (e.g., "Unnamed: 0").
        nrows = len(frame)
        if nrows == 0:
            raise ValueError("CSV is empty")

        sample_n = min(2000, nrows)
        best_col = None
        best_ratio = 0.0
        for c in cols[: min(8, len(cols))]:
            sample = frame[c].astype(str).str.strip().head(sample_n)
            parsed = pd.to_datetime(sample, dayfirst=True, errors='coerce')
            ratio = float(parsed.notna().mean())
            if ratio > best_ratio:
                best_ratio = ratio
                best_col = c

        if best_col is not None and best_ratio >= 0.8:
            return frame[best_col].astype(str).str.strip()

        raise ValueError(
            f"No timestamp column found. Columns: {cols}"
        )

    last_error = None
    attempted_columns = []
    for header_row in [0, 1, 2, 3]:
        df = None
        try:
            df = pd.read_csv(path, low_memory=False, header=header_row)
            ts_source = _candidate_datetime_series(df)
            df['Timestamp'] = pd.to_datetime(
                ts_source,
                dayfirst=True,  # SWaT uses DD/MM/YYYY
                errors='coerce'
            )

            n_bad = int(df['Timestamp'].isna().sum())
            if n_bad > 0:
                print(f"  ⚠️  Dropped {n_bad} rows with unparseable timestamps")
                df = df.dropna(subset=['Timestamp'])

            if len(df) == 0:
                raise ValueError("No valid timestamp rows after parsing")

            df = df.sort_values('Timestamp').reset_index(drop=True)
            return df
        except Exception as exc:
            last_error = exc
            attempted_columns.append(getattr(df, 'columns', []))

    raise ValueError(
        f"Failed to parse SWaT CSV timestamps in {path}. "
        f"Tried header rows 0-3. Last error: {last_error}. "
        f"Sample attempted columns: {[list(map(str, c))[:12] for c in attempted_columns]}"
    )


def _encode_label(df: pd.DataFrame) -> pd.DataFrame:
    """Encode Normal/Attack column as binary integer label."""
    label_candidates = [c for c in df.columns
                        if 'normal' in c.lower() or 'attack' in c.lower()
                        or 'label' in c.lower()]
    
    if label_candidates:
        lc = label_candidates[0]
        df['label'] = (df[lc].astype(str).str.strip().str.lower()
                       .str.contains('attack')).astype(int)
    else:
        # Normal CSV has no label — all normal
        df['label'] = 0
    
    return df


def _add_attack_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-row attack metadata from the official iTrust attack list.
    Columns added:
        attack_id       : int or NaN
        attacked_sensor : str (primary sensor) or None
        attack_type     : spoof/actuator/multi or None
        attack_category : SSSP/SSMP/MSSP/MSMP or None
    This is the GROUND TRUTH for your localization metric.
    """
    df['attack_id']       = np.nan
    df['attacked_sensor'] = None
    df['attack_type']     = None
    df['attack_category'] = None
    
    for aid, info in ATTACK_MAP.items():
        mask = ((df['Timestamp'] >= info['start']) &
                (df['Timestamp'] <= info['end']))
        df.loc[mask, 'attack_id']       = aid
        df.loc[mask, 'attacked_sensor'] = info['primary']
        df.loc[mask, 'attack_type']     = info['type']
        df.loc[mask, 'attack_category'] = info['category']
    
    return df


def _select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select and validate the 51 core features."""
    available = [f for f in ALL_FEATURES if f in df.columns]
    missing   = [f for f in ALL_FEATURES if f not in df.columns]
    
    if missing:
        print(f"  ⚠️  Missing features: {missing}")
        print(f"      Available columns: {df.columns.tolist()[:20]}...")
    
    if not available:
        raise ValueError("No SWaT features found. Check column names.")
    
    return df, available


# ══════════════════════════════════════════════════════════════════════════════
# 4. MAIN LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_swat(
    normal_csv: str = '/kaggle/input/datasets/naimparvezz/swat-dataset/SWaT_Dataset_Normal_v1.csv',
    attack_csv: str = '/kaggle/input/datasets/naimparvezz/swat-dataset/SWaT_Dataset_Attack_v0.csv',
    use_kaggle_merged: str = None,
    window_size: int = WINDOW_SIZE,
    verbose: bool = True
):
    """
    Load and preprocess the SWaT dataset.

    Parameters
    ----------
    normal_csv       : path to normal operations CSV (official iTrust)
    attack_csv       : path to attack CSV (official iTrust)
    use_kaggle_merged: if provided, load Kaggle merged.csv instead
    window_size      : sliding window length
    verbose          : print progress

    Returns
    -------
    train_df, val_df, test_df : DataFrames with features + labels + metadata
    scaler                    : fitted StandardScaler (for inverse transform)
    available_features        : list of feature columns found
    """
    if verbose:
        print("=" * 60)
        print("SWaT DATA LOADER")
        print("=" * 60)

    # ── Load ────────────────────────────────────────────────────
    if use_kaggle_merged:
        if verbose: print(f"\n📂 Loading Kaggle merged: {use_kaggle_merged}")
        df = _load_csv(use_kaggle_merged)
        df = _encode_label(df)
    else:
        if verbose: print(f"\n📂 Loading normal: {normal_csv}")
        normal = _load_csv(normal_csv)
        normal = _encode_label(normal)   # all 0s
        
        if verbose: print(f"📂 Loading attack: {attack_csv}")
        attack = _load_csv(attack_csv)
        attack = _encode_label(attack)
        
        df = pd.concat([normal, attack], ignore_index=True)
        df = df.sort_values('Timestamp').reset_index(drop=True)

    if verbose:
        print(f"\n✅ Raw shape: {df.shape}")
        print(f"   Date range: {df['Timestamp'].min()} → "
              f"{df['Timestamp'].max()}")

    # ── Remove stabilization period ─────────────────────────────
    before = len(df)
    df = df[df['Timestamp'] > STABILIZATION_CUTOFF].reset_index(drop=True)
    if verbose:
        removed = before - len(df)
        print(f"\n🔧 Removed {removed:,} rows (stabilization period)")

    # ── Add attack metadata ──────────────────────────────────────
    df = _add_attack_metadata(df)
    if verbose:
        print(f"🔧 Added attack metadata for {len(ATTACK_MAP)} attacks")

    # ── Select features ──────────────────────────────────────────
    df, available_features = _select_features(df)
    if verbose:
        print(f"🔧 Features: {len(available_features)}/51 found")

    # ── Temporal split ───────────────────────────────────────────
    # Matches Homaei et al. 2026 protocol for fair comparison
    train_df = df[df['Timestamp'] <= TRAIN_END].copy()
    val_df   = df[(df['Timestamp'] > TRAIN_END) &
                  (df['Timestamp'] <= VAL_END)].copy()
    test_df  = df[df['Timestamp'] > VAL_END].copy()

    if verbose:
        print(f"\n📊 TEMPORAL SPLIT (matches Homaei et al. 2026):")
        print(f"   Train: {len(train_df):>8,} rows | "
              f"attacks: {train_df['label'].sum():>5,} "
              f"({train_df['label'].mean()*100:.1f}%)")
        print(f"   Val:   {len(val_df):>8,} rows | "
              f"attacks: {val_df['label'].sum():>5,} "
              f"({val_df['label'].mean()*100:.1f}%)")
        print(f"   Test:  {len(test_df):>8,} rows | "
              f"attacks: {test_df['label'].sum():>5,} "
              f"({test_df['label'].mean()*100:.1f}%)")

    # ── Normalize continuous sensors ─────────────────────────────
    # CRITICAL: fit ONLY on train — never on val/test
    cont_available = [f for f in CONTINUOUS_SENSORS
                      if f in available_features]
    
    scaler = StandardScaler()
    train_df[cont_available] = scaler.fit_transform(
        train_df[cont_available])
    val_df[cont_available]   = scaler.transform(val_df[cont_available])
    test_df[cont_available]  = scaler.transform(test_df[cont_available])

    # Binary actuators: convert to float, keep values as-is
    bin_available = [f for f in BINARY_ACTUATORS
                     if f in available_features]
    for split in [train_df, val_df, test_df]:
        split[bin_available] = split[bin_available].astype(float)

    if verbose:
        print(f"\n🔧 Normalization:")
        print(f"   Continuous sensors normalized: {len(cont_available)}")
        print(f"   Binary actuators unchanged:    {len(bin_available)}")
        
        # Verify no leakage
        means = train_df[cont_available].mean().abs()
        stds  = train_df[cont_available].std()
        print(f"   Max |mean| in train:  {means.max():.4f} (should be ≈0)")
        print(f"   Min std in train:     {stds.min():.4f}  (should be ≈1)")

    return train_df, val_df, test_df, scaler, available_features


# ══════════════════════════════════════════════════════════════════════════════
# 5. PYTORCH DATASET
# ══════════════════════════════════════════════════════════════════════════════

class SWaTDataset(Dataset):
    """
    Sliding window PyTorch Dataset for SWaT time-series.
    
    Each item returns:
        x        : FloatTensor [window_size, n_features]
        y        : FloatTensor scalar (0=normal, 1=attack)
        metadata : dict with attack_id, attacked_sensor, attack_type
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        features: list,
        window_size: int = WINDOW_SIZE,
        label_strategy: str = 'max'
    ):
        """
        Parameters
        ----------
        df             : preprocessed DataFrame from load_swat()
        features       : list of feature column names
        window_size    : number of timesteps per window
        label_strategy : 'max' (any attack in window = attack)
                         'last' (label of final timestep)
        """
        self.features   = features
        self.window_size = window_size
        self.strategy   = label_strategy
        
        # Extract arrays
        self.X = df[features].values.astype(np.float32)
        self.y = df['label'].values.astype(np.float32)
        
        # Attack metadata arrays
        self.attack_ids      = df['attack_id'].values
        self.attacked_sensors = df['attacked_sensor'].values
        self.attack_types    = df['attack_type'].values
        self.timestamps      = df['Timestamp'].values
        
    def __len__(self):
        return len(self.X) - self.window_size + 1
    
    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx:idx + self.window_size])
        
        # Label strategy
        window_labels = self.y[idx:idx + self.window_size]
        if self.strategy == 'max':
            y = torch.tensor(float(window_labels.max()))
        else:
            y = torch.tensor(float(window_labels[-1]))
        
        # Metadata for localization evaluation
        # Use the last timestep's metadata (most relevant for label)
        last = idx + self.window_size - 1
        attack_id = self.attack_ids[last]
        if pd.isna(attack_id):
            attack_id = -1
        else:
            attack_id = int(attack_id)

        attacked_sensor = self.attacked_sensors[last]
        if attacked_sensor is None or pd.isna(attacked_sensor):
            attacked_sensor = ''
        else:
            attacked_sensor = str(attacked_sensor)

        attack_type = self.attack_types[last]
        if attack_type is None or pd.isna(attack_type):
            attack_type = ''
        else:
            attack_type = str(attack_type)

        metadata = {
            'attack_id':       attack_id,
            'attacked_sensor': attacked_sensor,
            'attack_type':     attack_type,
            'timestamp':       str(self.timestamps[last]),
            'window_start':    str(self.timestamps[idx]),
        }
        
        return x, y, metadata
    
    def get_attack_windows(self):
        """Return indices of all windows containing at least one attack."""
        attack_indices = []
        for i in range(len(self)):
            window_labels = self.y[i:i + self.window_size]
            if window_labels.max() > 0:
                attack_indices.append(i)
        return attack_indices
    
    def get_windows_by_attack_id(self, attack_id: int):
        """Return all windows overlapping a specific attack."""
        info = ATTACK_MAP.get(attack_id)
        if info is None:
            return []
        indices = []
        for i in range(len(self)):
            last = i + self.window_size - 1
            aid = self.attack_ids[last]
            if not np.isnan(float(aid)) and int(aid) == attack_id:
                indices.append(i)
        return indices


# ══════════════════════════════════════════════════════════════════════════════
# 6. FEATURE ENGINEERING (additional derived features — optional)
# ══════════════════════════════════════════════════════════════════════════════

def add_engineered_features(df: pd.DataFrame,
                             available_features: list) -> tuple:
    """
    Add engineered features that may improve localization:
    
    1. Rolling statistics (mean, std) per sensor over 5-step window
       — captures local trend changes typical of sensor spoofing
    2. Inter-stage flow ratios (FIT101/FIT201, FIT301/FIT401)
       — captures cross-stage consistency violations
    
    NOTE: These are OPTIONAL. Your core contribution works without them.
    Only use if baseline LSTM AUROC is below 0.80.
    """
    new_features = []
    
    # Rolling std over 5 steps for key continuous sensors
    key_sensors = ['LIT101', 'LIT301', 'LIT401',
                   'FIT101', 'FIT201', 'FIT301', 'FIT401',
                   'AIT202', 'AIT402', 'AIT504', 'DPIT301']
    
    for sensor in key_sensors:
        if sensor in df.columns:
            col = f'{sensor}_rollstd5'
            df[col] = df[sensor].rolling(5, min_periods=1).std().fillna(0)
            new_features.append(col)
    
    # Flow ratio: inlet / outlet (physics consistency check)
    if 'FIT101' in df.columns and 'FIT201' in df.columns:
        df['RATIO_FIT101_FIT201'] = (
            df['FIT101'] / (df['FIT201'].abs() + 1e-6))
        new_features.append('RATIO_FIT101_FIT201')
    
    if 'FIT301' in df.columns and 'FIT401' in df.columns:
        df['RATIO_FIT301_FIT401'] = (
            df['FIT301'] / (df['FIT401'].abs() + 1e-6))
        new_features.append('RATIO_FIT301_FIT401')
    
    enhanced_features = available_features + new_features
    return df, enhanced_features, new_features


# ══════════════════════════════════════════════════════════════════════════════
# 7. VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def verify_loader(train_df, val_df, test_df, scaler, features,
                  window_size=WINDOW_SIZE):
    """Run all sanity checks and print a report."""
    print("\n" + "=" * 60)
    print("VERIFICATION REPORT")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            print(f"  ✅ {name}")
            passed += 1
        else:
            print(f"  ❌ {name} — {detail}")
            failed += 1
    
    # Test 1: No data leakage (temporal ordering)
    check("Temporal ordering",
          train_df['Timestamp'].max() < val_df['Timestamp'].min(),
          "Train and val timestamps overlap")
    check("Val before test",
          val_df['Timestamp'].max() < test_df['Timestamp'].min(),
          "Val and test timestamps overlap")
    
    # Test 2: Train has no attacks (all normal for unsupervised training)
    train_attack_rate = train_df['label'].mean()
    check("Train mostly normal",
          train_attack_rate < 0.01,
          f"Train attack rate: {train_attack_rate:.3f}")
    
    # Test 3: Test has attacks
    test_attack_rate = test_df['label'].mean()
    check("Test contains attacks",
          test_attack_rate > 0.1,
          f"Test attack rate: {test_attack_rate:.3f}")
    
    # Test 4: Feature count
    check(f"Features: {len(features)}",
          len(features) >= 51,
          f"Only {len(features)} features found")
    
    # Test 5: Normalization check
    cont_in_features = [f for f in CONTINUOUS_SENSORS if f in features]
    if cont_in_features:
        means = train_df[cont_in_features].mean().abs()
        stds  = train_df[cont_in_features].std()
        check("Continuous sensors normalized (mean≈0)",
              means.max() < 0.01,
              f"Max |mean|={means.max():.4f}")
        check("Continuous sensors normalized (std≈1)",
              stds.min() > 0.5,
              f"Min std={stds.min():.4f}")
    
    # Test 6: Binary actuators not normalized
    bin_in_features = [f for f in BINARY_ACTUATORS if f in features]
    if bin_in_features:
        bin_max = test_df[bin_in_features].max().max()
        check("Binary actuators have values > 1 (not normalized to 0-1)",
              bin_max > 1,
              "All binary values are 0-1, may be incorrectly normalized")
    
    # Test 7: PyTorch dataset shape
    ds = SWaTDataset(test_df, features, window_size)
    x, y, meta = ds[0]
    check(f"Window shape [{window_size}, {len(features)}]",
          x.shape == (window_size, len(features)),
          f"Got shape {x.shape}")
    check("Label is scalar float",
          y.dim() == 0,
          f"Got dim {y.dim()}")
    
    # Test 8: Attack metadata
    attack_windows = ds.get_attack_windows()
    check("Attack windows found",
          len(attack_windows) > 0,
          "No attack windows in test set")
    
    # Test 9: Specific attack localization ground truth (from test split)
    # Pick the first attack ID that has at least one window in test.
    present_attack_id = None
    present_attack_windows = []
    for aid in sorted(ATTACK_MAP.keys()):
        windows = ds.get_windows_by_attack_id(aid)
        if windows:
            present_attack_id = aid
            present_attack_windows = windows
            break

    check("At least one known attack ID appears in test",
          present_attack_id is not None,
          "No ATTACK_MAP attack IDs found in test set")

    if present_attack_id is not None:
        expected_sensor = ATTACK_MAP[present_attack_id]['primary']
        _, _, meta_attack = ds[present_attack_windows[0]]
        check(f"Attack {present_attack_id} primary sensor matches ATTACK_MAP",
              meta_attack['attacked_sensor'] == expected_sensor,
              f"Expected {expected_sensor}, got {meta_attack['attacked_sensor']}")
    
    # Summary
    print(f"\n{'─'*40}")
    print(f"  Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n  🎉 ALL CHECKS PASSED — Ready for model training")
    else:
        print(f"\n  ⚠️  {failed} checks failed — Review before training")
    
    # Attack coverage summary
    print(f"\n📊 ATTACK COVERAGE IN TEST SET:")
    print(f"   {'Attack':>7} {'Sensor':<12} {'Type':<12} {'Windows':>8}")
    print(f"   {'─'*7} {'─'*12} {'─'*12} {'─'*8}")
    for aid in sorted(ATTACK_MAP.keys()):
        windows = ds.get_windows_by_attack_id(aid)
        info = ATTACK_MAP[aid]
        if len(windows) > 0:
            print(f"   {aid:>7} {info['primary']:<12} "
                  f"{info['type']:<12} {len(windows):>8}")
    
    return passed, failed


# ══════════════════════════════════════════════════════════════════════════════
# 8. QUICK START
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    
    print("SWaT Loader — Quick Start")
    print("Usage options:")
    print()
    print("  Option A — Official iTrust files:")
    print("    train, val, test, scaler, feats = load_swat(")
    print("        normal_csv='SWaT_Dataset_Normal_v1.csv',")
    print("        attack_csv='SWaT_Dataset_Attack_v0.csv'")
    print("    )")
    print()
    print("  Option B — Kaggle merged.csv:")
    print("    train, val, test, scaler, feats = load_swat(")
    print("        use_kaggle_merged='merged.csv'")
    print("    )")
    print()
    
    # Auto-detect available file
    import os
    
    if os.path.exists('merged.csv'):
        print("🔍 Found: merged.csv (Kaggle version)")
        train, val, test, scaler, feats = load_swat(
            use_kaggle_merged='merged.csv')
        verify_loader(train, val, test, scaler, feats)
        
    elif (os.path.exists('SWaT_Dataset_Normal_v1.csv') and
          os.path.exists('SWaT_Dataset_Attack_v0.csv')):
        print("🔍 Found: Official iTrust files")
        train, val, test, scaler, feats = load_swat(
            normal_csv='SWaT_Dataset_Normal_v1.csv',
            attack_csv='SWaT_Dataset_Attack_v0.csv')
        verify_loader(train, val, test, scaler, feats)
        
    else:
        print("⚠️  No dataset files found in current directory.")
        print("    Place CSV files here and re-run.")