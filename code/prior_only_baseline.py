"""
prior_only_baseline.py
======================

Computes a "use the numeric prior directly" forecasting baseline. For each
domain and pred_len, predicts y_hat[t+1:t+h] = prior_history_avg[t+1:t+h]
(constant for all h steps) and reports MSE and MAE in the SAME standardized
space the trained models report (StandardScaler fit on the train split of
OT, applied to both prediction and target).

This baseline answers: "How much of MM-TSFM performance is explainable by
the numeric prior alone, with no model trained at all?"

If TaTS / MM-TSFlib's C1 MSE is close to this baseline's MSE, those models
add little beyond what's already in the precomputed prior column.

USAGE
-----
    python code/prior_only_baseline.py
    python code/prior_only_baseline.py --domains Economy Health
    python code/prior_only_baseline.py --pred_lens 8 12

OUTPUT
------
Writes summaries/prior_only_baseline.csv with one row per (domain, pred_len),
plus a printed table to stdout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / 'data' / 'tats' / 'C1_original' / 'seed0'
OUT_DIR = PROJECT_ROOT / 'summaries'

TIME_MMD_HORIZONS = {
    'Environment': [48, 96, 192, 336],
    'Health':      [12, 24, 36, 48],
    'Energy':      [12, 24, 36, 48],
    'Agriculture': [6, 8, 10, 12],
    'Climate':     [6, 8, 10, 12],
    'Economy':     [6, 8, 10, 12],
    'Security':    [6, 8, 10, 12],
    'SocialGood':  [6, 8, 10, 12],
    'Traffic':     [6, 8, 10, 12],
}

# Per-domain seq_len matches Time-MMD paper: needed only to define test-window
# alignment so we score the same y_hat[t+1:t+h] points the trained models do.
TIME_MMD_SEQ_LEN = {
    'Environment': 96, 'Health': 36, 'Energy': 36,
    'Agriculture': 8, 'Climate': 8, 'Economy': 8,
    'Security': 8, 'SocialGood': 8, 'Traffic': 8,
}


def baseline_one_cell(df: pd.DataFrame, seq_len: int, pred_len: int) -> dict:
    """Compute prior-only MSE/MAE on the test split for one (domain, h).

    Mirrors the shipped data loaders' splitting:
      train_end   = floor(0.7 * N)
      val_end     = floor(0.7 * N) + (N - floor(0.7*N) - floor(0.2*N))
      test_start  = N - floor(0.2 * N) - seq_len   # to allow lookback
      test windows: i in [test_start, N - seq_len - pred_len + 1)
    """
    n = len(df)
    n_train = int(n * 0.7)
    n_test = int(n * 0.2)
    test_start = n - n_test - seq_len
    if test_start < 0:
        return {'n_windows': 0, 'mse': np.nan, 'mae': np.nan}

    y = pd.to_numeric(df['OT'], errors='coerce').to_numpy()
    if 'prior_history_avg' not in df.columns:
        return {'n_windows': 0, 'mse': np.nan, 'mae': np.nan,
                'note': 'no prior_history_avg column'}
    p = pd.to_numeric(df['prior_history_avg'], errors='coerce').to_numpy()

    # Fit the scaler on the TRAIN slice of OT, like the model loaders do.
    train_y = y[:n_train].reshape(-1, 1)
    scaler = StandardScaler().fit(train_y)
    y_s = scaler.transform(y.reshape(-1, 1)).flatten()
    p_s = scaler.transform(p.reshape(-1, 1)).flatten()

    # For each test-window starting position s, the model predicts
    # y[s+seq_len : s+seq_len+pred_len]. Our naive baseline uses the
    # corresponding slice from prior_history_avg.
    se_list = []
    ae_list = []
    n_windows = 0
    for s in range(test_start, n - seq_len - pred_len + 1):
        truth = y_s[s + seq_len : s + seq_len + pred_len]
        pred = p_s[s + seq_len : s + seq_len + pred_len]
        if np.isnan(pred).any() or np.isnan(truth).any():
            continue
        se_list.append(((pred - truth) ** 2).mean())
        ae_list.append(np.abs(pred - truth).mean())
        n_windows += 1

    if n_windows == 0:
        return {'n_windows': 0, 'mse': np.nan, 'mae': np.nan}

    return {
        'n_windows': n_windows,
        'mse': float(np.mean(se_list)),
        'mae': float(np.mean(ae_list)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domains', nargs='+', default=None)
    ap.add_argument('--pred_lens', nargs='+', type=int, default=None,
                    help='Restrict to specific pred_lens (across all domains).')
    ap.add_argument('--data_root', type=Path, default=DATA_ROOT)
    ap.add_argument('--out_dir', type=Path, default=OUT_DIR)
    args = ap.parse_args()

    domains = args.domains or list(TIME_MMD_HORIZONS.keys())
    rows = []
    for dom in domains:
        csv_path = args.data_root / f'{dom}.csv'
        if not csv_path.exists():
            print(f'SKIP {dom}: {csv_path} missing')
            continue
        df = pd.read_csv(csv_path, keep_default_na=False)
        seq_len = TIME_MMD_SEQ_LEN[dom]
        horizons = TIME_MMD_HORIZONS[dom]
        if args.pred_lens:
            horizons = [h for h in horizons if h in args.pred_lens]
        for h in horizons:
            stats = baseline_one_cell(df, seq_len, h)
            rows.append({
                'domain': dom, 'seq_len': seq_len, 'pred_len': h,
                'n_test_windows': stats['n_windows'],
                'prior_only_mse': stats['mse'],
                'prior_only_mae': stats['mae'],
            })

    out = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / 'prior_only_baseline.csv'
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()