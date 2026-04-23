"""
analyze_results.py
==================

Load all RunResults from results/, compute paired bootstrap significance
tests between conditions, emit summary tables (CSV and LaTeX) for the paper.

KEY STATISTICAL IDEA
--------------------
We have ≤3 seeds per (model, domain, pred_len) cell, which is too few for a
standalone test. Instead we POOL across (domain, pred_len) when asking
"does condition X differ from C1 for model M?" — giving up to
(9 domains × 4 horizons × 3 seeds) = 108 paired observations.

Paired design: for each (domain, pred_len, seed) we have MSE under every
condition. Take differences (e.g., MSE_C2 - MSE_C1), bootstrap the mean
difference, report the 95% CI. If the CI excludes zero → significant.

This is MORE informative than p-values — it tells the reader effect size
AND uncertainty in the same statistic.

NOTE: The paired bootstrap is the standard go-to when paired
observations exist, data is non-normal, and you don't want to assume
anything about the underlying distribution. Efron & Tibshirani (1993) is
the canonical reference; use B=10000 resamples for publication-quality CIs.

USAGE
-----
  python analyze_results.py                      # emit all tables
  python analyze_results.py --models aurora      # just one model
  python analyze_results.py --latex tables/      # write LaTeX into tables/
  python analyze_results.py --csv summaries/     # CSV outputs
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path('/home/karthik/Semantics_Structure_MMTS')
RESULTS_ROOT = PROJECT_ROOT / 'results'


# =============================================================================
#  Loading
# =============================================================================

def load_all_results(results_root: Path = RESULTS_ROOT) -> pd.DataFrame:
    """Walk results/ and return a DataFrame with one row per cell."""
    rows = []
    for json_path in results_root.rglob('*.json'):
        try:
            data = json.loads(json_path.read_text())
            spec = data.get('spec', {})
            rows.append({
                'model':     spec.get('model'),
                'condition': spec.get('condition'),
                'seed':      spec.get('seed'),
                'domain':    spec.get('domain'),
                'pred_len':  spec.get('pred_len'),
                'success':   data.get('success', False),
                'mse':       data.get('mse'),
                'mae':       data.get('mae'),
                'smape':     data.get('smape'),
                'wall_s':    data.get('wall_time_seconds'),
                'path':      str(json_path),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return pd.DataFrame(rows)


# =============================================================================
#  Summary statistics
# =============================================================================

def per_cell_summary(df: pd.DataFrame, metric: str = 'mse') -> pd.DataFrame:
    """Mean ± std across seeds, per (model, condition, domain, pred_len)."""
    df = df[df['success']].copy()
    g = df.groupby(['model', 'condition', 'domain', 'pred_len'])[metric]
    out = g.agg(['mean', 'std', 'count']).reset_index()
    out.columns = ['model', 'condition', 'domain', 'pred_len',
                   f'{metric}_mean', f'{metric}_std', 'n_seeds']
    return out


# =============================================================================
#  Paired bootstrap
# =============================================================================

def paired_bootstrap_diff(
    a: np.ndarray, b: np.ndarray,
    B: int = 10000, ci: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Bootstrap the mean of paired differences (a - b).

    Returns dict with keys:
        n_pairs, mean_diff, ci_lo, ci_hi, p_two_sided (via percentile),
        rel_diff (mean_diff / mean(b)) for intuition.

    a, b must be same length, NaNs removed.
    """
    if rng is None:
        rng = np.random.default_rng(0xC01DC0DE)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    n = len(a)
    if n == 0:
        return {'n_pairs': 0, 'mean_diff': np.nan, 'ci_lo': np.nan,
                'ci_hi': np.nan, 'p_two_sided': np.nan, 'rel_diff': np.nan}
    diff = a - b
    observed = diff.mean()
    # Bootstrap resample indices n times, B replicates
    idx = rng.integers(0, n, size=(B, n))
    boot_means = diff[idx].mean(axis=1)
    alpha = (1 - ci) / 2
    lo = np.quantile(boot_means, alpha)
    hi = np.quantile(boot_means, 1 - alpha)
    # Achieved significance: smallest alpha s.t. 0 is outside CI
    # Approximate as min(P(boot > 0), P(boot < 0)) * 2
    p = 2 * min((boot_means > 0).mean(), (boot_means < 0).mean())
    rel = observed / b.mean() if b.mean() != 0 else np.nan
    return {
        'n_pairs': n, 'mean_diff': observed,
        'ci_lo': lo, 'ci_hi': hi,
        'p_two_sided': p, 'rel_diff': rel,
    }


def compare_conditions(
    df: pd.DataFrame,
    model: str, test_cond: str, ref_cond: str = 'C1_original',
    metric: str = 'mse',
    group_cols: tuple = ('domain', 'pred_len', 'seed'),
) -> dict:
    """Paired bootstrap: does `test_cond` differ from `ref_cond` for `model`?

    Pairs are formed by matching on `group_cols` (default: same domain, same
    pred_len, same seed). This controls for data-specific difficulty.
    """
    df = df[df['success'] & (df['model'] == model)].copy()
    piv = df.pivot_table(index=list(group_cols),
                         columns='condition', values=metric,
                         aggfunc='first')
    if test_cond not in piv.columns or ref_cond not in piv.columns:
        return {'error': f'missing columns; have {list(piv.columns)}'}
    out = paired_bootstrap_diff(piv[test_cond].values, piv[ref_cond].values)
    out.update({'model': model, 'test': test_cond, 'ref': ref_cond,
                'metric': metric})
    return out


def all_pairwise(df: pd.DataFrame, metric: str = 'mse') -> pd.DataFrame:
    """Run paired bootstrap for every (model, test_cond) vs C1_original.

    Returns a DataFrame with one row per comparison.
    """
    rows = []
    models = sorted(df['model'].dropna().unique())
    conditions = sorted(df['condition'].dropna().unique())
    for model in models:
        for test in conditions:
            if test == 'C1_original':
                continue
            rows.append(compare_conditions(df, model, test, 'C1_original', metric))
    return pd.DataFrame(rows)


# =============================================================================
#  Tables
# =============================================================================

def main_results_table(df: pd.DataFrame, metric: str = 'mse') -> pd.DataFrame:
    """Main paper table: rows = conditions, cols = (model, domain_mean).

    Mean across seeds and pred_lens within each (model, domain) cell.
    """
    df = df[df['success']].copy()
    piv = df.groupby(['model', 'condition', 'domain'])[metric].mean().reset_index()
    wide = piv.pivot_table(index='condition', columns=['model', 'domain'],
                           values=metric)
    return wide


def to_latex(df: pd.DataFrame, path: Path, caption: str = '',
             label: str = '', float_format='%.4f'):
    """Write a DataFrame as a LaTeX table, with caption and label."""
    path.parent.mkdir(parents=True, exist_ok=True)
    latex = df.to_latex(float_format=float_format, escape=False,
                        caption=caption, label=label)
    path.write_text(latex)


# =============================================================================
#  CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=None)
    ap.add_argument('--metric', default='mse', choices=['mse', 'mae', 'smape'])
    ap.add_argument('--csv_dir', default='summaries')
    ap.add_argument('--latex_dir', default='tables')
    ap.add_argument('--bootstrap_B', type=int, default=10000)
    args = ap.parse_args()

    df = load_all_results()
    if df.empty:
        print('No results found. Run experiments first.')
        return
    if args.models:
        df = df[df['model'].isin(args.models)]

    n_total = len(df)
    n_ok = df['success'].sum()
    print(f'Loaded {n_total} result files ({n_ok} successful).')

    csv_dir = PROJECT_ROOT / args.csv_dir
    latex_dir = PROJECT_ROOT / args.latex_dir
    csv_dir.mkdir(parents=True, exist_ok=True)
    latex_dir.mkdir(parents=True, exist_ok=True)

    # 1. Per-cell summary
    summary = per_cell_summary(df, args.metric)
    summary.to_csv(csv_dir / f'per_cell_{args.metric}.csv', index=False)
    print(f'  wrote {csv_dir / f"per_cell_{args.metric}.csv"}')

    # 2. Main results table (wide)
    main = main_results_table(df, args.metric)
    main.to_csv(csv_dir / f'main_{args.metric}.csv')
    to_latex(main, latex_dir / f'main_{args.metric}.tex',
             caption=f'Main results: {args.metric.upper()} by condition and domain',
             label=f'tab:main_{args.metric}')
    print(f'  wrote main results table (csv + latex)')

    # 3. Pairwise comparisons
    pw = all_pairwise(df, args.metric)
    if not pw.empty:
        pw.to_csv(csv_dir / f'pairwise_{args.metric}.csv', index=False)
        print(f'  wrote pairwise bootstrap comparisons')
        # Show a quick terminal summary
        print('\nBootstrap summary (vs C1_original):')
        print(pw[['model', 'test', 'n_pairs', 'mean_diff',
                  'ci_lo', 'ci_hi', 'p_two_sided', 'rel_diff']]
              .round(4).to_string(index=False))

    print('\nAnalysis complete.')


if __name__ == '__main__':
    main()
