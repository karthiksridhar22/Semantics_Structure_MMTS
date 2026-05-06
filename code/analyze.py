"""analyze.py
==========

Lean results analyzer: loads probe JSONs, writes granular CSVs per model, and
emits per-(model, backbone) pivot tables (conditions x domains, averaged over
seeds and horizons). No LaTeX, no bootstrap, no statistics.

Output layout (under --out_dir, default ``summaries/``):

    master.csv                                 one row per (model, backbone,
                                               condition, seed, domain, pred_len)
    aurora.csv                                 master.csv filtered to aurora
    mmtsflib.csv                               master.csv filtered to mmtsflib
    tats.csv                                   master.csv filtered to tats
    tables/<model>_<backbone>_<metric>.csv     conditions (rows) x domains
                                               (cols), mean over (seed, pred_len)

By default only the canonical backbone set ``{Informer, DLinear, iTransformer}``
is kept for ``tats`` / ``mmtsflib`` (aurora always uses the single ``default``
backbone). Pass ``--all_backbones`` to expand that to the full 8-backbone set
``{Autoformer, Transformer, Crossformer, FiLM, FEDformer, DLinear, Informer,
iTransformer}``. Backbones outside the 8-set (e.g. ``LightTS``, ``PatchTST``,
``Reformer``, ``Nonstationary_Transformer``) are dropped in both modes.

Usage
-----
    python code/analyze.py
    python code/analyze.py --all_backbones
    python code/analyze.py --metrics mse mae
    python code/analyze.py --out_dir summaries
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / 'results'
DEFAULT_OUT = PROJECT_ROOT / 'summaries'

# Canonical condition / domain orderings. Cells outside these sets are still
# written to master.csv but pivot tables follow this ordering.
CONDITION_ORDER = [
    'C1_original', 'C2_empty', 'C3_shuffled', 'C4_crossdomain',
    'C5_constant', 'C6_unimodal', 'C8_oracle', 'C9_zero_priors', 'C10_empty_keep_priors', 'C11_constant_keep_priors',
]
DOMAIN_ORDER = [
    'Environment', 'Health', 'Energy', 'Agriculture',
    'Climate', 'Economy', 'Security', 'SocialGood', 'Traffic',
]

# Default backbones to keep when --all_backbones is not passed.
DEFAULT_BACKBONES = {
    'aurora':   ['default'],
    'mmtsflib': ['Informer', 'DLinear', 'iTransformer'],
    'tats':     ['Informer', 'DLinear', 'iTransformer'],
}

# Backbones to keep when --all_backbones IS passed. Anything on disk that is
# not in this set is dropped (so stale / exploratory backbones never leak
# into the master CSVs or pivot tables).
ALL_BACKBONES = {
    'aurora':   ['default'],
    'mmtsflib': ['Autoformer', 'Transformer', 'Crossformer', 'FiLM',
                 'FEDformer', 'DLinear', 'Informer', 'iTransformer'],
    'tats':     ['Autoformer', 'Transformer', 'Crossformer', 'FiLM',
                 'FEDformer', 'DLinear', 'Informer', 'iTransformer'],
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_results(results_root: Path) -> pd.DataFrame:
    """Walk ``results_root`` and load every successful JSON probe result."""
    rows = []
    for path in results_root.rglob('*.json'):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not data.get('success', False):
            continue
        spec = data.get('spec', {})
        rows.append({
            'model':     spec.get('model'),
            'backbone':  spec.get('backbone') or 'default',
            'condition': spec.get('condition'),
            'seed':      spec.get('seed'),
            'domain':    spec.get('domain'),
            'pred_len':  spec.get('pred_len'),
            'mse':       data.get('mse'),
            'mae':       data.get('mae'),
            'wall_s':    data.get('wall_time_seconds'),
        })
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(
        subset=['model', 'backbone', 'condition', 'seed', 'domain', 'pred_len'],
        keep='last',
    )
    return df


def filter_backbones(df: pd.DataFrame,
                     allowed: dict[str, list[str]]) -> pd.DataFrame:
    """Keep only ``(model, backbone)`` pairs listed in ``allowed``."""
    pieces = []
    for model, bbs in allowed.items():
        sub = df[(df['model'] == model) & (df['backbone'].isin(bbs))]
        pieces.append(sub)
    return pd.concat(pieces, ignore_index=True) if pieces else df.iloc[:0]


# --------------------------------------------------------------------------- #
# Pivot table builder
# --------------------------------------------------------------------------- #
def _ordered(values: Iterable[str], canonical: list[str]) -> list[str]:
    """Canonical order first, then any extras in alphabetical order."""
    values = list(values)
    head = [v for v in canonical if v in values]
    tail = sorted(v for v in values if v not in canonical)
    return head + tail


def pivot_condition_by_domain(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """conditions (rows) x domains (cols), mean of ``metric`` over seed+horizon."""
    if df.empty:
        return pd.DataFrame()
    table = (df.groupby(['condition', 'domain'])[metric]
               .mean()
               .unstack('domain'))
    table = table.reindex(_ordered(table.index, CONDITION_ORDER))
    table = table.reindex(columns=_ordered(table.columns, DOMAIN_ORDER))
    return table


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_root', type=Path, default=RESULTS_ROOT,
                    help='Root directory containing model/backbone/.../*.json')
    ap.add_argument('--out_dir', type=Path, default=DEFAULT_OUT,
                    help='Where to write CSVs (default: summaries/)')
    ap.add_argument('--metrics', nargs='+', default=['mse', 'mae'],
                    choices=['mse', 'mae'],
                    help='Metrics to emit pivot tables for (default: both)')
    ap.add_argument('--all_backbones', action='store_true',
                    help='Keep the full 8-backbone set {Autoformer, '
                         'Transformer, Crossformer, FiLM, FEDformer, DLinear, '
                         'Informer, iTransformer} instead of the default '
                         '3-backbone subset {Informer, DLinear, iTransformer}. '
                         'Anything on disk outside the 8-set is dropped.')
    args = ap.parse_args()

    df = load_results(args.results_root)
    if df.empty:
        print(f'No successful JSON results found under {args.results_root}',
              file=sys.stderr)
        sys.exit(1)

    allowed = ALL_BACKBONES if args.all_backbones else DEFAULT_BACKBONES
    df = filter_backbones(df, allowed)

    if df.empty:
        print('No rows left after backbone filtering.', file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = args.out_dir / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)

    # --- Master + per-model granular CSVs --------------------------------- #
    cols = ['model', 'backbone', 'condition', 'seed', 'domain', 'pred_len',
            'mse', 'mae', 'wall_s']
    master = df.sort_values(cols[:-3])[cols].reset_index(drop=True)
    master.to_csv(args.out_dir / 'master.csv', index=False)
    print(f'wrote {args.out_dir / "master.csv"}  ({len(master)} rows)')

    for model, sub in master.groupby('model', sort=True):
        out = args.out_dir / f'{model}.csv'
        sub.to_csv(out, index=False)
        print(f'wrote {out}  ({len(sub)} rows)')

    # --- Per-(model, backbone, metric) pivot tables ----------------------- #
    pairs = (master[['model', 'backbone']]
             .drop_duplicates()
             .sort_values(['model', 'backbone'])
             .itertuples(index=False, name=None))
    for model, backbone in pairs:
        sub = master[(master['model'] == model) &
                     (master['backbone'] == backbone)]
        for metric in args.metrics:
            tab = pivot_condition_by_domain(sub, metric)
            if tab.empty:
                continue
            out = tables_dir / f'{model}_{backbone}_{metric}.csv'
            tab.to_csv(out, float_format='%.6f')
            print(f'wrote {out}  ({tab.shape[0]} conds x {tab.shape[1]} doms)')

    # --- Quick coverage summary ------------------------------------------- #
    print('\nCoverage (rows per model/backbone):')
    counts = (master.groupby(['model', 'backbone'])
                    .size()
                    .reset_index(name='n_rows'))
    for _, r in counts.iterrows():
        print(f'  {r.model:<10s} {r.backbone:<14s} {r.n_rows:>6d}')


if __name__ == '__main__':
    main()
