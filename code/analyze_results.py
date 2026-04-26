"""
analyze_results.py
==================

Loads results, filters to canonical backbone subset, validates completeness,
runs paired bootstrap statistics, and emits paper-ready tables.

Outputs (under summaries/):
    completeness_report.txt       Missing cells per (model, backbone)
    per_cell.csv                  One row per (model, backbone, cond, seed, dom, h)
    main_results.csv              Headline table: condition x model
    main_results.tex              Same as colored LaTeX (best=bold, 2nd=underline)
    per_backbone_<model>.csv/tex  Backbone-level breakdown
    per_domain_<model>.csv        Domain-level breakdown
    bootstrap_vs_c1.csv           Paired bootstrap: each condition vs C1
    bootstrap_per_backbone.csv    Same, broken down by backbone
    backbone_disagreement.csv     Where backbones disagree most strongly

USAGE
-----
    python code/analyze_results.py
    python code/analyze_results.py --models mmtsflib   # one model
    python code/analyze_results.py --include_all_backbones
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / 'results'
OUT_DIR = PROJECT_ROOT / 'summaries'

CANONICAL_BACKBONES = {
    'aurora':   {'default'},
    'tats':     {'iTransformer', 'Informer', 'DLinear'},
    'mmtsflib': {'iTransformer', 'Informer', 'DLinear'},
}

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

ALL_DOMAINS = list(TIME_MMD_HORIZONS.keys())
ALL_CONDITIONS = ['C1_original', 'C2_empty', 'C3_shuffled', 'C4_crossdomain',
                  'C5_constant', 'C6_unimodal', 'C7_null', 'C8_oracle',
                  'C9_zero_priors']

# Conditions excluded from paper tables/figures by default (still loaded
# into per_cell.csv for completeness). C7_null is redundant with C5_constant
# (both zero text and priors; C7 was a sanity-check variant) — we drop it
# from main paper tables to save space.
PAPER_EXCLUDED_CONDITIONS = {'C7_null'}
ALL_SEEDS = [2021, 2022, 2023]


def load_all_results(results_root: Path = RESULTS_ROOT) -> pd.DataFrame:
    rows = []
    for json_path in results_root.rglob('*.json'):
        try:
            data = json.loads(json_path.read_text())
            spec = data.get('spec', {})
            backbone = spec.get('backbone') or 'default'
            rows.append({
                'model':     spec.get('model'),
                'backbone':  backbone,
                'condition': spec.get('condition'),
                'seed':      spec.get('seed'),
                'domain':    spec.get('domain'),
                'pred_len':  spec.get('pred_len'),
                'success':   data.get('success', False),
                'mse':       data.get('mse'),
                'mae':       data.get('mae'),
                'wall_s':    data.get('wall_time_seconds'),
                'host':      data.get('hostname'),
            })
        except (json.JSONDecodeError, OSError):
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df[df['success']].copy()
    df = df.drop_duplicates(
        subset=['model', 'backbone', 'condition', 'seed', 'domain', 'pred_len'],
        keep='last',
    )
    return df


def filter_canonical(df: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for model, bbs in CANONICAL_BACKBONES.items():
        sub = df[(df['model'] == model) & (df['backbone'].isin(bbs))]
        keep.append(sub)
    return pd.concat(keep, ignore_index=True) if keep else df


def expected_grid() -> pd.DataFrame:
    rows = []
    for model, bbs in CANONICAL_BACKBONES.items():
        for bb in bbs:
            for cond in ALL_CONDITIONS:
                for seed in ALL_SEEDS:
                    for dom in ALL_DOMAINS:
                        for h in TIME_MMD_HORIZONS[dom]:
                            rows.append({
                                'model': model, 'backbone': bb,
                                'condition': cond, 'seed': seed,
                                'domain': dom, 'pred_len': h,
                            })
    return pd.DataFrame(rows)


def completeness_report(df: pd.DataFrame, out_path: Path) -> dict:
    expected = expected_grid()
    have = df[['model', 'backbone', 'condition', 'seed', 'domain', 'pred_len']].copy()
    have['have'] = True

    merged = expected.merge(
        have, how='left',
        on=['model', 'backbone', 'condition', 'seed', 'domain', 'pred_len'],
    )
    merged['have'] = merged['have'].fillna(False)
    missing = merged[~merged['have']]

    lines = ['Completeness Report', '=' * 60, '']
    summary = {}
    for model in CANONICAL_BACKBONES:
        for bb in sorted(CANONICAL_BACKBONES[model]):
            sub_exp = expected[(expected['model'] == model) & (expected['backbone'] == bb)]
            sub_miss = missing[(missing['model'] == model) & (missing['backbone'] == bb)]
            n_exp = len(sub_exp)
            n_miss = len(sub_miss)
            n_have = n_exp - n_miss
            pct = 100.0 * n_have / n_exp if n_exp else 0
            label = f'{model}/{bb}'
            lines.append(f'{label:<30s} {n_have:>5d}/{n_exp:<5d} ({pct:5.1f}%)  missing={n_miss}')
            summary[label] = {'have': n_have, 'expected': n_exp, 'missing': n_miss}
            if 0 < n_miss <= 30:
                for _, r in sub_miss.iterrows():
                    lines.append(f'    MISSING: {r.condition}/seed{r.seed}/{r.domain}_h{r.pred_len}')
            elif n_miss > 30:
                lines.append(f'    (showing 5 of {n_miss} missing)')
                for _, r in sub_miss.head(5).iterrows():
                    lines.append(f'    MISSING: {r.condition}/seed{r.seed}/{r.domain}_h{r.pred_len}')

    lines.append('')
    lines.append(f'TOTAL EXPECTED: {len(expected)}')
    lines.append(f'TOTAL HAVE:     {len(expected) - len(missing)}')
    lines.append(f'TOTAL MISSING:  {len(missing)}')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(lines))
    return summary


def paired_bootstrap(a: np.ndarray, b: np.ndarray,
                     B: int = 10000, ci: float = 0.95,
                     seed: int = 0xC01DC0DE) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    n = len(a)
    if n == 0:
        return dict(n_pairs=0, mean_diff=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                    p_two=np.nan, rel_diff=np.nan, ref_mean=np.nan)
    rng = np.random.default_rng(seed)
    diff = a - b
    obs = diff.mean()
    boot = diff[rng.integers(0, n, size=(B, n))].mean(axis=1)
    alpha = (1 - ci) / 2
    lo = float(np.quantile(boot, alpha))
    hi = float(np.quantile(boot, 1 - alpha))
    p = 2 * min((boot > 0).mean(), (boot < 0).mean())
    p = max(p, 1.0 / B)
    return dict(n_pairs=int(n), mean_diff=float(obs), ci_lo=lo, ci_hi=hi,
                p_two=float(p),
                rel_diff=float(obs / b.mean()) if b.mean() else np.nan,
                ref_mean=float(b.mean()))


def bootstrap_vs_reference(df: pd.DataFrame, ref: str = 'C1_original',
                            metric: str = 'mse',
                            group_by: Optional[tuple] = None) -> pd.DataFrame:
    rows = []
    grouping = ['model'] + (list(group_by) if group_by else [])
    for keys, sub in df.groupby(grouping):
        if not isinstance(keys, tuple):
            keys = (keys,)
        piv = sub.pivot_table(
            index=['domain', 'pred_len', 'seed'],
            columns='condition', values=metric, aggfunc='first',
        )
        if ref not in piv.columns:
            continue
        for cond in piv.columns:
            if cond == ref:
                continue
            stats = paired_bootstrap(piv[cond].values, piv[ref].values)
            row = dict(zip(grouping, keys))
            row.update(condition=cond, **stats)
            rows.append(row)
    return pd.DataFrame(rows)


def main_results_table(df: pd.DataFrame, metric: str = 'mse') -> pd.DataFrame:
    cell = df.groupby(['model', 'condition'])[metric].mean().unstack('model')
    cell = cell.reindex(ALL_CONDITIONS)
    return cell


def per_backbone_table(df: pd.DataFrame, model: str,
                       metric: str = 'mse') -> pd.DataFrame:
    sub = df[df['model'] == model]
    if sub.empty:
        return pd.DataFrame()
    cell = sub.groupby(['condition', 'backbone'])[metric].mean().unstack('backbone')
    cell = cell.reindex(ALL_CONDITIONS)
    return cell


def per_domain_table(df: pd.DataFrame, model: str,
                     metric: str = 'mse') -> pd.DataFrame:
    sub = df[df['model'] == model]
    if sub.empty:
        return pd.DataFrame()
    cell = sub.groupby(['condition', 'domain'])[metric].mean().unstack('domain')
    cell = cell.reindex(ALL_CONDITIONS)
    cell = cell.reindex(columns=[d for d in ALL_DOMAINS if d in cell.columns])
    return cell


def backbone_disagreement(df: pd.DataFrame, metric: str = 'mse') -> pd.DataFrame:
    rows = []
    grp = df.groupby(['model', 'condition', 'domain', 'pred_len', 'backbone'])[metric].mean()
    grp = grp.unstack('backbone')
    for idx, row in grp.iterrows():
        vals = row.dropna().values
        if len(vals) < 2:
            continue
        mean = vals.mean()
        std = vals.std(ddof=0)
        rows.append({
            'model': idx[0], 'condition': idx[1],
            'domain': idx[2], 'pred_len': idx[3],
            'mean': mean, 'std': std,
            'cv': std / mean if mean else np.nan,
            'n_backbones': len(vals),
        })
    out = pd.DataFrame(rows).sort_values('cv', ascending=False)
    return out


def _fmt(v: float, digits: int = 4) -> str:
    if pd.isna(v):
        return '--'
    return f'{v:.{digits}f}'


def df_to_latex_colored(df: pd.DataFrame, *,
                        caption: str = '',
                        label: str = '',
                        digits: int = 4,
                        lower_is_better: bool = True,
                        bold_best: bool = True,
                        underline_second: bool = True,
                        deltas_vs_first_row: bool = False) -> str:
    df = df.copy()
    headers = list(df.columns)
    rows_out = []
    for ridx, (rname, rvals) in enumerate(df.iterrows()):
        vals = rvals.values.astype(float)
        nonnan = ~np.isnan(vals)
        order = np.argsort(np.where(nonnan, vals if lower_is_better else -vals, np.inf))
        best_i = order[0] if nonnan.any() else None
        second_i = order[1] if nonnan.sum() >= 2 else None

        cells = []
        for ci, v in enumerate(vals):
            text = _fmt(v, digits)
            if bold_best and ci == best_i:
                text = r'\textbf{' + text + '}'
            elif underline_second and ci == second_i:
                text = r'\underline{' + text + '}'
            if deltas_vs_first_row and ridx > 0 and not pd.isna(v):
                ref = df.iloc[0, ci]
                if ref and not pd.isna(ref):
                    delta = (v - ref) / ref * 100
                    color = 'red' if delta > 0 else 'teal'
                    sign = '+' if delta > 0 else ''
                    text += rf' {{\tiny \textcolor{{{color}}}{{({sign}{delta:.1f}\%)}}}}'
            cells.append(text)
        row_label = str(rname).replace('_', r'\_')
        rows_out.append(row_label + ' & ' + ' & '.join(cells) + r' \\')

    col_spec = 'l' + 'r' * len(headers)
    header_row = ' & '.join([''] + [str(h).replace('_', r'\_') for h in headers]) + r' \\'
    body = (
        '% Requires: \\usepackage{booktabs, xcolor}\n'
        r'\begin{table}[t]' + '\n'
        rf'\caption{{{caption}}}' + '\n'
        rf'\label{{{label}}}' + '\n'
        r'\centering' + '\n'
        r'\small' + '\n'
        r'\begin{tabular}{' + col_spec + '}\n'
        r'\toprule' + '\n'
        + header_row + '\n'
        r'\midrule' + '\n'
        + '\n'.join(rows_out) + '\n'
        r'\bottomrule' + '\n'
        r'\end{tabular}' + '\n'
        r'\end{table}' + '\n'
    )
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=None)
    ap.add_argument('--include_all_backbones', action='store_true')
    ap.add_argument('--keep_c7', action='store_true',
                    help='Keep C7_null in paper tables. Default: drop it.')
    ap.add_argument('--metric', default='mse', choices=['mse', 'mae'])
    ap.add_argument('--results_root', type=Path, default=RESULTS_ROOT)
    ap.add_argument('--out_dir', type=Path, default=OUT_DIR)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_results(args.results_root)
    if df.empty:
        print(f'No results found in {args.results_root}')
        sys.exit(1)
    print(f'Loaded {len(df)} unique successful results')
    print(f'Backbones present in results: {sorted(df["backbone"].unique())}')

    if not args.include_all_backbones:
        df = filter_canonical(df)
        print(f'\nAfter canonical-backbone filter: {len(df)} cells')
        for model, bbs in CANONICAL_BACKBONES.items():
            n = (df['model'] == model).sum()
            print(f'  {model}: {n} cells (backbones={sorted(bbs)})')
    if args.models:
        df = df[df['model'].isin(args.models)]

    # Always save the full (un-filtered) per-cell file for completeness.
    df.to_csv(args.out_dir / 'per_cell.csv', index=False)
    print(f'\n  wrote {args.out_dir / "per_cell.csv"}')

    # Drop conditions excluded from paper tables (C7 by default).
    if not args.keep_c7:
        df_paper = df[~df['condition'].isin(PAPER_EXCLUDED_CONDITIONS)].copy()
        print(f'  paper tables exclude {sorted(PAPER_EXCLUDED_CONDITIONS)}')
    else:
        df_paper = df

    print('\n=== Completeness ===')
    summary = completeness_report(df, args.out_dir / 'completeness_report.txt')
    for k, v in summary.items():
        miss_pct = 100.0 * v['missing'] / v['expected'] if v['expected'] else 0
        flag = ' <-- INCOMPLETE' if miss_pct > 1 else ''
        print(f'  {k:<30s}: {v["have"]}/{v["expected"]} '
              f'({100 - miss_pct:.1f}% complete){flag}')

    print('\n=== Main results ===')
    main_tab = main_results_table(df_paper, args.metric)
    print(main_tab.to_string(float_format=lambda x: f'{x:.4f}'))
    main_tab.to_csv(args.out_dir / 'main_results.csv')
    latex = df_to_latex_colored(
        main_tab,
        caption=(f'Main results: mean {args.metric.upper()} across all '
                 r'(backbone, domain, horizon, seed) cells. Bold = best, '
                 r'\underline{underline} = second-best. '
                 r'Coloured deltas vs.\ C1\_original.'),
        label=f'tab:main_{args.metric}',
        deltas_vs_first_row=True,
    )
    (args.out_dir / 'main_results.tex').write_text(latex)
    print(f'  wrote main_results.csv + .tex')

    for model in df_paper["model"].unique():
        bb_tab = per_backbone_table(df_paper, model, args.metric)
        if bb_tab.empty:
            continue
        bb_tab.to_csv(args.out_dir / f'per_backbone_{model}.csv')
        latex = df_to_latex_colored(
            bb_tab,
            caption=(f'{model}: {args.metric.upper()} per backbone, averaged '
                     r'over (domain, horizon, seed). Deltas vs.\ C1\_original.'),
            label=f'tab:bb_{model}',
            deltas_vs_first_row=True,
        )
        (args.out_dir / f'per_backbone_{model}.tex').write_text(latex)
        print(f'  wrote per_backbone_{model}.csv + .tex')

    for model in df_paper["model"].unique():
        dom_tab = per_domain_table(df_paper, model, args.metric)
        if dom_tab.empty:
            continue
        dom_tab.to_csv(args.out_dir / f'per_domain_{model}.csv')

    print('\n=== Bootstrap (model-level) vs C1_original ===')
    boot = bootstrap_vs_reference(df_paper, ref='C1_original', metric=args.metric)
    boot.to_csv(args.out_dir / 'bootstrap_vs_c1.csv', index=False)
    cols = ['model', 'condition', 'n_pairs', 'mean_diff', 'ci_lo', 'ci_hi',
            'p_two', 'rel_diff']
    print(boot[cols].to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    print('\n=== Bootstrap per backbone vs C1_original ===')
    boot_bb = bootstrap_vs_reference(df_paper, ref='C1_original', metric=args.metric,
                                      group_by=('backbone',))
    boot_bb.to_csv(args.out_dir / 'bootstrap_per_backbone.csv', index=False)
    cols_bb = ['model', 'backbone', 'condition', 'n_pairs', 'mean_diff',
               'ci_lo', 'ci_hi', 'p_two', 'rel_diff']
    print(boot_bb[cols_bb].to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    dis = backbone_disagreement(df_paper, args.metric)
    dis.to_csv(args.out_dir / 'backbone_disagreement.csv', index=False)
    print('\n=== Top 10 cells where backbones disagree most (CV of MSE) ===')
    print(dis.head(10).to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    print(f'\nAll tables written to {args.out_dir}')


if __name__ == '__main__':
    main()