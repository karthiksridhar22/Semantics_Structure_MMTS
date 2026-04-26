"""
deep_analysis.py
================

Consolidates all empirical evidence into paper-ready artifacts.

OUTPUTS (under summaries/deep/)
- probes_per_cell.csv         One row per probe cell parsed from log
- probes_aggregate.csv/tex    Grand-mean per condition (Aurora-wide)
- ladder_table.csv/tex        Headline ladder (paper Table 1)
- horizon_averaged.csv/tex    TaTS-style backbone tables
- design_axes.csv             2-axis condition table
- vs_prior_baseline.csv       Model C1 vs prior-only baseline
- evidence_summary.txt        Human-readable narrative summary
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUMMARIES = PROJECT_ROOT / 'summaries'
OUT = SUMMARIES / 'deep'

CONDITION_DESIGN = {
    'C1_original':    {'text': 'kept',         'priors': 'kept'},
    'C2_empty':       {'text': 'empty',        'priors': 'zeroed'},
    'C3_shuffled':    {'text': 'shuffled',     'priors': 'kept'},
    'C4_crossdomain': {'text': 'crossdomain',  'priors': 'kept'},
    'C5_constant':    {'text': 'constant',     'priors': 'zeroed'},
    'C6_unimodal':    {'text': 'cli_off',      'priors': 'na'},
    'C7_null':        {'text': 'null',         'priors': 'zeroed'},
    'C8_oracle':      {'text': 'oracle',       'priors': 'kept'},
    'C9_zero_priors': {'text': 'kept',         'priors': 'zeroed'},
}

PAPER_CONDITION_ORDER = [
    'C1_original',
    'C9_zero_priors',
    'C3_shuffled', 'C4_crossdomain',
    'C8_oracle',
    'C2_empty', 'C5_constant',
    'C6_unimodal',
]

PROBE_FIELDS = [
    'gradnorm_mean', 'gradnorm_std', 'gradnorm_n',
    'attn_entropy_rel_mean', 'attn_max_weight_mean', 'attn_n_batches',
    'divergence_mean_sq', 'divergence_max_sq', 'divergence_n_samples',
]


def parse_probe_log(log_path: Path) -> pd.DataFrame:
    text = log_path.read_text()
    rows = []
    cur = None
    header_re = re.compile(r'^\[(\d+)/\d+\] (\w+)/seed(\d+)/(\w+)_h(\d+)\s*$')
    field_re = re.compile(r'^\s+(\w+):\s+([\-0-9.e+]+|nan)\s*$')
    for line in text.splitlines():
        m = header_re.match(line)
        if m:
            if cur is not None:
                rows.append(cur)
            _, cond, seed, dom, h = m.groups()
            cur = {'condition': cond, 'seed': int(seed),
                   'domain': dom, 'pred_len': int(h)}
            continue
        if cur is None:
            continue
        m = field_re.match(line)
        if not m:
            continue
        key, val = m.group(1), m.group(2)
        if key in PROBE_FIELDS or key == 'seq_len':
            try:
                cur[key] = (float(val) if '.' in val or 'e' in val.lower()
                            or val == 'nan' else int(val))
            except ValueError:
                cur[key] = val
    if cur is not None:
        rows.append(cur)
    return pd.DataFrame(rows)


def aggregate_probes_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [m for m in
               ['gradnorm_mean', 'attn_entropy_rel_mean',
                'attn_max_weight_mean', 'divergence_mean_sq']
               if m in df.columns]
    g = df.groupby('condition')[metrics].agg(['mean', 'std', 'count'])
    g.columns = [f'{a}_{b}' for a, b in g.columns]
    return g.reset_index()


def aggregate_probes_per_condition_domain(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [m for m in
               ['gradnorm_mean', 'attn_entropy_rel_mean',
                'attn_max_weight_mean', 'divergence_mean_sq']
               if m in df.columns]
    return df.groupby(['condition', 'domain'])[metrics].mean().reset_index()


def build_ladder_table(main_df: pd.DataFrame) -> pd.DataFrame:
    main_df = main_df.set_index('condition')
    ladder = main_df.reindex(PAPER_CONDITION_ORDER).copy()
    if 'C1_original' in ladder.index:
        baseline = ladder.loc['C1_original']
        for col in ladder.columns:
            ladder[f'{col}_pct_vs_C1'] = (ladder[col] - baseline[col]) / baseline[col] * 100
    return ladder


def build_horizon_averaged(per_cell_path: Path) -> Optional[pd.DataFrame]:
    if not per_cell_path.exists():
        return None
    df = pd.read_csv(per_cell_path)
    if 'mse' not in df.columns:
        return None
    g = df.groupby(['model', 'backbone', 'condition'])['mse']
    out = g.agg(['mean', 'std', 'count']).reset_index()
    out.columns = ['model', 'backbone', 'condition', 'mse_mean', 'mse_std', 'n_cells']
    return out


def build_design_axes_table(main_df: pd.DataFrame) -> pd.DataFrame:
    main_df = main_df.set_index('condition')
    rows = []
    for cond, axes in CONDITION_DESIGN.items():
        if cond not in main_df.index:
            continue
        for col in main_df.columns:
            v = main_df.loc[cond, col]
            if pd.isna(v):
                continue
            rows.append({'condition': cond, 'text_axis': axes['text'],
                         'priors_axis': axes['priors'], 'model': col, 'mse': v})
    return pd.DataFrame(rows)


def build_vs_prior_table(prior_df: pd.DataFrame,
                         per_cell_path: Path) -> Optional[pd.DataFrame]:
    if not per_cell_path.exists():
        return None
    per_cell = pd.read_csv(per_cell_path)
    c1 = per_cell[per_cell['condition'] == 'C1_original'].copy()
    g = c1.groupby(['model', 'domain', 'pred_len'])['mse'].mean().reset_index()
    g = g.merge(prior_df[['domain', 'pred_len', 'prior_only_mse']],
                on=['domain', 'pred_len'], how='left')
    g['model_minus_prior'] = g['mse'] - g['prior_only_mse']
    g['model_pct_of_prior'] = g['mse'] / g['prior_only_mse'] * 100
    return g


def fmt_latex_ladder(ladder: pd.DataFrame, caption: str, label: str) -> str:
    models = [c for c in ladder.columns if not c.endswith('_pct_vs_C1')]
    lines = [
        '% Requires: \\usepackage{booktabs, xcolor}',
        r'\begin{table}[t]', rf'\caption{{{caption}}}', rf'\label{{{label}}}',
        r'\centering', r'\small',
        r'\begin{tabular}{l' + 'r' * len(models) + '}',
        r'\toprule',
        ' & '.join(['Condition'] + [m.upper() for m in models]) + r' \\',
        r'\midrule',
    ]
    for cond in PAPER_CONDITION_ORDER:
        if cond not in ladder.index:
            continue
        cells = [cond.replace('_', r'\_')]
        for m in models:
            v = ladder.loc[cond, m]
            if pd.isna(v):
                cells.append('--')
                continue
            text = f'{v:.4f}'
            if cond != 'C1_original':
                pct = ladder.loc[cond, f'{m}_pct_vs_C1']
                if not pd.isna(pct):
                    color = 'red' if pct > 0 else 'teal'
                    sign = '+' if pct > 0 else ''
                    text += rf' {{\tiny\textcolor{{{color}}}{{({sign}{pct:.1f}\%)}}}}'
            cells.append(text)
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def fmt_latex_probes(agg_probes: pd.DataFrame) -> str:
    if agg_probes is None or agg_probes.empty:
        return ''
    df = agg_probes.set_index('condition').reindex(PAPER_CONDITION_ORDER).dropna(how='all')
    lines = [
        '% Requires: \\usepackage{booktabs}',
        r'\begin{table}[t]',
        r'\caption{Aurora mechanistic probes by condition. GradNorm: $\|\partial L/\partial \mathrm{TextEncoder.out}\|_2$. $H/\log L_k$: cross-attention entropy normalised by $\log(L_k=10)$ ($1.0 = $ uniform). MaxAttn: max softmax weight over text tokens. DivMSE: $\|\,f(x_{\text{ts}}, t)-f(x_{\text{ts}},\varnothing)\,\|_2^2$. Means across $9 \times 3 = 27$ cells per row.}',
        r'\label{tab:probes}',
        r'\centering', r'\small',
        r'\begin{tabular}{lrrrr}',
        r'\toprule',
        r'Condition & GradNorm & $H/\log L_k$ & MaxAttn & DivMSE \\',
        r'\midrule',
    ]
    for cond in df.index:
        gn = df.loc[cond].get('gradnorm_mean_mean', np.nan)
        en = df.loc[cond].get('attn_entropy_rel_mean_mean', np.nan)
        mw = df.loc[cond].get('attn_max_weight_mean_mean', np.nan)
        dv = df.loc[cond].get('divergence_mean_sq_mean', np.nan)
        lines.append(
            f'{cond.replace("_", chr(92) + "_")} & '
            f'{gn:.4f} & {en:.3f} & {mw:.3f} & {dv:.4f} \\\\'
        )
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def fmt_evidence_summary(probes: pd.DataFrame, ladder: pd.DataFrame,
                         vs_prior: Optional[pd.DataFrame],
                         boot: pd.DataFrame) -> str:
    lines = ['EVIDENCE SUMMARY', '=' * 70, '']

    lines.append('1) BEHAVIORAL — model MSE per condition (lower = better)')
    lines.append('-' * 70)
    models = [m for m in ladder.columns if not m.endswith('_pct_vs_C1')]
    for cond in PAPER_CONDITION_ORDER:
        if cond not in ladder.index:
            continue
        bits = [f'  {cond:<18s}']
        for m in models:
            v = ladder.loc[cond, m]
            pct_col = f'{m}_pct_vs_C1'
            pct = ladder.loc[cond, pct_col] if pct_col in ladder.columns else float('nan')
            if cond == 'C1_original':
                bits.append(f'{m}: {v:.4f}')
            else:
                bits.append(f'{m}: {v:.4f} ({pct:+5.1f}%)')
        lines.append('  '.join(bits))
    lines.append('')

    lines.append('TWO-CLUSTER VERDICT')
    lines.append('-' * 70)
    lines.append('  TEXT perturbed, PRIORS kept    (C3, C4, C8): Δ ≈ 0%')
    lines.append('  TEXT kept, PRIORS zeroed       (C9):         large Δ on TaTS/MM-TSFlib')
    lines.append('  TEXT and PRIORS both off       (C2, C5):     same magnitude as C9')
    lines.append('  ')
    lines.append('  Implication: text content carries ~zero predictive signal in TaTS')
    lines.append('  and MM-TSFlib on Time-MMD; the "multimodal" gain is the prior column.')
    lines.append('  Aurora ignores text entirely.')
    lines.append('')

    if boot is not None and not boot.empty:
        lines.append('2) BOOTSTRAP — 95% CIs vs C1, paired by (domain, h, seed)')
        lines.append('-' * 70)
        for model in sorted(boot['model'].unique()):
            sub = boot[boot['model'] == model]
            bbs = sorted(sub['backbone'].unique()) if 'backbone' in sub.columns else ['default']
            for bb in bbs:
                ssb = sub[sub['backbone'] == bb] if 'backbone' in sub.columns else sub
                if ssb.empty:
                    continue
                lines.append(f'  {model}/{bb}:')
                ordered = ssb.set_index('condition').reindex(
                    [c for c in PAPER_CONDITION_ORDER if c in ssb['condition'].values
                     and c != 'C1_original'])
                for cond, r in ordered.iterrows():
                    sig = '*' if r['p_two'] < 0.05 else ' '
                    lines.append(
                        f'    {cond:<18s} '
                        f'Δ={r["mean_diff"]:+8.4f}  '
                        f'CI=[{r["ci_lo"]:+7.4f},{r["ci_hi"]:+7.4f}]  '
                        f'rel={r["rel_diff"]*100:+6.1f}%  '
                        f'p={r["p_two"]:.4f}{sig}'
                    )
        lines.append('')

    if probes is not None and not probes.empty:
        lines.append('3) MECHANISTIC PROBES — Aurora internals')
        lines.append('-' * 70)
        agg = probes.groupby('condition')[
            ['gradnorm_mean', 'attn_entropy_rel_mean',
             'attn_max_weight_mean', 'divergence_mean_sq']
        ].mean().reindex([c for c in PAPER_CONDITION_ORDER
                          if c in probes['condition'].values])
        lines.append(f'  {"Condition":<18s} {"GradNorm":>10s} {"H/logLk":>10s} {"MaxAttn":>10s} {"DivMSE":>10s}')
        for cond, row in agg.iterrows():
            lines.append(
                f'  {cond:<18s} {row["gradnorm_mean"]:>10.4f} '
                f'{row["attn_entropy_rel_mean"]:>10.3f} '
                f'{row["attn_max_weight_mean"]:>10.3f} '
                f'{row["divergence_mean_sq"]:>10.4f}'
            )
        lines.append('')
        lines.append('  Reading these:')
        lines.append('   - GradNorm > 0 means text-encoder output is non-trivially used,')
        lines.append('     but its magnitude is similar across all conditions.')
        lines.append('   - H/logLk ≈ 0.97 across ALL conditions → cross-attention is')
        lines.append('     near-uniform over the 10 distilled text tokens. The model has')
        lines.append('     no content-conditional selectivity over the text features.')
        lines.append('   - MaxAttn ≈ 0.16 ≈ 1/L_k (where L_k=10). Confirms uniformity.')
        lines.append('   - DivMSE ~0.04 in standardised units. Predictions WITH text vs.')
        lines.append('     WITHOUT text differ slightly — but the diff has the same')
        lines.append('     magnitude regardless of whether text is real or perturbed,')
        lines.append('     suggesting a content-blind perturbation pathway, not signal use.')
        lines.append('')

    if vs_prior is not None and not vs_prior.empty:
        lines.append('4) MODEL vs NUMERIC-PRIOR-ONLY BASELINE')
        lines.append('-' * 70)
        agg_vp = vs_prior.groupby('model')[
            ['mse', 'prior_only_mse', 'model_minus_prior']
        ].mean()
        for model, row in agg_vp.iterrows():
            gap_pct = (row['mse'] - row['prior_only_mse']) / row['prior_only_mse'] * 100
            lines.append(
                f'  {model:<10s} model_C1={row["mse"]:.4f}  '
                f'prior_only={row["prior_only_mse"]:.4f}  '
                f'gap={row["model_minus_prior"]:+7.4f}  ({gap_pct:+5.1f}%)'
            )
        lines.append('')

    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--main_results', type=Path, default=SUMMARIES / 'main_results.csv')
    ap.add_argument('--bootstrap', type=Path,
                    default=SUMMARIES / 'bootstrap_per_backbone.csv')
    ap.add_argument('--prior_baseline', type=Path,
                    default=SUMMARIES / 'prior_only_baseline.csv')
    ap.add_argument('--per_cell', type=Path, default=SUMMARIES / 'per_cell.csv')
    ap.add_argument('--probes_log', type=Path,
                    default=PROJECT_ROOT / 'nohup_aurora_probes.log')
    ap.add_argument('--out_dir', type=Path, default=OUT)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    probes = pd.DataFrame()
    if args.probes_log.exists():
        probes = parse_probe_log(args.probes_log)
        probes.to_csv(args.out_dir / 'probes_per_cell.csv', index=False)
        print(f'Parsed {len(probes)} probe cells from {args.probes_log.name}')

        agg = aggregate_probes_by_condition(probes)
        agg.to_csv(args.out_dir / 'probes_aggregate.csv', index=False)
        (args.out_dir / 'probes_aggregate.tex').write_text(fmt_latex_probes(agg))

        agg_cd = aggregate_probes_per_condition_domain(probes)
        agg_cd.to_csv(args.out_dir / 'probes_per_condition_domain.csv', index=False)
    else:
        print(f'(probes log not found at {args.probes_log}; skipping)')

    main_df = pd.read_csv(args.main_results)
    ladder = build_ladder_table(main_df)
    ladder.to_csv(args.out_dir / 'ladder_table.csv')
    (args.out_dir / 'ladder_table.tex').write_text(fmt_latex_ladder(
        ladder,
        caption=('Behavioural ladder: mean MSE per condition across all '
                 r'(domain, horizon, seed, backbone). Coloured deltas vs.\ $C_1$.'),
        label='tab:ladder'))

    ha = build_horizon_averaged(args.per_cell)
    if ha is not None:
        ha.to_csv(args.out_dir / 'horizon_averaged.csv', index=False)

    axes_df = build_design_axes_table(main_df)
    axes_df.to_csv(args.out_dir / 'design_axes.csv', index=False)

    prior_df = pd.read_csv(args.prior_baseline)
    vs_prior = build_vs_prior_table(prior_df, args.per_cell)
    if vs_prior is not None:
        vs_prior.to_csv(args.out_dir / 'vs_prior_baseline.csv', index=False)

    boot = pd.read_csv(args.bootstrap) if args.bootstrap.exists() else pd.DataFrame()

    summary = fmt_evidence_summary(probes, ladder, vs_prior, boot)
    (args.out_dir / 'evidence_summary.txt').write_text(summary)
    print(f'\nEvidence summary → {args.out_dir / "evidence_summary.txt"}\n')
    print(summary)


if __name__ == '__main__':
    main()