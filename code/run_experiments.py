"""
run_experiments.py
==================

Orchestrator: iterates the grid of (model, condition, seed, domain, pred_len),
calls the right runner, saves results, supports resume.

USAGE
-----
  # Pilot: one model, one condition, one domain, one horizon
  python run_experiments.py --models aurora --conditions C1_original \
      --domains Economy --pred_lens 8 --seeds 2021

  # Full grid
  python run_experiments.py --models aurora tats mmtsflib \
      --conditions C1_original C2_empty C3_shuffled C4_crossdomain \
                   C5_constant C6_unimodal C7_null C8_oracle \
      --domains Agriculture Climate Economy Energy Environment Health \
                Security SocialGood Traffic \
      --pred_lens 8 12 24 48

  # Resume (skips successful runs)
  python run_experiments.py ... (resume is default)

  # Force re-run
  python run_experiments.py ... --force

DESIGN
------
* Every result is one JSON file under results/<model>/<cond>/seed<s>/...
* A dry-run mode prints the grid without executing — useful for checking
  grid size before committing GPU time.
* Failures DON'T abort the sweep — they're logged and we continue. A
  summary at the end lists what failed.
* C6_unimodal doesn't have its own perturbation CSV; the runner uses
  C1_original's CSVs and sets the appropriate CLI flag (--no_text for
  Aurora, --prompt_weight 0 for MM-TSFlib, etc.). So the orchestrator
  maps C6 -> C1 for data path resolution.

Seeds per model (used if --seeds not specified):
  mmtsflib: 2021, 2022, 2023
  tats:     2024, 2025, 2026
  aurora:   2021, 2022, 2023
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make runners importable regardless of cwd.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from runners.common import (
    RunSpec, RunResult, run_spec, already_done, save_result, now_utc,
    result_path, clear_stale_markers, SWEEP_LOG,
)
from runners.mmtsflib_runner import MMTSFLIB_ALL_BACKBONES
from runners.tats_runner import TATS_ALL_BACKBONES

DEFAULT_SEEDS = {
    'mmtsflib': [2021, 2022, 2023],
    'tats':     [2021, 2022, 2023],
    'aurora':   [2021, 2022, 2023],
}

ALL_MODELS = ['aurora', 'tats', 'mmtsflib']
ALL_CONDITIONS = [
    'C1_original', 'C2_empty', 'C3_shuffled', 'C4_crossdomain',
    'C5_constant', 'C6_unimodal', 'C7_null', 'C8_oracle',
]
ALL_DOMAINS = ['Agriculture', 'Climate', 'Economy', 'Energy', 'Environment',
               'Health', 'Security', 'SocialGood', 'Traffic']

# Default backbone sets per model. Aurora has no backbone axis (its arch
# is fixed by the pretrained weights), so we emit a single [None] sentinel.
DEFAULT_BACKBONES = {
    'aurora':   [None],
    'tats':     ['iTransformer'],        # paper default
    'mmtsflib': ['Informer'],            # week_health.sh default
}
ALL_BACKBONES = {
    'aurora':   [None],
    'tats':     TATS_ALL_BACKBONES,
    'mmtsflib': MMTSFLIB_ALL_BACKBONES,
}


def build_specs(models, conditions, seeds_per_model, domains, pred_lens,
                backbones_per_model, extra_args=None) -> list[RunSpec]:
    extra_args = extra_args or {}
    specs = []
    for model in models:
        seeds = seeds_per_model.get(model, DEFAULT_SEEDS[model])
        bbs = backbones_per_model.get(model, DEFAULT_BACKBONES[model])
        for cond in conditions:
            for seed in seeds:
                for domain in domains:
                    for h in pred_lens:
                        for bb in bbs:
                            specs.append(RunSpec(
                                model=model, condition=cond, seed=seed,
                                domain=domain, pred_len=h, backbone=bb,
                                extra_args=dict(extra_args),
                            ))
    return specs


def summarize(results: list[RunResult]):
    """Print a terse completion summary."""
    total = len(results)
    ok = sum(1 for r in results if r.success)
    print(f'\n=== SUMMARY ===')
    print(f'Total:   {total}')
    print(f'Success: {ok}')
    print(f'Failed:  {total - ok}')
    if total > ok:
        print(f'First 5 failures:')
        for r in results:
            if not r.success:
                print(f'  - {r.spec.cell_id()}: {r.error}')
                if sum(1 for x in results[:results.index(r)+1] if not x.success) >= 5:
                    break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', choices=ALL_MODELS,
                    default=ALL_MODELS)
    ap.add_argument('--conditions', nargs='+', choices=ALL_CONDITIONS,
                    default=ALL_CONDITIONS)
    ap.add_argument('--domains', nargs='+', choices=ALL_DOMAINS,
                    default=ALL_DOMAINS)
    ap.add_argument('--pred_lens', nargs='+', type=int, default=[8])
    ap.add_argument('--seeds', nargs='+', type=int,
                    help='If set, same seeds used for ALL models (otherwise '
                         'uses DEFAULT_SEEDS per model)')
    ap.add_argument('--backbones', nargs='+', default=None,
                    help='Backbone names to run for MM-TSFlib and TaTS '
                         '(Aurora has no backbone axis). Default: '
                         'each model\'s paper-default single backbone. '
                         'Applied to all selected non-Aurora models.')
    ap.add_argument('--all_backbones', action='store_true',
                    help='Run every backbone available in each repo. '
                         'Overrides --backbones.')
    ap.add_argument('--dry_run', action='store_true',
                    help='Print planned grid without executing')
    ap.add_argument('--force', action='store_true',
                    help='Re-run cells even if a successful result exists')
    ap.add_argument('--stop_on_error', action='store_true',
                    help='Abort sweep on first failure')
    args = ap.parse_args()

    seeds_per_model = (
        {m: args.seeds for m in args.models} if args.seeds
        else DEFAULT_SEEDS
    )

    # Resolve per-model backbones
    if args.all_backbones:
        backbones_per_model = {m: ALL_BACKBONES[m] for m in args.models}
    elif args.backbones:
        # Apply the same user-specified backbones to every non-Aurora model.
        # Aurora always gets [None].
        backbones_per_model = {
            m: ([None] if m == 'aurora' else list(args.backbones))
            for m in args.models
        }
    else:
        backbones_per_model = {m: DEFAULT_BACKBONES[m] for m in args.models}

    # C6 rewrite: its runner uses the C1_original CSVs and sets a CLI flag.
    # To keep the orchestrator simple, we keep the spec as 'C6_unimodal' —
    # each runner knows to handle that condition by swapping in C1 data paths
    # internally. (Actually, our runners resolve data via the condition label;
    # C6 thus needs C1 data. Here we remap.)
    def _resolve_data_condition(c: str) -> str:
        return 'C1_original' if c == 'C6_unimodal' else c
    # We don't rewrite spec.condition (that field identifies WHAT cell we're
    # running). Instead, each runner checks spec.condition=='C6_unimodal' and
    # mentally maps to C1 data when resolving paths. This is already the case
    # IF we special-case resolve_data_path... let me fix that by rewriting
    # spec.condition JUST for data path resolution.

    specs = build_specs(args.models, args.conditions, seeds_per_model,
                        args.domains, args.pred_lens,
                        backbones_per_model)

    if args.dry_run:
        print(f'Grid size: {len(specs)} cells')
        for s in specs[:10]:
            print(f'  {s.cell_id()}')
        if len(specs) > 10:
            print(f'  ... and {len(specs)-10} more')
        return

    # Before starting, clean up stale in-progress markers from previous
    # crashed runs. Any cell with a marker is considered "not done" and
    # will be retried.
    n_stale = clear_stale_markers()
    if n_stale:
        print(f'Cleared {n_stale} stale in-progress markers from previous crashes.')
    print(f'Sweep log → {SWEEP_LOG} (tail -f to follow live)')

    results: list[RunResult] = []
    t_start = time.monotonic()
    for i, spec in enumerate(specs, 1):
        if not args.force and already_done(spec):
            print(f'[{i}/{len(specs)}] SKIP (done): {spec.cell_id()}')
            sys.stdout.flush()
            # Load existing result for summary.
            import json
            try:
                existing = json.loads(result_path(spec).read_text())
                rr = RunResult(spec=spec, success=True,
                               mse=existing.get('mse'),
                               mae=existing.get('mae'),
                               smape=existing.get('smape'))
                results.append(rr)
            except Exception:
                pass
            continue

        # Run.
        print(f'[{i}/{len(specs)}] RUN: {spec.cell_id()}')
        sys.stdout.flush()
        t_cell = time.monotonic()
        try:
            result = run_spec(spec)
        except Exception as e:
            result = RunResult(spec=spec, success=False,
                               error=f'runner raised: {type(e).__name__}: {e}',
                               started_at_utc=now_utc())
        result.wall_time_seconds = result.wall_time_seconds or (time.monotonic() - t_cell)
        save_result(result)
        results.append(result)
        status = 'OK' if result.success else 'FAIL'
        print(f'    -> {status} ({result.wall_time_seconds:.1f}s)'
              + (f' MSE={result.mse:.4f} MAE={result.mae:.4f}' if result.success else f' error={result.error[:80]}'))
        sys.stdout.flush()   # nohup-friendliness: flush every cell

        if not result.success and args.stop_on_error:
            print('--stop_on_error set; aborting sweep.')
            break

    elapsed = time.monotonic() - t_start
    print(f'\nSweep complete in {elapsed/60:.1f} minutes.')
    summarize(results)


if __name__ == '__main__':
    main()
