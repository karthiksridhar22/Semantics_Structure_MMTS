"""
run_experiments.py
==================

Orchestrator: iterates the grid of (model, backbone, condition, seed,
domain, pred_len), calls the right runner, saves results, supports
resume after crashes.

USAGE
-----
  # Pilot: one model, one condition, one domain, one horizon
  python run_experiments.py --models aurora --conditions C1_original \
      --domains Economy --pred_lens 8 --seeds 2021

  # Full grid with Time-MMD paper horizons (per-domain):
  #   Environment:         pred_lens = 48, 96, 192, 336  (seq_len=96)
  #   Health, Energy:      pred_lens = 12, 24, 36, 48    (seq_len=36)
  #   Other 6 monthly doms: pred_lens = 6, 8, 10, 12     (seq_len=8)
  #
  # For Aurora: seq_len stays at Aurora's native per-domain defaults
  # (Aurora is zero-shot; its seq_len is a pretrained-context choice,
  # not a benchmark knob). Only pred_lens from the preset are applied.
  python run_experiments.py --models tats --preset time_mmd \
      --backbones iTransformer DLinear Autoformer \
      --conditions C1_original C2_empty C3_shuffled C4_crossdomain \
                   C5_constant C6_unimodal C7_null C8_oracle \
      --domains Agriculture Climate Economy Energy Environment Health \
                Security SocialGood Traffic \
      --seeds 2021 2022 2023

  # Resume is automatic: cells with a successful result JSON are skipped.
  # .running markers from crashed runs are auto-cleared on startup.
  python run_experiments.py ... (same command as before; just re-run)

  # Force re-run (ignore existing results)
  python run_experiments.py ... --force

  # Dry-run to check grid size before committing GPU time
  python run_experiments.py ... --dry_run

BACKBONE HANDLING
-----------------
* Aurora has no backbone axis (arch is fixed by pretrained weights).
* TaTS and MM-TSFlib: default is each model's paper-default (iTransformer
  for TaTS, Informer for MM-TSFlib). Use --backbones X Y Z for an explicit
  subset, or --all_backbones to iterate every backbone registered in each
  repo's model_dict.

CHECKPOINTING & RESUME
----------------------
* Every result is one JSON file at:
    results/<model>/<backbone>/<condition>/seed<s>/<domain>_h<pred_len>.json
* Atomic writes: result JSONs are written to .tmp then renamed (so a
  crash mid-write never leaves a corrupt file).
* In-progress marker: while a cell is running, a sibling `.running` file
  is present. On startup, leftover markers from crashed runs are cleared,
  and those cells are retried.
* Per-cell full stdout/stderr saved to:
    logs/<model>/<backbone>/<condition>/seed<s>/<domain>_h<pred_len>.log
* Global append-only progress log: `sweep_log.jsonl` (one line per cell).

CONDITIONS
----------
* C1 original, C2 empty, C3 shuffled (seed-dependent), C4 crossdomain
  (date-aligned, deterministic), C5 constant, C7 null, C8 oracle — all
  are CSV-level perturbations.
* C6 unimodal has no perturbation CSV; the runner uses C1's CSV and sets
  the model's own unimodal flag (--no_text for Aurora, prompt_weight=0
  for MM-TSFlib, text_emb=0 for TaTS).

Failures DO NOT abort the sweep — they're logged and we continue. A
summary at the end lists what failed with log file paths for debugging.

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

# Time-MMD paper horizon & window presets (arXiv:2406.08627, Section "Setup").
# Keyed by domain → (seq_len, label_len, [pred_lens]).
#   Daily    (Environment):       seq_len=96, label_len=48, pred_lens=[48,96,192,336]
#   Weekly   (Health, Energy):    seq_len=36, label_len=18, pred_lens=[12,24,36,48]
#   Monthly  (everything else):   seq_len=8,  label_len=4,  pred_lens=[6,8,10,12]
TIME_MMD_PRESET = {
    'Environment': (96, 48, [48, 96, 192, 336]),
    'Health':      (36, 18, [12, 24, 36, 48]),
    'Energy':      (36, 18, [12, 24, 36, 48]),
    'Agriculture': (8, 4, [6, 8, 10, 12]),
    'Climate':     (8, 4, [6, 8, 10, 12]),
    'Economy':     (8, 4, [6, 8, 10, 12]),
    'Security':    (8, 4, [6, 8, 10, 12]),
    'SocialGood':  (8, 4, [6, 8, 10, 12]),
    'Traffic':     (8, 4, [6, 8, 10, 12]),
}

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
                backbones_per_model, extra_args=None,
                use_time_mmd_preset=False) -> list[RunSpec]:
    """Build the flat list of RunSpecs to execute.

    If `use_time_mmd_preset=True`:
      - TaTS and MM-TSFlib: each domain uses the Time-MMD paper's full
        preset (seq_len, label_len, pred_lens).
      - Aurora: each domain uses the paper's pred_lens, but seq_len/label_len
        stay at Aurora's own per-domain defaults. Aurora is pretrained with a
        specific context-window behavior; using seq_len=8 for monthly data
        would starve it of context and produce degenerate zero-shot results.

    Otherwise, pred_lens is applied uniformly across domains and
    seq_len/label_len use each runner's default.
    """
    extra_args = extra_args or {}
    specs = []
    for model in models:
        seeds = seeds_per_model.get(model, DEFAULT_SEEDS[model])
        bbs = backbones_per_model.get(model, DEFAULT_BACKBONES[model])
        for cond in conditions:
            for seed in seeds:
                for domain in domains:
                    if use_time_mmd_preset:
                        if domain not in TIME_MMD_PRESET:
                            raise ValueError(f'no Time-MMD preset for {domain}')
                        preset_seq, preset_lab, domain_pred_lens = TIME_MMD_PRESET[domain]
                        # Aurora keeps its own seq_len (pretrained context);
                        # the preset only contributes pred_lens for Aurora.
                        if model == 'aurora':
                            seq_len = None   # runner uses its own default
                            label_len = None
                        else:
                            seq_len = preset_seq
                            label_len = preset_lab
                    else:
                        seq_len = None
                        label_len = None
                        domain_pred_lens = pred_lens
                    for h in domain_pred_lens:
                        for bb in bbs:
                            specs.append(RunSpec(
                                model=model, condition=cond, seed=seed,
                                domain=domain, pred_len=h,
                                seq_len=seq_len, label_len=label_len,
                                backbone=bb,
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
    ap.add_argument('--pred_lens', nargs='+', type=int, default=[8],
                    help='Ignored if --preset time_mmd is used')
    ap.add_argument('--preset', choices=['time_mmd'], default=None,
                    help='Apply per-domain horizon preset. time_mmd uses '
                         'the Time-MMD paper\'s horizon groups: '
                         'daily(Environment)=[48,96,192,336], '
                         'weekly(Health,Energy)=[12,24,36,48], '
                         'monthly(rest)=[6,8,10,12]. Also sets per-domain '
                         'seq_len and label_len to the paper\'s defaults.')
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
    ap.add_argument('--shard_id', type=int, default=0,
                    help='Which shard of the grid this worker handles '
                         '(0-indexed). Use with --num_shards to run '
                         'multiple orchestrators in parallel, each on a '
                         'disjoint subset of cells.')
    ap.add_argument('--num_shards', type=int, default=1,
                    help='Total number of shards. Each cell goes to shard '
                         '(cell_idx % num_shards). Default 1 = no sharding.')
    ap.add_argument('--batch_size', type=int, default=None,
                    help='Override batch size for TaTS and MM-TSFlib. '
                         'Default: 32 (runners\' internal default). '
                         'Use smaller (e.g. 16) if you hit OOM; larger '
                         '(e.g. 64, 128) if your GPU has more headroom.')
    ap.add_argument('--preserve_checkpoints', action='store_true',
                    help='Keep trained model checkpoints on disk. Default: '
                         'delete after each cell. ~60-170 MB/cell × 15k+ '
                         'cells = >1 TB if kept. Enable only for a small '
                         'subset you plan to probe later.')
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

    extra_args = {}
    if args.batch_size is not None:
        extra_args['batch_size'] = args.batch_size
    if args.preserve_checkpoints:
        extra_args['preserve_checkpoints'] = True

    specs = build_specs(args.models, args.conditions, seeds_per_model,
                        args.domains, args.pred_lens,
                        backbones_per_model,
                        extra_args=extra_args,
                        use_time_mmd_preset=(args.preset == 'time_mmd'))

    # Apply sharding: keep only cells whose index modulo num_shards equals
    # shard_id. This lets N workers run in parallel on disjoint subsets of
    # the grid. We shard AFTER build_specs so all workers agree on the
    # global cell order (deterministic) before partitioning.
    if args.num_shards > 1:
        if not (0 <= args.shard_id < args.num_shards):
            raise ValueError(f'--shard_id must be in [0, {args.num_shards})')
        specs = [s for i, s in enumerate(specs)
                 if i % args.num_shards == args.shard_id]
        print(f'Shard {args.shard_id}/{args.num_shards}: '
              f'handling {len(specs)} cells of the global grid')

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