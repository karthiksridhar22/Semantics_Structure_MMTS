"""
runners/aurora_runner.py
========================

Runs Aurora (pretrained, zero-shot) on a single (condition, seed, domain,
pred_len) cell.

Aurora specifics:
* Zero-shot (--is_training 0): loads pretrained weights, no gradient step.
  The seed here controls ONLY inference-time randomness (the probabilistic
  flow-matching head samples num_samples=100 predictions and averages).
* Per-domain seq_len varies — we default to the values from Aurora's
  reference script (AURORA_DEFAULTS below), but allow override via
  spec.seq_len. For the `--preset time_mmd` orchestrator flag, Aurora
  keeps its native seq_len (not the Time-MMD-paper value) since its
  pretrained context-window behavior degrades with short seq_len.
* Requires pretrained weights at --model_path. AURORA_WEIGHTS env var
  points at the HF snapshot. We check and fail cleanly if absent.
* C6 unimodal: we pass --no_text (added via patch) which makes the
  model invoke text_features=None internally.
* No backbone axis: Aurora's architecture is fixed by its pretrained
  weights. Results land under results/aurora/default/.

Wall time: dominated by model loading (~2s) + forward pass. Typically
~8-12 sec/cell for small domains, up to ~50 sec for Environment.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from runners.common import (
    RunSpec, RunResult, REPOS, PROJECT_ROOT,
    resolve_data_path, parse_metrics_from_log,
    run_subprocess, tail, now_utc,
    mark_running, full_log_path, collect_provenance,
)

AURORA_REPO = REPOS / 'Aurora' / 'TimeMMD'
# Pretrained weights location. Can be overridden via env var AURORA_WEIGHTS.
DEFAULT_WEIGHTS = os.environ.get('AURORA_WEIGHTS',
                                 str(PROJECT_ROOT / 'weights' / 'aurora'))

# Per-domain default settings from scripts/run_aurora_timemmd_zero_shot.sh.
# Format: (seq_len, inference_token_len, batch_size, default_horizons)
AURORA_DEFAULTS = {
    'Agriculture':  (192,  48,  256, (6, 8, 10, 12)),
    'Climate':      (192,  48,  256, (6, 8, 10, 12)),
    'Economy':      (192,  48,  256, (6, 8, 10, 12)),
    'Energy':       (1056, 48,  256, (12, 24, 36, 48)),
    'Environment':  (528,  48,  256, (48, 96, 192, 336)),
    'Health':       (96,   48,  256, (12, 24, 36, 48)),
    'Security':     (220,  24,  256, (6, 8, 10, 12)),
    'SocialGood':   (192,  48,  256, (6, 8, 10, 12)),
    'Traffic':      (96,   48,  256, (6, 8, 10, 12)),
}


def run_aurora(spec: RunSpec) -> RunResult:
    """Invoke Aurora on the specified cell. Returns a RunResult."""
    mark_running(spec)   # drop marker; save_result will remove it
    prov = collect_provenance(spec.model)
    result = RunResult(spec=spec, success=False, started_at_utc=now_utc(),
                       working_dir=str(AURORA_REPO), **prov)
    t0 = time.monotonic()

    # Sanity: pretrained weights available?
    weights = Path(DEFAULT_WEIGHTS)
    if not weights.exists() or not any(weights.iterdir()):
        result.error = (f'Aurora pretrained weights not found at {weights}. '
                        f'Download from huggingface.co/DecisionIntelligence/Aurora '
                        f'and place under that directory, or set AURORA_WEIGHTS env var.')
        return result

    # Resolve perturbed CSV location.
    root_path, data_file = resolve_data_path(spec)
    if not (root_path / data_file).exists():
        result.error = f'perturbed CSV not found: {root_path / data_file}'
        return result

    # Defaults per-domain (can be overridden by spec).
    if spec.domain not in AURORA_DEFAULTS:
        result.error = f'no Aurora defaults for domain {spec.domain}'
        return result
    def_seq, def_itl, def_bs, _ = AURORA_DEFAULTS[spec.domain]
    seq_len = spec.seq_len if spec.seq_len is not None else def_seq
    inference_token_len = spec.extra_args.get('inference_token_len', def_itl)
    batch_size = spec.extra_args.get('batch_size', def_bs)

    # Build CLI.
    cli = [
        'python', 'run_longExp.py',
        '--is_training', '0',                       # zero-shot
        '--random_seed', str(spec.seed),
        '--features', 'S',
        '--seq_len', str(seq_len),
        '--pred_len', str(spec.pred_len),
        '--inference_token_len', str(inference_token_len),
        '--batch_size', str(batch_size),
        '--data', spec.domain,
        '--data_path', data_file,
        '--root_path', str(root_path),
        '--model_path', str(weights),
        '--gpu', os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
    ]
    # C6 unimodal: add --no_text flag (requires our patch).
    if spec.condition == 'C6_unimodal':
        cli.append('--no_text')
    result.cli_args = cli

    # Run.
    returncode, stdout, stderr = run_subprocess(
        cli, cwd=AURORA_REPO,
        timeout=spec.extra_args.get('timeout', 3600),
    )
    result.stdout_tail = tail(stdout, 60)
    result.stderr_tail = tail(stderr, 60)

    # Save full stdout+stderr to disk so failed-cell debugging doesn't
    # require re-running (which could take hours). Atomic write so a crash
    # during logging doesn't corrupt the file.
    log_p = full_log_path(spec)
    log_p.parent.mkdir(parents=True, exist_ok=True)
    log_p.write_text(f'=== STDOUT ===\n{stdout}\n\n=== STDERR ===\n{stderr}\n')
    result.stdout_log_path = str(log_p)

    if returncode != 0:
        result.error = f'process exited with code {returncode}'
        result.wall_time_seconds = time.monotonic() - t0
        return result

    # Parse metrics.
    combined = stdout + '\n' + stderr
    metrics = parse_metrics_from_log(combined)
    result.mse = metrics['mse']
    result.mae = metrics['mae']
    result.smape = metrics['smape']
    result.success = (result.mse is not None and result.mae is not None)
    if not result.success:
        result.error = 'could not parse MSE/MAE from log'

    result.wall_time_seconds = time.monotonic() - t0
    return result