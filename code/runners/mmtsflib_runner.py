"""
runners/mmtsflib_runner.py
==========================

Runs MM-TSFlib on a single cell.

MM-TSFlib specifics:
* Late-fusion architecture: output = (1-prompt_weight)*TS + prompt_weight*
  (norm(text_emb) + prior_y). The prior_y comes from the `prior_history_avg`
  column (a closed-LLM-derived numeric forecast).
* C6 unimodal: prompt_weight=0. This zeroes both the LLM embedding contribution
  AND the prior_y addend, leaving pure TS backbone output.
* Uses features='S' (we verified features='M' crashes on string columns in
  shipped code). All our perturbation CSVs work with features='S'.
* LLM: default BERT (frozen). Reference script uses text_len=4
  (reads Final_Search_4).
* Reference hyperparameters (scripts/week_health.sh):
    seed=2021, text_len=4, prompt_weight=0.1, seq_len=24, label_len=12,
    pred_len in {12,24,36,48}, llm_model=BERT, features=M [BROKEN].

Compute: similar to TaTS, ~5-15 min/cell on a GPU.
"""

from __future__ import annotations

import os
import time

from runners.common import (
    RunSpec, RunResult, REPOS,
    resolve_data_path, parse_metrics_from_log,
    run_subprocess, tail, now_utc,
)

MMTSFLIB_REPO = REPOS / 'MM-TSFlib'

DEFAULT_BACKBONE = 'Informer'   # choose one; week_health.sh uses this


def run_mmtsflib(spec: RunSpec) -> RunResult:
    result = RunResult(spec=spec, success=False, started_at_utc=now_utc(),
                       working_dir=str(MMTSFLIB_REPO))
    t0 = time.monotonic()

    root_path, data_file = resolve_data_path(spec)
    if not (root_path / data_file).exists():
        result.error = f'perturbed CSV not found: {root_path / data_file}'
        return result

    backbone = spec.extra_args.get('backbone', DEFAULT_BACKBONE)
    text_len = spec.extra_args.get('text_len', 4)              # reads Final_Search_4
    llm_model = spec.extra_args.get('llm_model', 'BERT')
    train_epochs = spec.extra_args.get('train_epochs', 10)
    patience = spec.extra_args.get('patience', 3)

    # C6 operationalization: prompt_weight=0.
    if spec.condition == 'C6_unimodal':
        prompt_weight = 0.0
    else:
        prompt_weight = spec.extra_args.get('prompt_weight', 0.1)

    model_id = f'probe_{spec.cell_id()}'

    cli = [
        'python', '-u', 'run.py',
        '--task_name', 'long_term_forecast',
        '--is_training', '1',
        '--root_path', str(root_path),
        '--data_path', data_file,
        '--model_id', model_id,
        '--model', backbone,
        '--data', 'custom',
        '--features', 'S',          # critical — 'M' crashes on string cols
        '--seq_len', str(spec.seq_len or 24),
        '--label_len', str(spec.label_len or 12),
        '--pred_len', str(spec.pred_len),
        '--text_len', str(text_len),
        '--prompt_weight', str(prompt_weight),
        '--llm_model', llm_model,
        '--use_closedllm', '0',     # read Final_Search_N, not Final_Output
        '--seed', str(spec.seed),
        '--train_epochs', str(train_epochs),
        '--patience', str(patience),
        '--des', 'Exp',
    ]
    result.cli_args = cli

    returncode, stdout, stderr = run_subprocess(
        cli, cwd=MMTSFLIB_REPO,
        env_extra={'CUDA_VISIBLE_DEVICES':
                   os.environ.get('CUDA_VISIBLE_DEVICES', '0')},
        timeout=spec.extra_args.get('timeout', 7200),
    )
    result.stdout_tail = tail(stdout, 60)
    result.stderr_tail = tail(stderr, 60)

    if returncode != 0:
        result.error = f'process exited with code {returncode}'
        result.wall_time_seconds = time.monotonic() - t0
        return result

    metrics = parse_metrics_from_log(stdout + '\n' + stderr)
    result.mse = metrics['mse']
    result.mae = metrics['mae']
    result.smape = metrics['smape']
    result.success = (result.mse is not None and result.mae is not None)
    if not result.success:
        result.error = 'could not parse MSE/MAE from log'

    result.wall_time_seconds = time.monotonic() - t0
    return result
