"""
runners/tats_runner.py
======================

Runs TaTS (text-as-time-series covariates) on a single cell. Unlike Aurora,
TaTS trains (default 5 epochs) before eval.

TaTS specifics:
* Architecture: backbone (iTransformer by default) takes concat([TS, text_emb
  projected channels]). enc_in = 1 + text_emb. With text_emb=0, becomes pure
  univariate backbone — which is our C6 unimodal configuration.
* Precomputes LLM embeddings at data-load time using a frozen LLM (default
  GPT-2). Heavy on first load, cheap on subsequent batches.
* C6: --text_emb 0 --prior_weight 0 (both flags to zero out the text branch
  AND the numeric prior fusion).
* Reference hyperparameters from scripts/main_forecast.sh:
    seq_len=24, label_len=12, pred_len=48, text_emb=12, prior_weight=0.5,
    train_epochs=5, patience=5.

Compute: each cell ~5-15 minutes on a GPU depending on domain size.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from runners.common import (
    RunSpec, RunResult, REPOS,
    resolve_data_path, parse_metrics_from_log,
    run_subprocess, tail, now_utc,
    mark_running, full_log_path, collect_provenance,
)

TATS_REPO = REPOS / 'TaTS'

# All backbones available in TaTS/models/.
# Each cell's backbone is controlled by spec.extra_args['backbone']. Default
# = iTransformer (main_forecast.sh's choice).
TATS_ALL_BACKBONES = [
    'iTransformer', 'Autoformer', 'Transformer', 'DLinear',
    'FEDformer', 'Informer', 'PatchTST', 'Crossformer', 'FiLM',
]
DEFAULT_BACKBONE = 'iTransformer'


def run_tats(spec: RunSpec) -> RunResult:
    """Invoke TaTS on the specified cell."""
    mark_running(spec)
    prov = collect_provenance(spec.model)
    result = RunResult(spec=spec, success=False, started_at_utc=now_utc(),
                       working_dir=str(TATS_REPO), **prov)
    t0 = time.monotonic()

    root_path, data_file = resolve_data_path(spec)
    if not (root_path / data_file).exists():
        result.error = f'perturbed CSV not found: {root_path / data_file}'
        return result

    backbone = spec.backbone or spec.extra_args.get('backbone', DEFAULT_BACKBONE)
    train_epochs = spec.extra_args.get('train_epochs', 5)
    patience = spec.extra_args.get('patience', 5)
    llm_model = spec.extra_args.get('llm_model', 'GPT2')
    hug_token = spec.extra_args.get('huggingface_token', 'NA')

    # C6 unimodal operationalization: zero text_emb and prior_weight.
    # Both are necessary: text_emb=0 makes enc_in=dec_in=1 (univariate backbone),
    # and prior_weight=0 disables the numeric-prior blend.
    if spec.condition == 'C6_unimodal':
        text_emb = 0
        prior_weight = 0.0
    else:
        text_emb = spec.extra_args.get('text_emb', 12)
        prior_weight = spec.extra_args.get('prior_weight', 0.5)

    # Model ID used by TaTS for checkpoint naming — include our cell_id so
    # different conditions don't stomp each other's checkpoints.
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
        '--seq_len', str(spec.seq_len or 24),
        '--label_len', str(spec.label_len or 12),
        '--pred_len', str(spec.pred_len),
        '--text_emb', str(text_emb),
        '--des', 'Exp',
        '--seed', str(spec.seed),
        '--prior_weight', str(prior_weight),
        '--save_name', f'result_{spec.cell_id()}',
        '--llm_model', llm_model,
        '--huggingface_token', hug_token,
        '--train_epochs', str(train_epochs),
        '--patience', str(patience),
    ]
    result.cli_args = cli

    returncode, stdout, stderr = run_subprocess(
        cli, cwd=TATS_REPO,
        env_extra={'CUDA_VISIBLE_DEVICES':
                   os.environ.get('CUDA_VISIBLE_DEVICES', '0')},
        timeout=spec.extra_args.get('timeout', 7200),
    )
    result.stdout_tail = tail(stdout, 60)
    result.stderr_tail = tail(stderr, 60)

    log_p = full_log_path(spec)
    log_p.parent.mkdir(parents=True, exist_ok=True)
    log_p.write_text(f'=== STDOUT ===\n{stdout}\n\n=== STDERR ===\n{stderr}\n')
    result.stdout_log_path = str(log_p)

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
