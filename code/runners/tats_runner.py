"""
runners/tats_runner.py
======================

Runs TaTS (text-as-time-series covariates) on a single cell. Unlike Aurora,
TaTS trains (default 5 epochs) before eval.

TaTS specifics:
* Architecture: backbone (iTransformer by default) takes concat([TS,
  text_emb projected channels]). enc_in = 1 + text_emb. With text_emb=0,
  the backbone becomes pure univariate — our C6 unimodal configuration.
* Precomputes LLM embeddings at data-load time using a frozen LLM
  (default GPT-2). Heavy on first load, cheap on subsequent batches.
* C6 unimodal: --text_emb 0 --prior_weight 0 (zero out text branch AND
  numeric prior fusion).
* Reference hyperparameters from scripts/main_forecast.sh:
    seq_len=24, label_len=12, pred_len=48, text_emb=12, prior_weight=0.5,
    train_epochs=5, patience=5.
  These can be overridden per-cell via RunSpec.seq_len / label_len; the
  orchestrator's --preset time_mmd flag sets per-domain paper values.

Backbones:
* TaTS's own exp_basic.py only registers `iTransformer` in model_dict.
* Our apply_repo_patches.py patch adds 8 more: Autoformer, DLinear,
  FEDformer, FiLM, Informer, PatchTST, Transformer, Crossformer.
* TATS_ALL_BACKBONES (this module) enumerates all 9 after patch.

Checkpoints:
* TaTS saves a checkpoint.pth at repos/TaTS/checkpoints/<setting>/
  during training (via early_stopping). The runner parses the setting
  string from stdout and records the checkpoint path in
  RunResult.extra['checkpoint_path'] so future probe scripts can
  reload the trained model without re-training.

Compute: each cell ~15-30 seconds on an A10G for small domains,
up to ~2 minutes for Environment.
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

# All backbones registered in TaTS/exp/exp_basic.py AFTER our patch.
# The original TaTS ships only iTransformer in model_dict; our patch adds
# the rest. If patches aren't applied, only iTransformer will work.
TATS_ALL_BACKBONES = [
    'iTransformer', 'Autoformer', 'DLinear', 'FEDformer',
    'FiLM', 'Informer', 'PatchTST', 'Transformer', 'Crossformer',
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
    # Batch size: default 32 (the repo's own default; matches your pre-existing
    # runs). Overridable via --batch_size on the orchestrator. When sharding
    # with multiple shards, keeping bs=32 avoids OOM since each shard needs
    # its own activation memory.
    #
    # num_workers: we DO NOT pass --num_workers to TaTS, so the repo default
    # (10) is used. Overriding num_workers changes the DataLoader worker
    # seeding, which shifts batches in the first few epochs even with the
    # same torch.manual_seed. Keeping the default preserves bit-level
    # compatibility with your pre-existing completed cells.
    batch_size = spec.extra_args.get('batch_size', 32)

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
        '--batch_size', str(batch_size),
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

    # Record the setting string (from stdout) so we can find the saved
    # checkpoint later for probing. TaTS training always writes
    # checkpoints to <repo>/checkpoints/<setting>/checkpoint.pth.
    import re, shutil
    m = re.search(r'>>>>>>>start training : ([^>]+)>>>', stdout)
    setting_dir = None
    if m:
        setting = m.group(1).strip()
        ckpt_path = TATS_REPO / 'checkpoints' / setting / 'checkpoint.pth'
        setting_dir = TATS_REPO / 'checkpoints' / setting
        result.extra['checkpoint_path'] = str(ckpt_path)
        result.extra['training_setting'] = setting

    # Checkpoint cleanup. At ~60-170 MB per checkpoint and 7000+ cells,
    # keeping everything would need >1 TB. Default is to delete after each
    # cell. Pass --preserve_checkpoints on the orchestrator to keep them
    # (necessary if you plan to probe the trained models).
    preserve_ckpt = spec.extra_args.get('preserve_checkpoints', False)
    if setting_dir is not None and not preserve_ckpt:
        try:
            if setting_dir.exists():
                shutil.rmtree(setting_dir)
            result.extra['checkpoint_cleaned'] = True
        except Exception as e:
            result.extra['checkpoint_cleanup_error'] = str(e)

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