"""
runners/mmtsflib_runner.py
==========================

Runs MM-TSFlib on a single cell.

MM-TSFlib specifics:
* Late-fusion architecture: output = (1-prompt_weight)*TS_output +
  prompt_weight*(norm(text_emb) + prior_y). The prior_y comes from
  the `prior_history_avg` column (a closed-LLM-derived numeric
  forecast).
* C6 unimodal: prompt_weight=0. Zeroes both the LLM embedding
  contribution AND the prior_y addend, leaving pure TS backbone output.
* Uses features='S' (we verified features='M' crashes on string
  columns in shipped code — all 9 domains fail). All our perturbation
  CSVs work with features='S'.
* LLM: default BERT (frozen). Reference script uses text_len=4
  (reads Final_Search_4).
* Reference hyperparameters (scripts/week_health.sh):
    seed=2021, text_len=4, prompt_weight=0.1, seq_len=24, label_len=12,
    pred_len in {12,24,36,48}, llm_model=BERT, features=M [BROKEN, we
    use S].
  These can be overridden per-cell via RunSpec.seq_len / label_len; the
  orchestrator's --preset time_mmd flag sets per-domain paper values.

Backbones:
* MM-TSFlib registers 22 backbones in exp/exp_basic.py (out-of-box,
  no patch needed — unlike TaTS). See MMTSFLIB_ALL_BACKBONES below.
* Time-MMD paper uses 10 of these (the 10 in `PAPER_BACKBONES`).

Checkpoints:
* MM-TSFlib saves a checkpoint.pth at repos/MM-TSFlib/checkpoints/<setting>/
  during training (via early_stopping). The runner parses the setting
  string from stdout and records the checkpoint path in
  RunResult.extra['checkpoint_path'] so future probe scripts can
  reload the trained model without re-training.

Compute: similar to TaTS, ~15-30 sec/cell on an A10G for small domains,
several minutes for Environment.
"""

from __future__ import annotations

import os
import time

from runners.common import (
    RunSpec, RunResult, REPOS,
    resolve_data_path, parse_metrics_from_log,
    run_subprocess, tail, now_utc,
    mark_running, full_log_path, collect_provenance,
)

MMTSFLIB_REPO = REPOS / 'MM-TSFlib'

# All backbones available in MM-TSFlib/exp/exp_basic.py.
# For the paper's main table we'll run all of them; each cell's backbone is
# controlled by spec.extra_args['backbone']. If absent, we default to
# Informer (week_health.sh's choice).
MMTSFLIB_ALL_BACKBONES = [
    'Informer', 'Autoformer', 'Transformer', 'Nonstationary_Transformer',
    'DLinear', 'FEDformer', 'TimesNet', 'LightTS', 'Reformer', 'ETSformer',
    'PatchTST', 'Pyraformer', 'MICN', 'Crossformer', 'FiLM', 'iTransformer',
    'Koopa', 'TiDE', 'FreTS', 'TimeMixer', 'TSMixer', 'SegRNN',
]
DEFAULT_BACKBONE = 'Informer'


def run_mmtsflib(spec: RunSpec) -> RunResult:
    mark_running(spec)
    prov = collect_provenance(spec.model)
    result = RunResult(spec=spec, success=False, started_at_utc=now_utc(),
                       working_dir=str(MMTSFLIB_REPO), **prov)
    t0 = time.monotonic()

    root_path, data_file = resolve_data_path(spec)
    if not (root_path / data_file).exists():
        result.error = f'perturbed CSV not found: {root_path / data_file}'
        return result

    backbone = spec.backbone or spec.extra_args.get('backbone', DEFAULT_BACKBONE)
    text_len = spec.extra_args.get('text_len', 4)              # reads Final_Search_4
    llm_model = spec.extra_args.get('llm_model', 'BERT')
    train_epochs = spec.extra_args.get('train_epochs', 10)
    patience = spec.extra_args.get('patience', 3)
    # Batch size: default 32 (the repo's own default; matches your pre-existing
    # runs). Overridable via --batch_size on the orchestrator.
    # We DO NOT pass --num_workers so the repo default (10) is used, preserving
    # bit-level compatibility with pre-existing completed cells.
    batch_size = spec.extra_args.get('batch_size', 32)

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
        '--batch_size', str(batch_size),
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

    log_p = full_log_path(spec)
    log_p.parent.mkdir(parents=True, exist_ok=True)
    log_p.write_text(f'=== STDOUT ===\n{stdout}\n\n=== STDERR ===\n{stderr}\n')
    result.stdout_log_path = str(log_p)

    # Record the setting/checkpoint path so we can probe the trained model
    # later. MM-TSFlib writes checkpoints to <repo>/checkpoints/<setting>/.
    import re, shutil
    m = re.search(r'>>>>>>>start training : ([^>]+)>>>', stdout)
    setting_dir = None
    if m:
        setting = m.group(1).strip()
        ckpt_path = MMTSFLIB_REPO / 'checkpoints' / setting / 'checkpoint.pth'
        setting_dir = MMTSFLIB_REPO / 'checkpoints' / setting
        result.extra['checkpoint_path'] = str(ckpt_path)
        result.extra['training_setting'] = setting

    # Checkpoint cleanup — default is to delete (saving all cells' ckpts
    # would exceed 1 TB). Pass --preserve_checkpoints to keep.
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