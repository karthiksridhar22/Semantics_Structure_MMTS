# Semantics or Structure? Probing Text Perturbations in Multimodal Time Series Models

Infrastructure for the ICML 2026 FMSD workshop submission *"Semantics or Structure? Disentangling What Text Contributes to Multimodal Time Series Models"*. Probes Aurora, TaTS, and MM-TSFlib on Time-MMD across eight text-perturbation conditions, nine domains, three seeds, and multiple backbones.

## Repository structure

```
├── README.md
├── code/
│   ├── generate_perturbations.py    Build perturbed CSVs (C1–C5, C7, C8)
│   ├── apply_repo_patches.py        Apply/revert 13 repo patches
│   ├── run_experiments.py           Orchestrator: grid, resume, sharding
│   ├── aurora_probes.py             Standalone Aurora diagnostic probes
│   ├── analyze_results.py           Aggregate results → CSV/LaTeX
│   └── runners/
│       ├── common.py                RunSpec/RunResult, paths, atomic I/O
│       ├── aurora_runner.py
│       ├── tats_runner.py
│       └── mmtsflib_runner.py
├── repos/                           Vendored upstream code (patched in place)
│   ├── Aurora/
│   ├── TaTS/
│   ├── MM-TSFlib/
│   └── Time-MMD/
├── weights/aurora/                  Aurora pretrained weights (~2 GB)
├── data/                            Generated perturbation CSVs (162 files)
│   ├── mmtsflib/
│   ├── tats/
│   └── manifest.json
├── results/<model>/<backbone>/<cond>/seed<s>/<domain>_h<h>.json
├── logs/<model>/<backbone>/<cond>/seed<s>/<domain>_h<h>.log
├── probes/aurora/                   Aurora probe outputs (separate from results)
└── sweep_log.jsonl                  Append-only global progress log
```

## Experimental design

### Perturbation conditions

| ID | Text | Numeric priors | Seed-dependent | Tests |
|---|---|---|---|---|
| C1 | Unmodified | preserved | no | Baseline |
| C2 | Empty string | zeroed | no | Text-derived signal at all |
| C3 | Within-domain shuffle | preserved | yes | Text encoder semantic content |
| C4 | Cross-domain, date-aligned | preserved | no | Domain-specificity of text |
| C5 | Constant string | zeroed | no | Token presence vs absence |
| C6 | Unimodal (CLI flag) | N/A | no | Paper's unimodal baseline |
| C7 | `"null"` | zeroed | no | Minimal non-empty token |
| C8 | Oracle (future OT) | preserved | no | Positive control |

C6 is activated by a CLI flag per repo (`--no_text` for Aurora, `prompt_weight=0` for MM-TSFlib, `text_emb=0 prior_weight=0` for TaTS) and reads the C1 CSV. Only C3 varies with seed; the other five CSV-level conditions are byte-identical across seeds and stored once under `seed0/`.

### Horizons and window sizes (Time-MMD paper, arXiv:2406.08627)

| Frequency | Domains | `seq_len` | `label_len` | `pred_lens` |
|---|---|---|---|---|
| Daily | Environment | 96 | 48 | 48, 96, 192, 336 |
| Weekly | Health, Energy | 36 | 18 | 12, 24, 36, 48 |
| Monthly | Agriculture, Climate, Economy, Security, SocialGood, Traffic | 8 | 4 | 6, 8, 10, 12 |

Applied via `--preset time_mmd`. TaTS and MM-TSFlib use all three values; Aurora takes only the `pred_lens` (seq_len stays at Aurora's native per-domain defaults since it is zero-shot).

### Seeds

`{2021, 2022, 2023}`. Seed 2021 matches MM-TSFlib's reference script; the other two are for paired-bootstrap variance estimates.

### Backbones

| Model | Count | Members |
|---|---|---|
| Aurora | 1 | fixed architecture (pretrained) |
| TaTS (after patch) | 9 | iTransformer, Autoformer, DLinear, FEDformer, FiLM, Informer, PatchTST, Transformer, Crossformer |
| MM-TSFlib | 22 | Informer, Autoformer, Transformer, Nonstationary_Transformer, DLinear, FEDformer, TimesNet, LightTS, Reformer, ETSformer, PatchTST, Pyraformer, MICN, Crossformer, FiLM, iTransformer, Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN |

Time-MMD paper's comparable subset:
- MM-TSFlib: Informer, Autoformer, Transformer, Nonstationary_Transformer, DLinear, FEDformer, Reformer, Crossformer, iTransformer, FiLM (10)
- TaTS: Informer, Autoformer, Transformer, DLinear, FEDformer, Crossformer, iTransformer, FiLM, PatchTST (9)

### Metric outputs

Each successful run writes a JSON with `mse`, `mae`, provenance (hostname, git SHAs, python/torch versions), CLI args, and `extra.training_setting` (for reference when probing specific cells later).

## Environment setup

Three separate conda envs (conflicting pin requirements).

### Aurora (Python 3.10)

```bash
conda create -n aurora python=3.10 -y
conda activate aurora
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.40,<5.0" "tokenizers<0.21" "accelerate<1.0" einops huggingface_hub rarfile
```

If an existing aurora env has `transformers>=5.0` or `torchvision` from CPU-only PyPI (symptom: `operator torchvision::nms does not exist`):

```bash
pip uninstall -y transformers tokenizers accelerate
pip install "transformers>=4.40,<5.0" "tokenizers<0.21" "accelerate<1.0"
pip uninstall -y torch torchvision
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
```

Do not use `--force-reinstall`; it pulls CPU-only torchvision and breaks the CUDA pairing.

### TaTS (Python 3.11)

```bash
conda create -n tats python=3.11 -y
conda activate tats
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu121
pip install pandas scikit-learn patool tqdm sktime matplotlib reformer_pytorch transformers
```

### MM-TSFlib (Python 3.9)

```bash
conda create -n mmtsflib python=3.9.18 -y
conda activate mmtsflib
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.23.5 pandas==1.5.3 scikit-learn==1.2.2 transformers==4.40.2 \
            tokenizers==0.19.1 einops==0.4.0 patool==1.12 sktime==0.16.1 \
            sentencepiece==0.1.99 reformer-pytorch==1.4.4 huggingface-hub==0.23.0 \
            accelerate==0.33.0 tqdm matplotlib
```

### Aurora weights

```bash
huggingface-cli download AutonLab/Aurora --local-dir weights/aurora
```

Export `AURORA_WEIGHTS=$(pwd)/weights/aurora` before running Aurora sweeps.

### Environment-domain archive

MM-TSFlib ships Environment as `.rar`:

```bash
sudo apt-get install -y unrar
cd repos/MM-TSFlib/data/Environment && unrar x -y NewYork_AQI_Day.rar
```

## Quickstart

### 1. Apply repo patches

```bash
python3 code/apply_repo_patches.py            # apply
python3 code/apply_repo_patches.py --check    # verify markers
python3 code/apply_repo_patches.py --revert   # restore originals
```

13 patches: 2 MM-TSFlib, 5 TaTS, 6 Aurora. Idempotent. Backups stored as `*.probe_backup`.

### 2. Generate perturbation CSVs

```bash
python3 code/generate_perturbations.py --repo all
# Expected: "SUMMARY: 162 / 162 files validated clean"
# SocialGood drops 8 rows with empty OT values during coercion.
```

### 3. Pilot run

```bash
# Aurora
conda activate aurora
export AURORA_WEIGHTS=$(pwd)/weights/aurora
python3 code/run_experiments.py --models aurora \
    --conditions C1_original --domains Economy --pred_lens 8 --seeds 2021

# TaTS
conda activate tats
python3 code/run_experiments.py --models tats --backbones iTransformer \
    --conditions C1_original --domains Economy --pred_lens 8 --seeds 2021

# MM-TSFlib
conda activate mmtsflib
python3 code/run_experiments.py --models mmtsflib --backbones Informer \
    --conditions C1_original --domains Economy --pred_lens 8 --seeds 2021
```

Each should finish with `Success: 1 / 1`.

## Running the full sweep

The orchestrator supports per-cell resume, N-shard parallelism, and custom batch size. Cells are identified by the tuple `(model, backbone, condition, seed, domain, pred_len)` and written to a deterministic path; a completed cell is always skipped on re-launch.

### Aurora (no sharding needed; fast because it's zero-shot)

```bash
conda activate aurora
export AURORA_WEIGHTS=$(pwd)/weights/aurora
nohup python3 -u code/run_experiments.py \
    --models aurora --preset time_mmd \
    --conditions C1_original C2_empty C3_shuffled C4_crossdomain \
                 C5_constant C6_unimodal C7_null C8_oracle \
    --domains Agriculture Climate Economy Energy Environment \
              Health Security SocialGood Traffic \
    --seeds 2021 2022 2023 \
    > nohup_aurora.log 2>&1 &
```

864 cells total. All previously completed h=8 cells auto-skip.

### TaTS — 5 parallel shards

```bash
conda activate tats
for s in 0 1 2 3 4; do
    nohup python3 -u code/run_experiments.py \
        --models tats --preset time_mmd --all_backbones \
        --conditions C1_original C2_empty C3_shuffled C4_crossdomain \
                     C5_constant C6_unimodal C7_null C8_oracle \
        --domains Agriculture Climate Economy Energy Environment \
                  Health Security SocialGood Traffic \
        --seeds 2021 2022 2023 \
        --num_shards 5 --shard_id $s \
        > nohup_tats_shard${s}.log 2>&1 &
    sleep 30
done
```

7776 cells split across 5 shards (~1555 each). Total VRAM ~18 GB on an A10G.

### MM-TSFlib — 4 parallel shards

```bash
conda activate mmtsflib
for s in 0 1 2 3; do
    nohup python3 -u code/run_experiments.py \
        --models mmtsflib --preset time_mmd \
        --backbones Informer Autoformer Transformer Nonstationary_Transformer \
                    DLinear FEDformer Reformer Crossformer iTransformer FiLM \
        --conditions C1_original C2_empty C3_shuffled C4_crossdomain \
                     C5_constant C6_unimodal C7_null C8_oracle \
        --domains Agriculture Climate Economy Energy Environment \
                  Health Security SocialGood Traffic \
        --seeds 2021 2022 2023 \
        --num_shards 4 --shard_id $s \
        > nohup_mmtsflib_shard${s}.log 2>&1 &
    sleep 30
done
```

8640 cells split across 4 shards. Total VRAM ~10-20 GB.

## Sharding

Sharding partitions the global grid by `cell_idx % num_shards`. Properties verified by test:

- Disjoint across shards (no cell runs twice).
- Union is the full grid (no cell is missed).
- `result_path(spec)` depends only on `RunSpec` contents, not on sharding config. Result files from any shard count are mutually compatible for resume.

If a shard crashes, the others keep running. Stale `.running` markers are cleared on orchestrator startup; the failed cells retry automatically on the next launch.

## Resume semantics

On startup the orchestrator:

1. Calls `clear_stale_markers()` to remove `*.running` files from crashed runs.
2. Walks the grid. For each cell:
   - If `results/.../<cell>.json` exists with `"success": true` and no marker → print `SKIP (done)` and continue.
   - Otherwise → run the cell.

Pass `--force` to re-run everything regardless of existing results.

## Disk management

Both TaTS and MM-TSFlib save a checkpoint via their built-in `early_stopping` at `repos/<Model>/checkpoints/<setting>/checkpoint.pth`. Checkpoint size ranges from ~25 MB (small models) to ~170 MB (Crossformer). For the full sweep this would accumulate to >1 TB.

**Default: the runner deletes each cell's checkpoint directory after the subprocess returns.** The training setting string is preserved in `result.extra.training_setting` for every JSON, so specific cells can be reproduced later if needed.

To retain checkpoints for cells you want to probe later:

```bash
python3 code/run_experiments.py ... --preserve_checkpoints
```

To purge stale checkpoints manually:

```bash
rm -rf repos/TaTS/checkpoints/* repos/MM-TSFlib/checkpoints/*
```

## Monitoring

```bash
tail -f nohup_tats_shard0.log
tail -f sweep_log.jsonl

# Completed count
find results -name '*.json' | xargs grep -l '"success": true' 2>/dev/null | wc -l

# In-flight count (should be ≤ num_shards)
find results -name '*.running' | wc -l

# Disk usage on checkpoint dirs
du -sh repos/TaTS/checkpoints repos/MM-TSFlib/checkpoints 2>/dev/null

# GPU
nvidia-smi
```

## Aurora diagnostic probes

Standalone; does not couple to the main runner. Run after the main Aurora sweep completes.

```bash
conda activate aurora
export AURORA_WEIGHTS=$(pwd)/weights/aurora
python3 code/aurora_probes.py \
    --conditions C1_original C2_empty C6_unimodal \
    --domains Agriculture Economy Health \
    --seeds 2021 --pred_lens 8
```

Three probes:

- **A** — gradient-norm at the text-embedding input
- **B** — entropy of `TextGuider` cross-attention weights
- **C** — output divergence with vs without text

Outputs go to `probes/aurora/...` (separate from `results/`). The `--validate_invariance` flag verifies that hook installation does not change model outputs bit-for-bit.

TaTS and MM-TSFlib probes are not yet implemented. The runner records `training_setting` so a future probe script can reproduce the trained model from a specific cell.

## Analyzing results

```bash
python3 code/analyze_results.py
```

Writes per-condition aggregates to `summaries/*.csv` and LaTeX tables to `tables/*.tex`.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `operator torchvision::nms does not exist` | torchvision reinstalled from default PyPI (CPU-only) | Reinstall torch+torchvision from `cu121` index together |
| `KeyError: 'Autoformer'` on TaTS | TaTS's `exp_basic.py` registers only iTransformer | Re-run `apply_repo_patches.py` |
| `ValueError: __len__() should return >= 0` | Val split too small for chosen `pred_len` | Use `--preset time_mmd` (auto-sizes `seq_len` per domain) |
| `PytorchStreamWriter failed writing file` | Disk full from accumulated checkpoints | Default cleanup handles this; also `rm -rf repos/*/checkpoints/*` |
| `AttributeError: all_tied_weights_keys` (Aurora) | `transformers>=5.0` installed | `pip install "transformers>=4.40,<5.0"` |
| Aurora SocialGood fails every condition | 8 empty `OT` values at end of 2024 | Re-run `generate_perturbations.py`; coercion drops those rows |

Full per-cell stdout/stderr is saved to `logs/<model>/<backbone>/<cond>/seed<s>/<domain>_h<h>.log` for any failed run.

## References

- Time-MMD: `arXiv:2406.08627`
- MM-TSFlib: (same authors as Time-MMD)
- TaTS: forthcoming
- Aurora: `arXiv:2410.10819`