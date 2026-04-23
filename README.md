# Text Probing of MM-TSFMs on Time-MMD

Workshop paper for ICML 2026 FMSD. Probes what MM-TSFlib, TaTS, and Aurora actually use from text when forecasting on Time-MMD.

## Pipeline flow (read this first)

```
 ┌─────────────┐   ┌──────────────┐   ┌────────────────┐   ┌──────────────┐   ┌────────────┐
 │ Clone repos │ → │ Apply patches│ → │ Generate 378   │ → │ Run sweep    │ → │ Analyze    │
 │ (×3 models) │   │ (idempotent) │   │ perturbed CSVs │   │ (per-model)  │   │ (bootstrap)│
 └─────────────┘   └──────────────┘   └────────────────┘   └──────────────┘   └────────────┘
                           ↑                   ↑                   ↑                  ↑
                    apply_repo_patches  generate_perturbations  run_experiments  analyze_results
                           .py                 .py                 .py              .py
```

## What each condition tests

| Code | Text content | Numeric text-derived cols (e.g. `prior_history_avg`) |
|---|---|---|
| C1 Original | Real facts from Time-MMD | **Intact** |
| C2 Empty | `""` (literal empty string) | **Zeroed** |
| C3 Shuffled | Within-domain permutation (seed-controlled) | Intact |
| C4 Cross-domain | Paired domain's text, cycled + shuffled | Intact |
| C5 Constant | `"Time series data point."` | **Zeroed** |
| C6 Unimodal | — (no CSV; CLI flag only) | Zeroed via each paper's own flag |
| C7 Null | `"0"` | **Zeroed** |
| C8 Oracle | Injects future OT values into text | Intact |

Paired domains for C4: Ag↔Sec, Cli↔Ene, Eco↔Hea, SG↔Tra; Env self-shuffles.

## Project layout

```
probe_project/
├── repos/                     # cloned model repos (you clone these)
│   ├── MM-TSFlib/
│   ├── TaTS/
│   ├── Aurora/                # has sub-folders; we use Aurora/TimeMMD/
│   └── Time-MMD/              # raw dataset; not called at runtime
├── weights/aurora/            # pretrained Aurora ckpts (~2GB, you download)
├── data/                      # 378 perturbed CSVs (generator creates these)
├── code/                      # our pipeline
├── results/                   # one JSON per experiment cell
├── summaries/ tables/         # paper tables
└── notes/                     # design docs
```

## Repo-verified dependencies

Each repo has different Python version and dependency constraints. Create three separate conda envs.

### 1. TaTS env (from `repos/TaTS/README.md` verbatim)

```bash
conda create -y -n tats python=3.11.11
conda activate tats
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
    --index-url https://download.pytorch.org/whl/cu121
pip install pandas scikit-learn patool tqdm sktime matplotlib \
    reformer_pytorch transformers
conda deactivate
```

### 2. MM-TSFlib env (core deps from `repos/MM-TSFlib/environment.txt`, Python 3.9.18)

**Why not `pip install -r environment.txt`?** That file is a conda-freeze with ~300 packages, most of them system-level (filesystem paths that don't exist on your box). It will fail on any fresh install. We install only the core deps.

**Known issue in the original freeze:** it pins `torch==2.4.0` with `torchvision==0.15.2` — these versions are **incompatible** (torchvision 0.15.x pairs with torch 2.0.x; torch 2.4.0 pairs with torchvision 0.19.0). We use the correct torch 2.4.0 + torchvision 0.19.0 pairing per PyTorch's official compatibility matrix.

```bash
conda create -y -n mmtsflib python=3.9.18
conda activate mmtsflib
pip install torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.23.5 pandas==1.5.3 scikit-learn==1.2.2 \
    transformers==4.40.2 tokenizers==0.19.1 einops==0.4.0 \
    patool==1.12 sktime==0.16.1 sentencepiece==0.1.99 \
    reformer-pytorch==1.4.4 huggingface-hub==0.23.0 \
    accelerate==0.33.0 tqdm matplotlib
python -c "import torch, transformers, sktime; print('MM-TSFlib deps OK')"
conda deactivate
```

### 3. Aurora env

**Note on the repo contradiction you may spot:** `repos/Aurora/TimeMMD/requirements.txt` lists `torch==1.11.0`. The top-level `repos/Aurora/README.md` says `torch==2.4.0 torchvision==0.19.0 transformers[torch]`. The latter is correct — the model code uses PyTorch 2.x features. The sub-folder `requirements.txt` is stale.

**Known transformers compatibility issue:** Aurora was built against transformers 4.x. Transformers 5.0+ introduced a new internal API (`all_tied_weights_keys`) that custom models must implement, which Aurora doesn't. Without a version pin, pip installs the latest transformers (currently 5.x) and model loading crashes with `AttributeError: 'AuroraForPrediction' object has no attribute 'all_tied_weights_keys'`. Pin transformers below 5.0.

```bash
conda create -y -n aurora python=3.10
conda activate aurora
pip install torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.40,<5.0" "tokenizers<0.21" \
    einops huggingface_hub numpy matplotlib pandas scikit-learn rarfile "accelerate<1.0"
conda deactivate
```

**If you already created the aurora env with transformers 5.x** (and hit the AttributeError), fix in place. Do NOT use `--force-reinstall` — it also reinstalls torchvision from default PyPI, which breaks the CUDA torch/torchvision pairing (`RuntimeError: operator torchvision::nms does not exist`). Instead:

```bash
conda activate aurora
pip uninstall -y transformers tokenizers accelerate
pip install "transformers>=4.40,<5.0" "tokenizers<0.21" "accelerate<1.0"
python -c "import transformers; print('transformers:', transformers.__version__)"   # should be 4.x
python -c "import torch, torchvision; print('torch', torch.__version__, '| torchvision', torchvision.__version__)"  # both should be +cu121
```

If `torchvision` already got damaged by a prior `--force-reinstall`, restore the correct CUDA build:

```bash
pip uninstall -y torch torchvision
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
```

### CUDA note

Pinned CUDA above is 12.1 (`cu121`). If `nvidia-smi` says CUDA 11.x, change `cu121` → `cu118`. CPU-only: drop the `--index-url` flag.

## Aurora pretrained weights (~2GB)

```bash
conda activate aurora
mkdir -p weights
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='DecisionIntelligence/Aurora',
                  local_dir='weights/aurora',
                  local_dir_use_symlinks=False,
                  repo_type='model')
"
```

Tell the runner where:

```bash
export AURORA_WEIGHTS=$(pwd)/weights/aurora
```

Persist by adding to `~/.bashrc`.

## Environment-domain unrar (MM-TSFlib only)

MM-TSFlib ships the Environment-domain CSV as `.rar`:

```bash
sudo apt-get install -y unrar
cd repos/MM-TSFlib/data/Environment && unrar x -y NewYork_AQI_Day.rar
```

## Horizon presets — per-domain pred_lens (Time-MMD paper protocol)

Different domains have different natural horizons per the Time-MMD paper (arXiv:2406.08627):

| Frequency | Domains | `seq_len` | `label_len` | `pred_lens` |
|---|---|---|---|---|
| Daily | Environment | 96 | 48 | [48, 96, 192, 336] |
| Weekly | Health, Energy | 36 | 18 | [12, 24, 36, 48] |
| Monthly | Agriculture, Climate, Economy, Security, SocialGood, Traffic | 8 | 4 | [6, 8, 10, 12] |

Total: 36 (domain × horizon) pairs if you run all 9 domains at all 4 paper horizons each.

**Use the `--preset time_mmd` flag to apply all of this automatically.**

```bash
# Runs each domain at ITS paper horizons; sets seq_len/label_len per paper
python3 code/run_experiments.py --preset time_mmd [other args...]
```

Behavior of `--preset time_mmd`:

- **TaTS & MM-TSFlib**: each domain gets the paper's `(seq_len, label_len, pred_lens)` tuple exactly.
- **Aurora**: pred_lens from paper, but `seq_len` stays at Aurora's native per-domain defaults. Rationale: Aurora is zero-shot; its seq_len is an architectural choice about context window, not a Time-MMD benchmark parameter. Starving Aurora with seq_len=8 for monthly data produces degenerate results. If you want Aurora+Time-MMD-seq_len for comparison, pass `--seq_len 8` etc. explicitly (not currently a CLI flag; for that, edit `AURORA_DEFAULTS` in `aurora_runner.py`).

**If you use `--preset time_mmd`, the `--pred_lens` flag is ignored.**

**Backwards compatibility**: your existing 168 Aurora `h=8` results are automatically reused — resume logic matches cells by `result_path` which includes `h<pred_len>`. Running the preset will SKIP cells that are already complete and only run new horizons.

## Pipeline execution

### Step 1: Apply the repo patches (one time)

```bash
python3 code/apply_repo_patches.py           # apply
python3 code/apply_repo_patches.py --check   # verify
python3 code/apply_repo_patches.py --revert  # undo anytime
```

Expect 13 patches (2 MM-TSFlib, 5 TaTS [incl. backbone registration], 6 Aurora). All marked `# PROBE_PATCH_APPLIED` and backed up `*.probe_backup`.

### Step 2: Generate the perturbed CSVs

```bash
python3 code/generate_perturbations.py --repo all
# Expect: "SUMMARY: 162 / 162 files validated clean"
```

**Why 162, not 378?** Only C3 (shuffled) has seed-dependent content after we switched C4 to date-aligned. C1, C2, C4, C5, C7, C8 produce byte-identical CSVs regardless of seed, so we write them once under `seed0/`. The runner's RunSpec.seed still controls the *model's* RNG (weight init, dropout, data order); it just reads the same CSV for non-seeded conditions.

On-disk layout:
- `data/tats/C1_original/seed0/Economy.csv` ← one copy, all runs read it
- `data/tats/C3_shuffled/seed{2021,2022,2023}/Economy.csv` ← three distinct shuffles
- `data/tats/C6_unimodal/` ← does not exist; runner substitutes C1's path

Writes to `data/mmtsflib/{condition}/seed{0 or seed}/{domain_dir}/{file}.csv` and `data/tats/{condition}/seed{0 or seed}/{domain}.csv`. TaTS and Aurora share `data/tats/...`.

### Step 3: Pilot run

```bash
conda activate aurora
export AURORA_WEIGHTS=$(pwd)/weights/aurora
export PROBE_STREAM=1    # stream model stdout to console
python3 code/run_experiments.py \
    --models aurora --conditions C1_original C6_unimodal \
    --domains Economy --pred_lens 8 --seeds 2021
```

Output: two JSONs at `results/aurora/default/{C1_original,C6_unimodal}/seed2021/Economy_h8.json` with non-null `mse`/`mae`. Takes a few minutes.

### Step 4: Full sweep with per-domain horizons (nohup-safe)

Each `--model` must run in its own conda env. Orchestrator is resume-safe: cells with a successful result are skipped; cells with a stale `.running` marker from a crashed run are auto-cleared and retried.

The following commands use `--preset time_mmd` (per-domain horizons + window sizes per the Time-MMD paper) and the paper's backbone set.

**Aurora — all conditions, all domains, all paper horizons, 3 seeds:**

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
echo "Aurora PID: $!"
# Total: 864 cells. Existing h=8 cells auto-skip.
```

**TaTS — 3 representative backbones (iTransformer / DLinear / Autoformer):**

```bash
conda activate tats
nohup python3 -u code/run_experiments.py \
    --models tats --preset time_mmd \
    --backbones iTransformer DLinear Autoformer \
    --conditions C1_original C2_empty C3_shuffled C4_crossdomain \
                 C5_constant C6_unimodal C7_null C8_oracle \
    --domains Agriculture Climate Economy Energy Environment \
              Health Security SocialGood Traffic \
    --seeds 2021 2022 2023 \
    > nohup_tats.log 2>&1 &
echo "TaTS PID: $!"
# Total: 2592 cells.
```

**TaTS — all 9 backbones (paper-complete):**

```bash
conda activate tats
nohup python3 -u code/run_experiments.py \
    --models tats --preset time_mmd --all_backbones \
    --conditions C1_original C2_empty C3_shuffled C4_crossdomain \
                 C5_constant C6_unimodal C7_null C8_oracle \
    --domains Agriculture Climate Economy Energy Environment \
              Health Security SocialGood Traffic \
    --seeds 2021 2022 2023 \
    > nohup_tats_all.log 2>&1 &
# Total: 7776 cells.
```

**MM-TSFlib — paper's 10 backbones:**

```bash
conda activate mmtsflib
nohup python3 -u code/run_experiments.py \
    --models mmtsflib --preset time_mmd \
    --backbones Informer Autoformer Transformer Nonstationary_Transformer \
                DLinear FEDformer Reformer Crossformer iTransformer FiLM \
    --conditions C1_original C2_empty C3_shuffled C4_crossdomain \
                 C5_constant C6_unimodal C7_null C8_oracle \
    --domains Agriculture Climate Economy Energy Environment \
              Health Security SocialGood Traffic \
    --seeds 2021 2022 2023 \
    > nohup_mmtsflib.log 2>&1 &
echo "MM-TSFlib PID: $!"
# Total: 8640 cells.
```

**MM-TSFlib — full 22-backbone sweep (expensive but complete):**

```bash
conda activate mmtsflib
nohup python3 -u code/run_experiments.py \
    --models mmtsflib --preset time_mmd --all_backbones \
    --conditions C1_original C2_empty C3_shuffled C4_crossdomain \
                 C5_constant C6_unimodal C7_null C8_oracle \
    --domains Agriculture Climate Economy Energy Environment \
              Health Security SocialGood Traffic \
    --seeds 2021 2022 2023 \
    > nohup_mmtsflib_all.log 2>&1 &
# Total: 19008 cells. Budget: probably too much for one week on one GPU.
```

**`python3 -u`** is important: disables output buffering, so `tail -f nohup_tats.log` actually shows progress. Our runner also calls `sys.stdout.flush()` after each cell for the same reason.

**Available backbones** (see `runners/tats_runner.py` → `TATS_ALL_BACKBONES` and `runners/mmtsflib_runner.py` → `MMTSFLIB_ALL_BACKBONES`):

- **TaTS** (9, after our patch): iTransformer, Autoformer, DLinear, FEDformer, FiLM, Informer, PatchTST, Transformer, Crossformer
- **MM-TSFlib** (22): Informer, Autoformer, Transformer, Nonstationary_Transformer, DLinear, FEDformer, TimesNet, LightTS, Reformer, ETSformer, PatchTST, Pyraformer, MICN, Crossformer, FiLM, iTransformer, Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN

**Tracking progress during a sweep:**

```bash
tail -f nohup_tats.log                  # live log
tail -f sweep_log.jsonl                 # one line per completed cell
find results -name '*.json' | xargs grep -l '"success": true' 2>/dev/null | wc -l
find results -name '*.running' | wc -l                       # count in-progress
```

**If the instance dies mid-sweep**, just re-launch the same command. The orchestrator's `clear_stale_markers()` cleans leftover `.running` files, and successful results are auto-skipped. Grep for `SKIP (done)` in the log to confirm resume worked:

```bash
grep -c "SKIP (done)" nohup_tats.log
```

**Per-cell full logs** are saved to `logs/<model>/<backbone>/<condition>/seed{s}/<domain>_h<h>.log` so you don't have to re-run failed cells to debug.

### Running two GPU instances in parallel

You have Instance A for Aurora+TaTS and Instance B for MM-TSFlib. Both can read/write the same `results/` via a shared volume if available, or run fully independently on separate copies of the project.

For independent runs, each instance has its own `results/` tree; merge them at analysis time by copying `results/mmtsflib/` from Instance B onto Instance A before running `analyze_results.py`.

### Model checkpoints are saved automatically (for future probe use)

TaTS and MM-TSFlib both save `checkpoint.pth` during training via their built-in `early_stopping` mechanism. These persist at `repos/<Model>/checkpoints/<setting>/checkpoint.pth`. The runner records this path in each RunResult's `extra.checkpoint_path` field so future probe scripts can reload the exact trained model without re-running training.

Aurora is zero-shot — no checkpoint is produced; the pretrained weights at `$AURORA_WEIGHTS` serve the same purpose.

This means you can run TaTS/MM-TSFlib probes *after* the main sweep completes without re-training, by:
1. Reading `result.extra.checkpoint_path` from a RunResult JSON
2. Loading the checkpoint into the same architecture with the same args
3. Attaching hooks and running diagnostics

(A TaTS/MM-TSFlib probe script is not yet built — deferred until after main results. The infrastructure to enable it is in place.)

### Step 5 (optional): Aurora diagnostic probes

Standalone, non-intrusive. Runs after the main sweep — doesn't touch any file in `results/`.

```bash
conda activate aurora
export AURORA_WEIGHTS=$(pwd)/weights/aurora

# Step 5a: VERIFY that hooks don't change model outputs (one-time check)
python3 code/aurora_probes.py --validate_invariance
# Expect: "*** INVARIANCE CONFIRMED: hooks do not change outputs ***"

# Step 5b: Run the three probes on selected cells
python3 code/aurora_probes.py \
    --conditions C1_original C3_shuffled C6_unimodal C8_oracle \
    --domains Economy Health Energy \
    --seeds 2021 --pred_lens 8
# Writes probes/aurora/<cond>/seed<s>/<domain>_h<h>.json
```

The three probes and what they tell you:

**Probe A — gradient norm at text-encoder input embeddings (`gradnorm_mean`).**
Measures whether gradients flow into text content. If `gradnorm(C1) ≈ gradnorm(C3)` with C3's text being random-shuffled, the model is propagating gradients through the text path regardless of content — evidence that the text branch is trained but content doesn't differentiate behavior.

**Probe B — cross-attention entropy (`attn_entropy_rel_mean`).**
Measures how peaked Aurora's cross-attention over text tokens is. Values close to 1.0 = uniform (attending everywhere equally → no text selectivity). Values close to 0 = highly peaked on specific tokens. Compare C1 vs C3: if entropies match, attention isn't content-differentiated.

**Probe C — output divergence between with-text and without-text (`divergence_mean_sq`).**
Model-agnostic sanity check. Small = text changes the prediction little = text is being ignored downstream.

**Reversibility:** probes attach forward/backward hooks in-process, run, and detach. Nothing persistent is written into the repos. If you want to skip probes entirely for a given run, use `--no_gradnorm` / `--no_attention` / `--no_divergence` flags.

**Why Aurora-only?** Aurora has an explicit text encoder + cross-attention — clean hook points. MM-TSFlib's fusion is a weighted sum (no cross-attention); TaTS precomputes text embeddings at data-load (no live text encoder during training). Aurora is also where your main results show the smallest multimodal/unimodal gap — probes here are the most informative for your paper's narrative.

## Analyze

```bash
python3 code/analyze_results.py
```

Writes `summaries/per_cell_mse.csv`, `summaries/main_mse.csv`, `summaries/pairwise_mse.csv`, `tables/main_mse.tex`.

Paired-bootstrap: `mean_diff` is MSE(test) − MSE(C1) averaged across pairs. `ci_lo > 0` → test condition made things worse. `rel_diff` = effect size relative to C1.

## On `features=S` vs `features=M` (why we use S for all three)

This flag controls whether the TS backbone sees only `OT` (univariate, S) or also other numerical columns like `Exports`, `Imports` (multivariate, M).

- **Aurora**: reference script uses `features=S` (univariate Time-MMD setup).
- **TaTS**: force-overrides to `features=S` on line 161 of `run.py` regardless of user input — the paper is univariate.
- **MM-TSFlib**: reference script says `--features M`, but we verified **all 9 domains crash** under `features='M'` with the shipped preprocessed data (`StandardScaler.fit()` tries to ingest string columns like `start_date`, `Month`, `REGION`, etc., which aren't filtered out before the scaler). Either this code path was never tested, or the published numbers used a patched private variant not in their public release.

**Our choice:** `features='S'` for all three. It's what two-of-three do natively, it's what's actually runnable for MM-TSFlib, and it's the standard univariate Time-MMD setup described in the paper's main Table 1.

**Reviewer defence:** state plainly in the paper that MM-TSFlib's public `features=M` code path is non-functional on the shipped preprocessed data; we use `S` for fair cross-model comparison. Reference file:line in MM-TSFlib and include an appendix entry showing the traceback.

## Design decisions log

- **`features=S`** for all three models (see above section).
- **C8 oracle included** — positive control; without it, null C3/C4 results are unfalsifiable.
- **Zero `prior_history_avg` under C2/C5/C7**, not C3/C4/C6/C8.
- **C4 is date-aligned**, not tile-shuffled. For each target row, we take the paired domain's latest fact with date ≤ target row's date. Target rows before the paired domain's start date get `''` (empty fallback). This avoids repetition artifacts.
- **Unified seeds {2021, 2022, 2023}** — only C3 actually uses them.
- **Primary backbones**: Informer for MM-TSFlib, iTransformer for TaTS.

## Known confounds (flag in paper)

- MM-TSFlib's C6 keeps `prior_history_avg` as a TS input channel under `features=S` — this reproduces their reported unimodal baseline exactly.
- C3/C4 leave `prior_history_avg` intact (aligned with original target; it's a closed-LLM forecast from original text) — so C3/C4 probe the text-*encoder* specifically, not total "no text info."
- TaTS precomputes text embeddings at data-load with a frozen LLM; caching across seeds within (condition, domain) would speed TaTS substantially — not implemented.