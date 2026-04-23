# Text Probing of MM-TSFMs on Time-MMD

**Workshop paper for ICML 2026 FMSD.** Probes what MM-TSFlib (late-fusion), TaTS (text-as-covariate), and Aurora (pretrained multimodal TSFM) actually use from text when forecasting on Time-MMD.

## What this pipeline does

For each model × condition × seed × domain × pred_len cell, train/infer, then compute MSE/MAE. Conditions:

| Code | Text content | Numeric text-derived cols (e.g. `prior_history_avg`) |
|---|---|---|
| C1 Original | Real facts from Time-MMD | **Intact** |
| C2 Empty | `""` (literal) | **Zeroed** |
| C3 Shuffled | Within-domain permutation (seed-controlled) | Intact |
| C4 Cross-domain | Paired domain's text, cycled + shuffled | Intact |
| C5 Constant | `"Time series data point."` | **Zeroed** |
| C6 Unimodal | — (CLI flag only; no CSV perturbation) | Zeroed *through* each paper's flag |
| C7 Null | `"0"` | **Zeroed** |
| C8 Oracle | Injects future OT values into the text | Intact |

Paired domains for C4: Ag↔Sec, Cli↔Ene, Eco↔Hea, SG↔Tra; Env self-shuffles.

## Project layout

```
probe_project/
├── repos/                     # MM-TSFlib, TaTS, Aurora, Time-MMD (--depth 1 clones)
├── data/                      # 378 perturbed CSVs (C1-C5, C7, C8 × 3 seeds × 9 domains × 2 layouts)
│   ├── mmtsflib/<cond>/seed<s>/<domain_dir>/<file>.csv
│   └── tats/<cond>/seed<s>/<domain>.csv    # also serves Aurora
├── weights/                   # pretrained checkpoints (user-populated; see setup)
│   └── aurora/                # from huggingface.co/DecisionIntelligence/Aurora
├── code/
│   ├── generate_perturbations.py     # creates the 378 CSVs
│   ├── apply_repo_patches.py         # patches repo loaders for clean C2/C6
│   ├── run_experiments.py            # orchestrator
│   ├── analyze_results.py            # paired bootstrap, tables
│   └── runners/
│       ├── common.py                 # RunSpec, RunResult, metrics parser
│       ├── aurora_runner.py
│       ├── tats_runner.py
│       └── mmtsflib_runner.py
├── results/                   # one JSON per cell (model/cond/seed/domain_h{h}.json)
├── summaries/ tables/         # analysis output
├── notes/                     # design docs from development
└── README.md (this file)
```

## Setup (one time)

### 1. Python environments

Each repo has its own deps. Create separate envs to avoid version conflicts:

```bash
# Aurora (PyTorch 2.4.1, transformers, einops)
python3.10 -m venv envs/aurora && source envs/aurora/bin/activate
pip install -r repos/Aurora/TimeMMD/requirements.txt
pip install torch==2.4.0 torchvision==0.19.0 transformers rarfile
deactivate

# TaTS
python3.10 -m venv envs/tats && source envs/tats/bin/activate
pip install torch transformers numpy pandas scikit-learn
deactivate

# MM-TSFlib
python3.10 -m venv envs/mmtsflib && source envs/mmtsflib/bin/activate
pip install torch transformers numpy pandas scikit-learn sentencepiece
deactivate
```

Tip: if you have enough GPU memory, all three can share one env built from the superset of requirements. Start separate; merge later if convenient.

### 2. Aurora pretrained weights (~2GB)

```bash
mkdir -p weights/aurora
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='DecisionIntelligence/Aurora',
                  local_dir='weights/aurora',
                  local_dir_use_symlinks=False, repo_type='model')
"
# Set env var so runner finds weights:
export AURORA_WEIGHTS=$(pwd)/weights/aurora
```

### 3. Apply repo patches (one time)

```bash
python3 code/apply_repo_patches.py
# Verify:
python3 code/apply_repo_patches.py --check
# All files should show ≥1 marker. Revert anytime with --revert.
```

### 4. Regenerate perturbation CSVs (if not present)

```bash
python3 code/generate_perturbations.py --repo all
# Should report "378 / 378 files validated clean"
```

## Running experiments

### Pilot (fast smoke test — Aurora on Economy)
```bash
source envs/aurora/bin/activate
python3 code/run_experiments.py \
    --models aurora \
    --conditions C1_original C6_unimodal \
    --domains Economy \
    --pred_lens 8 \
    --seeds 2021
# Should finish in minutes. Output goes to results/aurora/...
```

### Full grid (WARNING: multi-GPU-day)
```bash
# In each repo's env, run the corresponding subset:
# Aurora is fastest (zero-shot) — start here
source envs/aurora/bin/activate
python3 code/run_experiments.py --models aurora --pred_lens 6 8 10 12 24 48

# Then TaTS / MM-TSFlib
source envs/tats/bin/activate
python3 code/run_experiments.py --models tats --pred_lens 12 24 36 48

source envs/mmtsflib/bin/activate
python3 code/run_experiments.py --models mmtsflib --pred_lens 12 24 36 48
```

Orchestrator **resumes by default** — if you stop mid-sweep and re-run, successful cells are skipped. Pass `--force` to redo everything.

Set `PROBE_STREAM=1` to see model stdout in real time for a single cell (useful when debugging).

### Failure handling
Each cell saves `results/<model>/<cond>/seed<s>/<domain>_h<h>.json` with `success: true/false` and an `error` field on failure. The sweep never aborts on a single failure unless `--stop_on_error` is set. Re-run a specific failed cell with its exact argv.

## Analyzing results

```bash
python3 code/analyze_results.py
# Writes:
#   summaries/per_cell_mse.csv       - one row per (model,cond,domain,pred_len) with mean±std
#   summaries/main_mse.csv           - wide paper table
#   tables/main_mse.tex              - LaTeX version
#   summaries/pairwise_mse.csv       - bootstrap comparisons of each condition vs C1
```

Paired-bootstrap output interprets as: `mean_diff` is MSE under the test condition minus MSE under C1, averaged across all (domain, pred_len, seed) pairs. If `ci_lo > 0`, the test condition significantly **increased** MSE (worse). `rel_diff` gives this as a fraction of C1's MSE — the effect size.

## Reproducibility gate (before full grid)

1. Run C1 + C6 for one domain × one pred_len on each model.
2. Compare to the paper's reported numbers. If off by >10%, debug before committing full compute.
3. Record the commit hashes of this repo + the three model repos into `results/provenance.json` (TODO — add to runner).

## Design decisions log

See `notes/` for:
- `01_phase0_findings.md` — repo inspection notes with file:line citations
- `02_phase0_step2_perturbations.md` — perturbation generator design + landmines

Key decisions:
- **MM-TSFlib uses `features=S`** (not `M` as in shipped script — `M` crashes on string columns).
- **C8 oracle included** (positive control — without it, null C3/C4 results are unfalsifiable).
- **C2/C5/C7 zero `prior_history_avg`** (these conditions mean "no text-derived signal of any kind"); C3/C4 leave it intact (they probe the text encoder specifically).
- **Unified seeds {2021, 2022, 2023}** across all models (Aurora's reference default).
- **One primary backbone per non-Aurora model**: Informer for MM-TSFlib, iTransformer for TaTS. Full backbone sweep goes to appendix.

## Known confounds (flag in paper)

- MM-TSFlib's C6 (`prompt_weight=0`) still uses `prior_history_avg` as a TS input channel under `features=S`. To get a truly text-free baseline, we also zero that column for C2/C5/C7 but NOT for C6 (C6 reproduces their paper's own unimodal exactly).
- Under C3/C4, `prior_history_avg` remains aligned with the original target (closed-LLM forecast generated from the original text). These conditions probe the text-encoder path, not total "no text info."
- TaTS's text embeddings are precomputed at data-load time with a frozen GPT-2. Caching across seeds within (condition, domain) would speed up TaTS substantially — not implemented yet.
