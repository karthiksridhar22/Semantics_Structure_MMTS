# Phase 0 Findings — Repo Inspection

**Purpose.** Ground-truth reference for the three models' data formats, text handling, unimodal paths, and default hyperparameters. Every claim below is verified against the actual cloned code (not memory, not papers). Citations are to file:line in the repos at `/home/claude/probe_project/repos/`.

---

## 1. Data formats: three different preprocessed CSV schemas

There is **no single "Time-MMD benchmark CSV"**. Each model's repo ships its own preprocessed version:

| Model | Data location | File naming | Key text column(s) |
|---|---|---|---|
| MM-TSFlib | `MM-TSFlib/data/{Domain}/` | `US_<target>_<freq>.csv` (varies) | `Final_Search_{2,4,6}`, `Final_Output` |
| TaTS | `TaTS/data/` | `{Domain}.csv` | `fact`, `preds` |
| Aurora | **not shipped**; expects `{Domain}.csv` at `root_path` | `{Domain}.csv` | `fact` |

**Consequence for experiment design:** C1–C7 must be generated three times, once per repo's data layout. The text columns to perturb differ.

### 1a. MM-TSFlib schema details

Source: `MM-TSFlib/data_provider/data_loader.py:60-114`.

Per-domain single CSV with ~11–26 columns depending on domain. Always contains:
- `date`, `start_date`, `end_date` (alignment anchors; `date` is used as the time index)
- `OT` (target)
- Domain-specific numerical features (e.g. Economy has `Exports`, `Imports`; Health has `AGE 0-4`…`AGE 65`, `ILITOTAL`, etc.)
- `prior_history_avg` (always), `prior_history_std` (most domains)
- `his_avg_{1..7}`, `his_std_{1..7}` (Economy, Agriculture only — GPT-3.5-derived numeric forecasts at horizons 1–7)
- `Final_Search_{2,4,6}` (always): concatenated past-fact text at lookback widths 2/4/6
- `Final_Output` (Energy, Health, Traffic only): GPT-3.5-generated forecast text

**Text column actually consumed at training time** (line 63–67): exactly one, controlled by `--text_len` ∈ {2,4,6} (default `Final_Search_3` but reference script uses `Final_Search_4`). When `--use_closedllm=1` it reads `Final_Output` instead and forces BERT encoder.

**"Prior" channel is numeric and is applied in fusion**:
- `prior_history_avg` is one-hot pulled via `get_prior_y()` (line 115–124) and added inside the text-branch fusion: `prompt_y = norm(text_emb) + prior_y`. Then `output = (1-prompt_weight)·TS + prompt_weight·prompt_y`.
- When `prompt_weight=0`, the `prior_y` contribution also zeros out. So the unimodal path is clean.
- **Caveat**: `prior_history_avg` ALSO appears in `df_data` (line 68, 79) as a regular numerical feature alongside the target's numerical features. So even with `prompt_weight=0`, the TS backbone sees closed-LLM-derived numeric signal as an input channel. This is a subtle confound to flag in the paper.

**Environment domain ships as `.rar`** (`data/Environment/NewYork_AQI_Day.rar`) — not directly readable. Needs `unrar` extraction.

### 1b. TaTS schema details

Source: `TaTS/data_provider/data_loader.py:60-146`.

One CSV per domain, naming is correct (`Agriculture.csv`, not the MM-TSFlib typo "Algriculture"; "Health" not "Public_Health").

Columns (Economy example): `Month, Exports, Imports, OT, start_date, date, end_date, prior_history_avg, fact, preds`.

Columns (Health example): adds `REGION TYPE, REGION, YEAR, WEEK, % WEIGHTED ILI, AGE *, ILITOTAL, NUM. OF PROVIDERS, TOTAL PATIENTS, YEAR_WEEK, prior_history_avg, prior_history_std, fact, preds`.

**Text column consumed** (line 67): always `fact`. `Final_Search_*`, `Final_Output` don't exist here.

**NaN handling** (line 143–145): if `fact[i]` is `NaN`, replaced with `'No information available'`. This matters for C2 — a true empty string `""` is NOT the same as `NaN`, which is NOT the same as `'No information available'`. All three produce different tokenizations. We should standardize: use `""` and accept that NaN rows → `""`.

**Tokenization happens ONCE at data-loading time** (line 149–161): `self.input_ids`, `self.attn_mask`, `self.text_embeddings` are precomputed from a frozen LLM. Unlike MM-TSFlib which tokenizes per batch.

### 1c. Aurora schema details (TimeMMD sub-benchmark)

Source: `Aurora/TimeMMD/data_provider/data_loader.py:62-124`.

**Aurora does NOT ship Time-MMD data.** The script (`run_aurora_timemmd_zero_shot.sh:55`) hardcodes `--root_path "/home/Aurora/TimeMMD/dataset"`. We must supply our own CSVs there.

**Text column consumed** (line 63): `fact` — exactly the same as TaTS.

**Implication:** We can reuse TaTS's preprocessed CSVs for Aurora (schema-compatible — both use `fact`; Aurora's loader is a near-copy of TaTS's). This cuts our perturbation-generation work from three branches to two:
- Branch A: MM-TSFlib format (perturb `Final_Search_4` and `Final_Output`)
- Branch B: TaTS/Aurora format (perturb `fact`)

---

## 2. Reference hyperparameters (verified from shipped scripts)

### MM-TSFlib — `scripts/week_health.sh`
- `seed=2021` (code default is 2024; script overrides)
- `text_len=4` → uses `Final_Search_4`
- `prompt_weight=0.1`
- `seq_len=24`, `label_len=12`, `pred_len ∈ {12,24,36,48}`
- `llm_model=BERT` (frozen, `use_fullmodel=0`)
- `features=M` (multivariate)
- Models shown in script: Informer, Reformer. Paper benchmarks many more; each picks its own `prompt_weight`.

### TaTS — `scripts/main_forecast.sh`
- `seed=2025` (code default 2024; script overrides)
- `text_emb=12` (integer — text is projected to 12 additional covariate channels concatenated with the TS)
- `prior_weight=0.5` (→ `prompt_weight=prior_weight` via line 167 of `run.py`)
- `seq_len=24`, `label_len=12`, `pred_len=48`
- `llm_model=GPT2` (frozen, precomputed at data-load time)
- `train_epochs=5`, `patience=5`
- Backbone in reference script: `iTransformer`

### Aurora — `scripts/run_aurora_timemmd_zero_shot.sh`
- `random_seed=2021` (code default 2021)
- **Zero-shot** (`is_training=0`): no training, just inference with pretrained weights
- `features=S` (univariate)
- Per-domain `seq_len` varies: Agriculture/Climate/Economy/SocialGood=192, Health/Traffic=96, Security=220, Environment=528, Energy=1056
- Per-domain horizons vary: Energy/Health={12,24,36,48}, Environment={48,96,192,336}, others={6,8,10,12}
- `inference_token_len=48` (except Security: 24)
- Pretrained weights: `huggingface.co/DecisionIntelligence/Aurora` — must download separately

**Seeds decision for our experiment (3 seeds each):**
- MM-TSFlib: `{2021, 2022, 2023}` — centered on their reference
- TaTS: `{2024, 2025, 2026}` — centered on their reference (code default 2024, script 2025)
- Aurora: `{2021, 2022, 2023}` — zero-shot, seeds only matter for any stochastic sampling inside `model.generate()`

---

## 3. Unimodal (C6) operationalization — verified paths

**Design decision already agreed:** use each paper's reported unimodal baseline, not a uniform zero-out. Concrete operationalization per model:

### MM-TSFlib C6
- **Command:** same as multimodal run but with `--prompt_weight 0`.
- **Why this is legitimate:** Fusion is `output = (1-prompt_weight)·TS + prompt_weight·prompt_y` (`exp/exp_long_term_forecasting.py:461,578,711`). With `prompt_weight=0`, text branch is zeroed end-to-end (including `prior_y`).
- **Caveat:** `prior_history_avg` still enters as a numeric input feature to the TS backbone. To match a "truly text-free" baseline, we'd need to zero that column too. **Flag this in the paper as a design choice.**
- **Compute cost:** Same as a multimodal run — text is still forward-passed, just zeroed out in fusion. No speedup.

### TaTS C6
- **Command:** same as multimodal run but with `--text_emb 0` AND `--prior_weight 0`.
- **Why `text_emb=0`:** `run.py:162-163` sets `enc_in = dec_in = 1 + text_emb`. With `text_emb=0`, the backbone sees a pure univariate TS of shape `(B, seq_len, 1)` — exactly the backbone's unimodal configuration.
- **Why also `prior_weight=0`:** otherwise the `prior_y` (closed-LLM numeric) is still being blended in.
- **Paper reproducibility check:** TaTS's main-table "backbone only" numbers correspond to iTransformer/Autoformer/etc. with `text_emb=0, prior_weight=0`. This should reproduce.

### Aurora C6
- **Path:** pass `text_input_ids=None, text_attention_mask=None, text_token_type_ids=None` to the model.
- **Why this is legitimate:** `modeling_aurora.py:395-406` has `if text_input_ids is not None: ... else: text_features=None, attn_text=None` → `guided_bias=None` in the encoder. This is the model's built-in unimodal code path.
- **Implementation:** Smallest patch — modify `exp/exp_main.py:78-82,144` to pass `None` for the three text args under a new CLI flag `--no_text`. Alternatively: pre-generate a C6 CSV where the `fact` column is literally a sentinel string and handle it at data-loader level. Patch is cleaner.

---

## 4. Operationalizing C1–C7 on the actual column layouts

| Condition | MM-TSFlib ops | TaTS / Aurora ops |
|---|---|---|
| C1 Original | Use shipped CSV as-is | Use shipped CSV as-is |
| C2 Empty | `Final_Search_4 = ""`; (optionally `Final_Output=""`, `prior_history_avg=0`, `his_avg/std=0` for "truly text-free") | `fact = ""` |
| C3 Shuffled (within-domain, seed s) | `np.random.seed(s)` → permute `Final_Search_4` column | Permute `fact` column |
| C4 Cross-domain | Paste paired domain's `Final_Search_4` column, cycled to length | Paste paired domain's `fact` column, cycled |
| C5 Constant | `Final_Search_4 = "Time series data point."` | `fact = "Time series data point."` |
| C6 Unimodal | **No CSV perturbation** — set `--prompt_weight 0` at CLI | **No CSV perturbation** — for TaTS: `--text_emb 0 --prior_weight 0`; for Aurora: patch `exp_main.py` for `--no_text` |
| C7 Null | `Final_Search_4 = "0"` | `fact = "0"` |

**Important subtlety for C3/C4:** Shuffling within domain uses the train/val/test split boundary? No — we shuffle BEFORE the loader applies the 70/15/15 split. The loader (line 69–75 of both) slices by row index. So perturbation is at the row level and the split still uses the same rows, just with different text attached. Good — this keeps splits fixed across conditions.

**Paired-domain mapping for C4** (your spec):
- Agriculture ↔ Security
- Climate ↔ Energy
- Economy ↔ Health
- SocialGood ↔ Traffic
- Environment → self-shuffled (no pair)

When domains have different row counts, cycle (tile) the source column to the target's length.

**Decision needed: should C2/C5/C7 also zero the `prior_history_avg` / `his_avg*` columns?**
- **Yes** if the hypothesis is "remove ALL text-derived signal."
- **No** if the hypothesis is "remove natural-language text, keep LLM-derived numerics as a confound to be discussed."
- **Recommendation:** generate both variants (C2a/C2b, etc.) at least for one pilot domain; pick based on pilot results before the full grid. Cheap to generate, costs compute only if we run both.

---

## 5. Outstanding open question you haven't answered yet

**C8 Oracle-Text positive control.** You didn't answer this last round. My argument: without a condition where the text provably contains info useful for forecasting, a null result in C3/C4 is unfalsifiable (could be "model doesn't use text" OR "text was never useful"). Simplest C8: for each timestamp, set the text to a template like `"The observed value at the next {pred_len} steps will be approximately {next_ot_value}."`, using ground-truth values from the CSV. A multimodal model that actually reads text should do much better on C8 than on C1. If it doesn't → your probing methodology is broken, not the model.

**Still need your decision: include C8 or not?** This affects the perturbation generator code.

---

## 6. Other things worth flagging before coding

1. **Environment domain is a .rar.** Pilots should skip it or we add unrar to the environment.
2. **MM-TSFlib uses column `Final_Search_4` by default; we should run text_len=2 and text_len=6 as robustness-appendix.** Otherwise a reviewer asks "does your finding depend on this knob?" If budget is tight, defer to appendix.
3. **Aurora weights are ~2GB from HF.** Download once, cache.
4. **TaTS precomputes text embeddings at load time using a loaded LLM.** This means data loading is heavy (GPU time). For each of our 7 conditions × 9 domains × 3 seeds, TaTS will re-precompute embeddings. We can't easily cache because the text differs per condition — but within a (condition, domain), we can cache the precomputed embeddings across seeds (same text, just different weight init). Worth implementing if compute is tight.
5. **MM-TSFlib trains per-(model, domain, pred_len) — total runs multiply fast.** Their reference uses {Informer, Reformer}; paper has many models. We should pick ONE backbone per model-family to keep compute tractable: **Informer or iTransformer for MM-TSFlib, iTransformer for TaTS (matches their ref script), Aurora is zero-shot.** Running all backbones is appendix territory.
6. **Metrics.** Paper reports MSE, MAE. Also cheap to add: sMAPE (scale-invariant, useful for cross-domain aggregation), WQL (for Aurora's probabilistic output). Adding now is free.

---

## 7. File-system layout we've set up

```
/home/claude/probe_project/
├── repos/                        # cloned, ~100MB total
│   ├── MM-TSFlib/
│   ├── TaTS/
│   ├── Aurora/
│   └── Time-MMD/                 # raw source; we don't use at runtime
├── notes/
│   └── 01_phase0_findings.md    # THIS FILE
├── data/                         # will hold perturbation outputs
└── code/                         # will hold perturbation generator, run wrappers
```

---

## 8. Next steps (Phase 0 step 2, next conversation)

1. User decides: **C8 yes/no**, **`prior_history_avg`/`his_avg*` zeroing under C2/C5/C7**.
2. Write `code/generate_perturbations.py` that:
   - Reads MM-TSFlib domain CSVs → emits C1–C7 (C8) × 3 seeds into `data/mmtsflib/{cond}_seed{s}/{Domain}.csv`
   - Reads TaTS domain CSVs → emits C1–C7 (C8) × 3 seeds into `data/tats/{cond}_seed{s}/{Domain}.csv`
   - (Same files serve Aurora, since Aurora uses TaTS schema.)
   - Each output is a fresh CSV preserving all non-text columns identically (this is critical for fair comparison).
3. Write a schema validator that confirms perturbed CSVs match original row counts, date ranges, and non-text column values exactly.
4. Run C1 + C6 for one domain (pilot) on each model; verify numbers match published.

