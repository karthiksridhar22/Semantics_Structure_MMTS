# Phase 0 Step 2 — Perturbation Generator & Loader Patches

**Session deliverables:**
1. `code/generate_perturbations.py` — writes all C1/C2/C3/C4/C5/C7/C8 CSVs
2. `code/apply_repo_patches.py` — idempotent patches to the three repo loaders
3. `data/mmtsflib/` and `data/tats/` — 378 perturbed CSVs, all validated

## 1. Design decisions locked in

| Decision | Value | Rationale |
|---|---|---|
| C8 Oracle positive control | **INCLUDED** | Without it, null result on C3/C4 is unfalsifiable. |
| Zero numeric text-derived cols under C2/C5/C7 | **YES** | These conditions mean "no text signal at all." |
| Leave numeric priors INTACT under C3/C4/C8 | YES | These probe the text-encoder path specifically. |
| MM-TSFlib `features` flag | **`S` (not `M`)** | `M` is broken in shipped code — `StandardScaler.fit()` fails on string columns. Aurora also uses `S`. Standard univariate Time-MMD setup. |
| MM-TSFlib text column | `Final_Search_4` (primary) | Matches `week_health.sh` reference script. Appendix: `Final_Search_2`, `Final_Search_6`. |
| Seeds | MM-TSFlib {2021,2022,2023}; TaTS {2024,2025,2026}; Aurora {2021,2022,2023} | Centered on each repo's reference-script default. |

## 2. What C1–C8 actually look like in the data

Spot-check from `mmtsflib/Economy/seed2021/row 50`:

| Condition | `prior_history_avg[50]` | `Final_Search_4[50]` (truncated) |
|---|---|---|
| C1 Original | -66421.70 | `'Available facts are as follows: 2020-11-23: The United ...'` |
| C2 Empty | 0.00 | `''` (literal empty string with patched loader) |
| C3 Shuffled | -66421.70 | `'Available facts are as follows: 2004-05-24: The United ...'` |
| C4 Cross-domain | -66421.70 | `'Available facts are as follows: 2002-04-01: Objective f...'` (Health's text) |
| C5 Constant | 0.00 | `'Time series data point.'` |
| C7 Null | 0.00 | `'0'` |
| C8 Oracle | -66421.70 | `'Available facts are as follows: Step+1: The OT will be ...'` |

Sanity checks:
- **C3 diffs from C1 on 99.8%** of rows (expected: 1/N fixed points under random permutation)
- **C4 diffs from C1 on 100%** of rows (cross-domain, no possible matches)
- **C1, C3, C4, C8 leave numeric priors intact** (-66421.70 preserved)
- **C2, C5, C7 zero the numeric priors** (0.00 in all three)

## 3. The CSV-round-trip landmine (and how we avoided it)

Default `pd.read_csv()` converts empty CSV fields to `NaN`. That would corrupt C2 in two ways:
1. **MM-TSFlib**: its fstring formatter `f"...{text_info}..."` coerces NaN → literal string `"nan"`. C2 would become "model sees word `nan`," not "model sees empty text."
2. **TaTS/Aurora**: their loaders explicitly replace `NaN` → `'No information available'`. C2 would silently collapse into a constant string (effectively C5 with a different constant).

**Our two-part fix:**

**A. Write side** (in `generate_perturbations.py`): use `pd.to_csv(..., quoting=csv.QUOTE_NONNUMERIC)`. Empty strings are emitted as literal `""` (quoted empty), distinguishing them from missing fields.

**B. Read side** (in `apply_repo_patches.py`): patch each repo's `pd.read_csv()` call to add `keep_default_na=False`. With this, quoted empty `""` in the CSV reads back as Python `''` string instead of `NaN`. We also disabled the `pd.isnull(text) → 'No information available'` replacement in TaTS/Aurora so our literal `''` isn't silently overwritten.

**Verified:** after patches, TaTS loader reads our C2 CSVs as empty strings, `pd.isnull()` returns False, no silent replacement fires.

## 4. Summary of patches applied to each repo

| Repo | File | Change | Purpose |
|---|---|---|---|
| MM-TSFlib | `data_provider/data_loader.py` | Add `keep_default_na=False` to `pd.read_csv` | C2 `''` round-trip |
| TaTS | `data_provider/data_loader.py` | Add `keep_default_na=False`; replace `'No information available'` fallback with `''` | C2 `''` round-trip; no silent overwrite |
| Aurora | `TimeMMD/data_provider/data_loader.py` | Same two as TaTS | Same |
| Aurora | `TimeMMD/run_longExp.py` | Add `--no_text` CLI flag | C6 unimodal trigger |
| Aurora | `TimeMMD/exp/exp_main.py` | Wire `--no_text` into all 4 `self.model*(...)` calls to pass `text_input_ids=None` | C6 unimodal operationalization |

All patches are:
- Marked with `# PROBE_PATCH_APPLIED` so they're grep-able.
- Backed up as `*.probe_backup` so we can revert.
- Idempotent — safe to re-run.
- Revertible — `python3 code/apply_repo_patches.py --revert`.

## 5. Output file layout (378 CSVs)

```
data/
├── manifest.json                  # every file logged with row count, cols changed
├── mmtsflib/
│   ├── C1_original/
│   │   ├── seed2021/
│   │   │   ├── Algriculture/US_RetailBroilerComposite_Month.csv
│   │   │   ├── Climate/US_precipitation_month.csv
│   │   │   ├── Economy/US_TradeBalance_Month.csv
│   │   │   ├── Energy/US_GasolinePrice_Week.csv
│   │   │   ├── Environment/NewYork_AQI_Day.csv    (59MB, from .rar extraction)
│   │   │   ├── Public_Health/US_FLURATIO_Week.csv
│   │   │   ├── Security/US_FEMAGrant_Month.csv
│   │   │   ├── SocialGood/Unadj_UnemploymentRate_ALL_processed.csv
│   │   │   └── Traffic/US_VMT_Month.csv
│   │   ├── seed2022/...  (same structure)
│   │   └── seed2023/...
│   ├── C2_empty/... (same structure × 3 seeds × 9 domains)
│   ├── ... (C3 through C8)
└── tats/                           # also serves Aurora
    ├── C1_original/
    │   ├── seed2024/
    │   │   ├── Agriculture.csv
    │   │   ├── Climate.csv
    │   │   ├── Economy.csv
    │   │   ├── Energy.csv
    │   │   ├── Environment.csv
    │   │   ├── Health.csv
    │   │   ├── Security.csv
    │   │   ├── SocialGood.csv
    │   │   └── Traffic.csv
    │   ├── seed2025/...
    │   └── seed2026/...
    ├── C2_empty/...
    ├── ... (C3 through C8)
```

Total: 7 conditions × 3 seeds × 9 domains × 2 repo layouts = **378 CSVs**, all validated against the originals on every non-text, non-null-numeric column (byte-exact).

## 6. What's next (Phase 1: baseline reproduction)

Before running the full 378 × 3-model grid, we must verify the pipeline reproduces published numbers on C1 and C6. If it doesn't, nothing else matters.

**Step-by-step:**
1. **Install dependencies for each repo.** Each has its own `environment.txt`/`requirements.txt`. Start with Aurora (zero-shot, cheapest — quickest smoke test).
2. **Download Aurora pretrained weights** from `huggingface.co/DecisionIntelligence/Aurora` (~2GB).
3. **Pilot run 1: Aurora on Economy, pred_len 12, C1 (original) and C6 (--no_text).**
   Compare numbers to Aurora paper Table ? (need to look up).
4. **Pilot run 2: TaTS on Economy, pred_len 48, C1 and C6.**
   Compare to TaTS paper main table (backbone-only row = C6).
5. **Pilot run 3: MM-TSFlib on Health, pred_len 48, C1 and C6.**
   Compare to MM-TSFlib appendix unimodal baseline.
6. **Gate:** if any (model, domain) C1 is off by > 10% from paper, pause and debug before full grid.

**Compute budget estimate:**
- Aurora: 378 inferences, zero-shot, ~few minutes each on a GPU. Cheap.
- TaTS: 378 training runs × ~5-15 min per (epochs=5). Moderately expensive.
- MM-TSFlib: Similar to TaTS. Expensive.
- Total wall-clock: probably 3–7 GPU-days depending on hardware. Budget accordingly.

## 7. What we still haven't solved

1. **Find published reference numbers for C1 and C6** for each of the three models. Need these to pass the Phase 1 gate. Requires re-reading each paper's tables.
2. **Write runners.** One wrapper per model that reads from `data/{repo}/{cond}/seed{s}/{domain}.csv`, invokes the model with correct flags, saves MSE/MAE/sMAPE into a standard JSON.
3. **Aurora's `model_path` is hardcoded** to `/home/Aurora/aurora` in the reference script. We need to either symlink or pass it as CLI. Check.
4. **Repo environments.** Each repo has its own Python deps. Check if they conflict; we may need separate venvs.
