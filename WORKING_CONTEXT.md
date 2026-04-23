# WORKING_CONTEXT.md — Text Probing Experiment

Last updated: 2026-04-22, Phase 0 step 1 complete.

## One-paragraph description
Workshop paper for ICML 2026 FMSD. Probe what MM-TSFlib (late fusion), TaTS (text-as-covariate), and Aurora (pretrained multimodal TSFM) actually use from text when forecasting on Time-MMD. Method: apply a ladder of text perturbations (C1–C7, possibly C8) to each model's preprocessed data and measure MSE/MAE degradation. Null hypothesis-style: if shuffled/cross-domain/empty doesn't change performance, the model isn't using text semantics.

## Current status
- ✅ Phase 0 step 1: Repos cloned, data formats & reference configs inspected.
- 🟨 Waiting on user decisions: (1) include C8 oracle positive control? (2) zero out `prior_history_avg`/`his_avg*` under text-null conditions?
- ⏭ Next: write perturbation generator (Phase 0 step 2).

## Critical facts (from repo inspection, not memory)
1. **Three different preprocessed CSVs** — MM-TSFlib, TaTS, and Aurora each ship their own layout. Aurora's is ABSENT (not shipped); its loader is schema-compatible with TaTS, so we reuse TaTS files for Aurora.
2. **Text columns actually read**:
   - MM-TSFlib: `Final_Search_{2,4,6}` (reference uses `Final_Search_4`) or `Final_Output` when `--use_closedllm=1`
   - TaTS: `fact`
   - Aurora: `fact`
3. **Closed-LLM numerics** (`prior_history_avg`, `his_avg_1..7`) also enter as features — these are a confound for "remove all text signal" conditions.
4. **C6 unimodal operationalization verified**:
   - MM-TSFlib: `--prompt_weight 0`
   - TaTS: `--text_emb 0 --prior_weight 0`
   - Aurora: patch `exp/exp_main.py` to pass `text_input_ids=None` (model's own conditional path)
5. **Reference seeds**: MM-TSFlib ref script uses 2021; TaTS ref uses 2025; Aurora ref uses 2021. Code defaults: MM-TSFlib 2024, TaTS 2024, Aurora 2021.
6. **Paired-domain mapping for C4** (user-specified): Ag↔Sec, Cli↔Ene, Eco↔Hea, SG↔Tra, Env→self-shuffle.
7. **Environment domain is `.rar`** in MM-TSFlib — needs unrar.

## Compute notes
- MM-TSFlib: trains per (model, domain, condition, pred_len, seed). Default scripts use Informer/Reformer. Pick ONE backbone to keep grid tractable; full-sweep goes to appendix.
- TaTS: trains per (backbone, domain, condition, pred_len, seed). Uses iTransformer as ref. Text embeddings precomputed at data load — can cache across seeds within a (condition, domain) pair to speed up.
- Aurora: zero-shot, just inference. Cheapest. Pretrained weights ~2GB from `huggingface.co/DecisionIntelligence/Aurora`.

## Key decisions made
| Decision | Value | Rationale |
|---|---|---|
| C6 operationalization | Paper-reported unimodal baselines | Reproducibility with literature |
| Seeds per cell | 3 | Standard minimum for variance bars |
| Domains | All 9 (Env is .rar — resolve) | Appendix has all; main picks subset |

## Key decisions pending
| Decision | Options | Blocking? |
|---|---|---|
| Include C8 oracle positive control? | Yes / No | Blocks perturbation code |
| Zero numeric text-derived columns under C2/C5/C7? | Yes (clean) / No (scoped) / Both | Blocks perturbation code |
| Which backbones to run under MM-TSFlib and TaTS? | 1 each / 2-3 / all | Blocks run scripts |

## File layout
- `repos/` — MM-TSFlib, TaTS, Aurora, Time-MMD (all `--depth 1` clones)
- `notes/01_phase0_findings.md` — full findings with file:line citations
- `data/` — perturbation outputs (not yet populated)
- `code/` — perturbation generator, run wrappers (not yet populated)

## If handing off to Windsurf / another tool
1. Read `notes/01_phase0_findings.md` first — all design decisions live there.
2. The perturbation generator must emit two parallel CSV sets: one in MM-TSFlib's layout (with `Final_Search_4` and other aux cols intact), one in TaTS's layout (with `fact`). Aurora uses the TaTS set.
3. For all perturbations: preserve the `date`/`start_date`/`end_date` columns, `OT`, and all non-text numerical features byte-exact. Only the designated text column(s) change.
4. Verify row counts match original per-domain after perturbation.
5. For each run, log: commit hash of this repo, model repo's commit hash, full args (json), seed, GPU, wall-clock, peak memory.
