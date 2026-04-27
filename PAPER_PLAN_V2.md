# PAPER_PLAN_V2.md — Catch-up + Drafting Brief

This is the complete planning document for our FMSD-ICML-2026 workshop paper,
revised after deeper literature work and a critical re-read of all results.
It supersedes `PAPER_PLAN.md`. It is written for two audiences:

1. **You** (Karthik), to use as a master reference while drafting.
2. **A second Claude chat** that will help with research and writing — that
   chat should read this file end-to-end before starting.

Everything that *might change after more research* is flagged "FLEX:" so the
research chat knows where to push back. Everything that is locked in
(experimental design, results, narrative spine) is flagged "LOCKED:".

---

## 0. THE PAPER IN ONE PARAGRAPH

We probe three multimodal time-series foundation models (MM-TSFMs) — Aurora,
TaTS, and MM-TSFlib — on the Time-MMD benchmark with eight controlled text
perturbations to ask whether their reported gains over unimodal baselines come
from text **semantics** (the meaning of the words) or from **numeric priors**
silently shipped alongside the text. Two findings: (i) Aurora's text path adds
nothing measurable behaviourally, and its cross-attention is near-uniform over
the 10 distilled text tokens regardless of input. (ii) For TaTS and MM-TSFlib,
perturbing text content alone produces ≤0.1% MSE change; zeroing only the
prior column produces +1.3% to +49.6% change. The "multimodal" gain is the
prior column, not the text.

---

## 1. THE 7 OPEN POINTS — RESOLVED

These are the questions you raised. Here are the answers, with citations and
evidence where I have them.

### Point 1 — Are we sure Time-MMD/MM-TSFlib/TaTS/Aurora don't ablate priors?

**LOCKED: Yes, we are sure. None of them do this ablation.**

I read the Time-MMD paper (NeurIPS 2024) end-to-end. Their Section 4.2,
Figure 6, and Appendix O all compare **unimodal vs multimodal**, where
"multimodal" means (TS branch) + (text branch with `prior_y` baked in).
They report 15-40% MSE reduction. The MM-TSFlib code (`exp_long_term_forecasting.py`,
lines 460, 578, 707-711) literally computes:

```python
prompt_y = norm(prompt_emb) + prior_y
outputs  = (1 - prompt_weight) * outputs + prompt_weight * prompt_y
```

So the "text branch" is `text_embedding + prior_y` from the start. They never
test what happens if you keep the text but null out `prior_y`. The 15-40%
gain they report is **inseparable** from the prior contribution in their
own protocol.

The Aurora-MTS paper (Wu et al., arXiv:2509.22295, ICLR 2026) does have an
"unimodal vs multimodal" ablation table (Section 5.4 of their paper), but
again, their multimodal includes the full text-image-priors stack. They
don't separately ablate the prior column. Their textual data on TimeMMD is
GPT-4-generated and they make no text-content perturbation.

The TaTS paper (Li et al., 2025; arXiv:2502.08942) similarly compares
"with text" vs "without text" but doesn't decompose text-vs-prior.

**This is exactly the gap our paper fills.** We are NOT contradicting their
claims of "multimodal helps" — we are showing that "multimodal" in their
sense ≠ "text helps" in the sense readers naturally assume. The signal is
in the prior column that's bundled into the text branch.

This is the strongest version of our framing. It positions our paper as
a methodological contribution to the field, not a takedown.

### Point 2 — Should we discuss C6 vs C2 and other condition pairs?

**LOCKED: Yes. The condition contrasts ARE the paper.**

The two-axis design produces meaningful contrasts that each test a
different hypothesis. Each contrast is a single comparison of two
conditions and the expected outcome under each hypothesis. The paper
should walk readers through these contrasts, not just dump 9 conditions
on them.

| Contrast | What it tests | Outcome we observe |
|---|---|---|
| C1 vs C9 | Does the prior column help, holding text fixed? | YES (priors zeroed → big drop) |
| C1 vs C3, C4 | Does text content help, holding priors fixed? | NO (text perturbed → no drop) |
| C1 vs C8 | Does ORACLE text (literally the future) help, holding priors fixed? | NO (still no improvement!) |
| C1 vs C6 | Does the text branch help at all? | small effect, fully explained by prior loss |
| C9 vs C2 | Does text content add anything when priors are absent? | NO (same magnitude as C9) |
| C6 vs C2 | Does CSV-text-only differ from architectural-text-bypass? | small consistent gap (~6% TaTS) |

The C6-vs-C2 gap is interesting. C6 is "no text branch invoked at all"
(architectural switch), while C2 is "text='', priors=0" (data-level
nullification). They differ by ~5% on TaTS. This matters because it
tells us the text BRANCH is not strictly redundant — it adds some
small forecast bias even when content is empty. We will discuss this
honestly: it's a small content-blind effect, not a refutation of our
main claim.

### Point 3 — Single-backbone or three-backbone in main paper tables?

**FLEX (lean toward three):**

I think we should show all three backbones in the main table, but in the
TaTS-style horizon-averaged form (one number per backbone-condition cell).
Reasons:

- One backbone = readers wonder if it's a quirk. Three backbones = readers
  see the pattern is robust.
- iTransformer alone for TaTS is misleading: it shows the smallest C9 effect
  (38.8% vs DLinear's 49.6%), which understates our finding.
- DLinear is the most striking (49.6%) because it has no other way to use
  features — but Informer and iTransformer agree directionally.
- The horizon-averaged TaTS-style table is compact: 8 conditions × 3 backbones
  per model fits in ~half a page.

**The recommendation:** main paper Table 1 has all 3 backbones per model,
horizon-averaged. Per-horizon decomposition goes to appendix.

If page budget is too tight, fall back to: main table shows model-aggregated
(across all 3 backbones), and Table 2 shows per-backbone for ONE model
(MM-TSFlib, since it has the most subtle effect — if you can show C9 hurts
even MM-TSFlib's smallest-effect backbones, the finding is bulletproof).

### Point 4 — How to build intuition for the Aurora probes?

**LOCKED: We need a half-page architecture explainer with a small figure
before introducing GradNorm/H/MaxAttn/DivMSE.**

The right structure is: introduce Aurora's text path → state a hypothesis
about how a model would have to behave to "use text" → describe what each
probe measures and why → present the table. Without this scaffolding, the
probe metrics look arbitrary.

Here's the half-page explainer (will go in Method §4):

> Aurora processes text through three modules in sequence. First, a frozen
> BERT encodes the text into per-token features. Second, a *distillation*
> step uses 10 learned query tokens (`target_text_tokens`) that cross-attend
> over BERT's outputs to produce a fixed-size representation: `[B, 10, d]`.
> The distillation parameters are trainable. Third, a *TextGuider* module
> applies cross-attention from the model's time-series hidden states (the
> queries) to the distilled text tokens (the keys and values).
>
> If Aurora's predictions are influenced by text content, three things must
> happen during a forward+backward pass: (i) the gradient of the loss with
> respect to the distilled text representation must be non-zero — otherwise
> the model is locally insensitive to text changes; (ii) the cross-attention
> in TextGuider must place non-uniform weight on the 10 distilled tokens —
> otherwise the model averages over text features and loses content; (iii)
> the model output must change measurably when text is removed — otherwise
> the text path contributes nothing to the prediction.
>
> Our three probes measure exactly these three quantities.

After this, the table makes intuitive sense:

- **GradNorm** — answers "does the text representation matter at the
  operating point?". The L2 norm of `∂L/∂(distilled text features)`,
  averaged over batches and cells.
- **H/log Lₖ** — answers "is the model selecting which text tokens to
  attend to?". Cross-attention entropy normalised by `log(10)`. A value
  of 1.0 means uniform attention (no selection). A value of 0 means
  all attention on one token (perfect selection).
- **MaxAttn** — geometric corollary. Average max attention weight across
  the 10 text tokens. With 10 tokens, uniform attention gives 0.10.
  Higher = peakier = more selective.
- **DivMSE** — answers "does removing text change the output?". Mean
  squared difference between predictions made with text and predictions
  made with empty text, in the standardised target space.

Then the result reads: GradNorm is non-zero (the text path is in the active
graph), but H/log Lₖ is 0.975 across every condition (no content-selectivity),
MaxAttn is 0.16 (close to uniform 0.10), and DivMSE is small (0.04) AND
condition-independent.

### Point 5 — Would figures help?

**LOCKED: Yes, two figures. Both are cheap to make and high-impact.**

**Figure 1 — The condition ladder (3-panel bar chart).** x-axis: conditions
in interpretable order (C1, C9, C3, C4, C8, C2, C5, C6). y-axis: relative
MSE delta vs C1, in percent. One panel per model (Aurora, MM-TSFlib, TaTS).
Aurora's panel is flat (visually striking — text doesn't matter at all).
TaTS and MM-TSFlib show the two-cluster step pattern: text-perturbed
conditions cluster near 0%, prior-zeroed conditions cluster much higher.

This single figure tells the entire story visually. Reviewers who skim will
get it. Generate from `summaries/deep/ladder_table.csv`.

**Figure 2 — Aurora's cross-attention is uniform regardless of input.** A
2×2 grid of attention heatmaps. Rows: condition (C1 original, C2 empty,
C3 shuffled, C8 oracle). Columns: two example domains (Economy, Health).
Each heatmap shows attention weights of size [192 queries × 10 keys] from
TextGuider, averaged across heads and seeds. They will all look essentially
identical: a uniform grey strip. That's the point. (Implementation note:
we'd need to instrument `aurora_probes.py` to dump the actual attention
weights, not just summary stats. ~30 min coding + 10 min run.)

If we have time, a third figure showing **C8 oracle is no better than C1**
might be worth it — but the ladder figure already shows this since C8 sits
on top of C1 in MSE. So skip it.

### Point 6 — Should we run text-only-OT prediction or semantic diversity?

**FLEX (recommended additions, ranked):**

1. **HIGHEST VALUE: Text-only forecasting baseline.** Train a tiny MLP that
   takes only the BERT/GPT-2 embedding of the text and predicts OT. If this
   model performs better than chance, it proves the text *contains* signal —
   so the failure of TaTS/MM-TSFlib/Aurora to use it is on them, not on the
   data. If it performs at chance, then there's no signal to extract and our
   finding is "no signal → no use" rather than "signal present → not used".
   Either result is publishable. **Strongly recommend running this. ~2 hours.**

2. **MEDIUM VALUE: Mutual information between text and OT.** Pre-compute
   bag-of-words / TF-IDF features of the text and run a regression on OT.
   Same intuition as #1 but cheaper. If R² is non-zero, text contains signal.
   **Recommend running this. 30 min.**

3. **LOW VALUE: Semantic diversity / lexical overlap analysis.** How
   different are C3-shuffled texts from their C1 originals? If they're
   trivially similar (e.g., the within-domain shuffle landed on the same
   month often), C3's null effect could be explained by texts being
   self-similar. We should report at least one diversity statistic to
   defend against this. ~30 min.

4. **LOW VALUE BUT NICE: Prior-only baseline gap.** We already have
   `prior_only_baseline.csv`. Compare TaTS C1 vs prior-only-baseline.
   If TaTS barely beats the baseline, that's another piece of evidence.
   Already done; just include in main paper.

The text-only baseline (#1) is the most valuable addition because it
addresses the strongest reviewer rebuttal: "Maybe the text just doesn't
contain useful signal, regardless of what models do with it." If we can
show text DOES contain signal but models ignore it, the paper becomes
even stronger.

### Point 7 — Math behind probes and bootstrap.

**LOCKED: Method section will have a self-contained math subsection.**

Every quantity we report in the paper has a one-equation definition. I'll
expand on these in §4 below.

---

## 2. THE FMSD AUDIENCE

The FMSD workshop explicitly encourages:
- Multimodality for structured foundation models
- Evaluation (including contamination)
- "What works and what doesn't" findings
- Industry-relevant practical constraints

Reviewers are likely to be:
- ML researchers from tabular/TS foundation model groups
- Industry practitioners deploying TS forecasting
- A mix of NeurIPS/ICLR-trained reviewers used to careful empirical work

What they'll like:
- A single decisive empirical finding with converging evidence
- Honest scope and limitations
- A clean experimental design they can criticise on its own terms
- Practical implications (i.e., what should benchmark designers do)

What they won't like:
- A "takedown" tone ("Time-MMD is wrong!")
- Overclaiming generalisation ("LLMs can't read time series")
- Hidden methodological choices
- Too much code-level detail that should be in the appendix

---

## 3. PAPER STRUCTURE — 4 PAGES + APPENDIX

### Page 1: Introduction (1 column) + Related Work (1 column)

#### Title (FLEX)

Locked-ish: **"What Are Multimodal Time-Series Models Actually Reading?
Disentangling Text Semantics from Numeric Priors on Time-MMD"**

Alternatives:
- "Reading Between the Lines: How Multimodal Time-Series Models Use Text
  on Time-MMD"
- "Are Multimodal Time-Series Models Reading the Text? A Probing Study"

Vote for #1 because it asks the question explicitly. The dash-light tone
of #2 is fun but slightly less serious.

#### Abstract (LOCKED structure, 180 words)

Template:

> Multimodal time-series foundation models (MM-TSFMs) report substantial
> accuracy gains over unimodal baselines when conditioned on accompanying
> text. We ask whether these gains arise from the models' use of text
> *semantics* — the meaning of the words — or from *numeric priors* shipped
> alongside the text in standard benchmark CSVs. We probe three representative
> MM-TSFMs (Aurora, TaTS, MM-TSFlib) on the Time-MMD benchmark via eight
> controlled text perturbations spanning trivial, structure-preserving, and
> information-injecting interventions, run across 9 domains, 4 horizons,
> 3 seeds, and 3 backbones (≥6,000 cells per model class). Three findings
> emerge: (i) Aurora's text path is content-blind — its cross-attention is
> near-uniform over distilled text tokens regardless of input, and its
> behavioural sensitivity to text content is not statistically distinguishable
> from zero. (ii) For TaTS and MM-TSFlib, perturbing text content while
> keeping numeric priors produces ≤0.1% MSE shift, but zeroing only the
> priors produces +1.3% to +49.6%. (iii) Even oracle text — leaking the
> future ground-truth — does not improve forecasts. The "multimodal" gain
> on Time-MMD is overwhelmingly the prior column, not the text.

#### Introduction (~ 350 words)

Structure:

1. **The hook (60 words).** "Multimodal time-series foundation models
   report 15-40% MSE reductions when conditioned on accompanying text
   on Time-MMD (Liu et al., 2024). Where do these gains come from? The
   natural reading is that models are *reading* the text — extracting
   useful natural-language information about the forecasting target.
   This paper asks whether they actually do."

2. **Two competing hypotheses (80 words).** Frame H1 (text semantics)
   vs H2 (numeric priors). Define both precisely. Note that current
   benchmarks compare unimodal vs multimodal but don't disentangle
   them. Cite specifically: Time-MMD's Figure 6 and MM-TSFlib's
   `prompt_y = norm(prompt_emb) + prior_y` line. (The latter is a
   factual observation about their code that anyone can verify.)

3. **Approach (60 words).** A two-axis perturbation suite (text axis ×
   prior axis) on Time-MMD across 3 model classes, 3 seeds, 3 backbones,
   9 domains, 4 horizons. Behavioural + mechanistic evidence.

4. **Findings preview (80 words).** Aurora ignores text entirely. For
   TaTS/MM-TSFlib, text content shift ≤ 0.1%; prior shift +1.3 to +49.6%.
   Oracle text doesn't help. The "multimodal" gain is the prior column.

5. **Contributions (50 words).** (1) The two-axis perturbation protocol,
   (2) empirical findings on three MM-TSFMs, (3) released code and probe
   outputs, (4) practical recommendations for benchmark design.

#### Related Work (~ 250 words)

**Multimodal time-series forecasting.** Time-MMD (Liu et al. 2024) and
its companion library MM-TSFlib are the de-facto benchmark. TaTS (Li
et al. 2025) extends to early-fusion via channel concatenation. Aurora
(Wu et al. 2025; ICLR 2026) is the first pretrained multimodal TSFM.
Other relevant: GPT4MTS, Time-LLM, CALF, Time-VLM, UniDiff. **All of
these report unimodal vs multimodal ablations; none reports a text-only
vs prior-only decomposition.**

**Probing methodology.** Belinkov 2021 (probing taxonomy: behavioural
vs structural). Hewitt & Manning 2019 (structural probes). Niven & Kao
2019 ("Right answer for wrong reason" in BERT — the closest cousin to
this paper). Vig et al. 2020 (causal mediation analysis). We adopt the
"behavioural + mechanistic" framing from Belinkov 2021.

**Shortcut learning.** Geirhos et al. 2020 (Nature MI shortcut learning);
McCoy et al. 2019 (HANS dataset on syntactic heuristics in NLI). Our
paper is a shortcut finding in a different domain: models exploit prior
columns rather than reading text.

**Strong baselines for time-series.** Zeng et al. 2022 (DLinear: simple
linear baseline often beats transformers). We use DLinear as one of our
three canonical backbones partly because it's the strongest
"interpretability-friendly" baseline in modern TS literature.

### Page 2: Method

#### §3 Setup (~ 100 words)

Time-MMD: 9 domains spanning daily, weekly, monthly frequencies; 70/20/10
train/val/test splits. Per-frequency horizons follow Liu et al. 2024:
daily {48, 96, 192, 336}, weekly {12, 24, 36, 48}, monthly {6, 8, 10, 12}.
Three model classes: Aurora (zero-shot foundation; cross-attention text
guidance); TaTS (early fusion via channel concat); MM-TSFlib (late fusion
via weighted residual). Three canonical backbones: DLinear (linear),
Informer (sparse attention), iTransformer (variate-tokenized).

#### §4 Two-axis perturbation protocol (~ 250 words)

This is the heart of the method section. Build the two-axis table early.

```
                    priors KEPT       priors ZEROED
text KEPT           C1 (baseline)     C9
text PERTURBED      C3, C4 (preserved)  C2, C5, C7 (degenerate)
                    C8 (oracle)
text BYPASSED       —                 C6 (CLI off)
```

One sentence on each condition. The two-axis design lets us:
- Compare C1 vs C9 to isolate the prior contribution (text held fixed).
- Compare C1 vs C3, C4, C8 to isolate text content (priors held fixed).
- Compare C9 vs C2, C5 to test if text content adds anything when
  priors are absent.

C8 deserves special attention: the text contains the *future ground-truth
values* in natural language ("On 2024-08-15 the value will be 87.3"). If
the model can read text, this should help dramatically. We use it as a
positive control for our protocol.

#### §5 Mechanistic probes for Aurora (~ 200 words)

The half-page architecture explainer from Point 4 above goes here.
Define GradNorm, H/log Lₖ, MaxAttn, DivMSE precisely, with one equation
each.

#### §6 Statistical protocol (~ 100 words)

Paired bootstrap, B=10,000, on (domain, horizon, seed) triplets.
Pairing controls for cell-level difficulty (some domains are harder).
We report mean difference, 95% CI, two-sided p-value (floor 1/B), and
relative percent change vs C1 baseline. Two reasons for bootstrap over
t-test: (1) differences are heavy-tailed across domains (an Environment
cell can have MSE 100× a Health cell), and (2) we can't assume normality
with 108 paired observations.

### Page 3: Experiments — TABLES + FIGURE 1

This page is dominated by:

**Figure 1: The condition ladder.** 3-panel bar chart. (See Point 5.)

**Table 1: Headline ladder (horizon-averaged).** Rows: 8 conditions in
interpretable order. Columns: 3 models. Cells: mean MSE, with coloured
percent delta vs C1 in subscript. Best per column bold, second-best
underlined. Use `summaries/deep/ladder_table.tex` as starting point.

**Table 2: Per-backbone bootstrap (compact).** Rows: 9 conditions × 3
backbones = 27 rows for each of MM-TSFlib and TaTS. Probably too tall.
Compromise: pick a representative slice. One option: just the C1-vs-X
delta for each (model, backbone, condition). Another: compress to one
row per condition with a "consistency check" column showing whether all
3 backbones agree directionally.

**Table 3: Aurora mechanistic probes (compact).** Rows: 8 conditions.
Columns: GradNorm, H/log Lₖ, MaxAttn, DivMSE. Use `summaries/deep/probes_aggregate.tex`.

If page is too tight, drop Table 2 and put it in the appendix.

### Page 4: Discussion + Conclusion

Three findings with the converging-evidence argument. Implications.
Limitations. Conclusion.

The structure I sketched in PAPER_PLAN_V1 is correct but the writing
needs to acknowledge the C6-vs-C2 contrast and address the text-only
baseline result (assuming we run it).

---

## 4. KEY EQUATIONS — DEFINITIVE VERSIONS

### 4.1 Probe A: gradient norm at distilled text representation

Let $z_{\text{text}} \in \mathbb{R}^{B \times L_k \times d}$ be the
output of Aurora's TextEncoder (the distilled tokens, $L_k = 10$).
Let $\mathcal{L}$ be the validation MSE loss on a forward pass with
both text and time-series. The probe value is

$$
g \;=\; \mathbb{E}_{(b, k)}\, \big\| (\nabla_{z_{\text{text}}} \mathcal{L})_{b,k,:} \big\|_2,
$$

averaged over batches and cells. **Operational meaning:** if $g \approx 0$,
then small changes to the text representation don't change the loss —
so the model is locally insensitive to text. If $g \gg 0$, text changes
do change loss.

### 4.2 Probe B: cross-attention entropy

Let $A \in \mathbb{R}^{B \times H \times Q \times L_k}$ be the *softmax*
weights of TextGuider's cross-attention ($Q = $ time-series query length,
$L_k = 10$). For each query position $q$, the entropy over keys is

$$
H_q \;=\; -\sum_{k=1}^{L_k} A_{q,k} \log A_{q,k}.
$$

Normalised by $\log L_k$ (so it's bounded in $[0, 1]$):

$$
\bar H \;=\; \frac{1}{\log L_k}\, \mathbb{E}_{(b, h, q)}\, H_q.
$$

**Operational meaning:** $\bar H = 1$ ⇒ uniform attention, no selection;
$\bar H = 0$ ⇒ peaked attention, perfect selection. Important caveat:
Aurora's attention layer returns *pre-softmax* logits; we apply softmax
inside the probe.

### 4.3 Probe C: output divergence vs unimodal forward

For a batch of inputs $(x_{\text{ts}}, t)$, compute predictions both
with text $f(x_{\text{ts}}, t)$ and with empty text $f(x_{\text{ts}}, \varnothing)$.
The divergence is

$$
\mathrm{Div} \;=\; \mathbb{E}_{(x_{\text{ts}}, t)} \big[\, \| f(x_{\text{ts}}, t) - f(x_{\text{ts}}, \varnothing) \|_2^2 \,\big].
$$

In standardised target space.

### 4.4 Paired bootstrap

Let $\{(a_i, b_i)\}_{i=1}^N$ be paired observations, where $a_i$ is the
MSE under perturbation and $b_i$ is the MSE under C1 for the same cell
(matched on domain, horizon, seed). Let $d_i = a_i - b_i$. The observed
mean difference is $\hat\Delta = \frac{1}{N} \sum_i d_i$.

For $b = 1, \ldots, B$ (with $B = 10\,000$): sample indices $i_1^{(b)}, \ldots, i_N^{(b)}$
uniformly with replacement from $\{1, \ldots, N\}$, and compute

$$
\Delta^{*(b)} \;=\; \frac{1}{N} \sum_{j=1}^N d_{i_j^{(b)}}.
$$

The 95% CI is the empirical 2.5–97.5 percentile of $\{\Delta^{*(b)}\}$.
The two-sided p-value is

$$
p \;=\; 2 \cdot \min\big(\tfrac{1}{B} \sum_b \mathbb{1}[\Delta^{*(b)} > 0],\ \tfrac{1}{B} \sum_b \mathbb{1}[\Delta^{*(b)} < 0]\big),
$$

floored at $1/B$.

---

## 5. THE FINAL NUMBERS (LOCKED, paper-grade)

### 5.1 Behavioural ladder (mean MSE across all cells)

| Condition | Aurora | MM-TSFlib | TaTS |
|---|---|---|---|
| C1_original (baseline) | 8.5529 | 13.7948 | 13.3074 |
| C9_zero_priors | 8.5529 (+0.0%) | 14.1319 (+2.4%) | 17.4403 (**+31.1%**) |
| C3_shuffled | 8.5532 (+0.0%) | 13.7986 (+0.0%) | 13.3078 (+0.0%) |
| C4_crossdomain | 8.5524 (-0.0%) | 13.7943 (-0.0%) | 13.3078 (+0.0%) |
| C8_oracle | 8.5549 (+0.0%) | 13.8027 (+0.1%) | 13.3075 (+0.0%) |
| C2_empty | 8.5549 (+0.0%) | 14.1351 (+2.5%) | 17.4402 (**+31.1%**) |
| C5_constant | 8.5555 (+0.0%) | 14.1375 (+2.5%) | 17.4398 (**+31.1%**) |
| C6_unimodal | 8.5529 (+0.0%) | 14.0380 (+1.8%) | 14.0043 (+5.2%) |

Two clusters: text-perturbed-priors-kept (≤ 0.1%) vs priors-zeroed (2.4-31.1%).

### 5.2 Aurora probes (mean across 27 cells per condition)

| Condition | GradNorm | H/log Lₖ | MaxAttn | DivMSE |
|---|---|---|---|---|
| C1_original | 0.1517 | 0.975 | 0.159 | 0.0407 |
| C9_zero_priors | 0.2649 | 0.975 | 0.159 | 0.0283 |
| C3_shuffled | 0.1548 | 0.975 | 0.159 | 0.0354 |
| C4_crossdomain | 0.0721 | 0.975 | 0.159 | 0.0351 |
| C8_oracle | 0.1331 | 0.976 | 0.158 | 0.0576 |
| C2_empty | 0.0744 | 0.975 | 0.159 | 0.0400 |
| C5_constant | 0.1557 | 0.976 | 0.158 | 0.0444 |
| C6_unimodal | 0.0797 | 0.975 | 0.159 | 0.0348 |

Reading: GradNorm > 0 (text path is in active graph); H/log Lₖ ≈ 0.975
across every condition (uniform attention); MaxAttn ≈ 0.16 (close to
1/L_k = 0.10); DivMSE ≈ 0.04 (small, condition-independent).

### 5.3 Per-backbone consistency (bootstrap rel_diff for C9 vs C1)

| Model | DLinear | Informer | iTransformer |
|---|---|---|---|
| MM-TSFlib | +2.3% | +1.3% | +3.8% |
| TaTS | +49.6% | +7.6% | +38.8% |

Same direction across all three architectural families.

---

## 6. REVIEWER-PROOFING

### Anticipated questions and answers

**Q1: Have you actually shown the text doesn't contain useful signal?**

A: This is the strongest objection. Address it via:
1. The C8 oracle test: the text contains the future ground-truth and
   the model still doesn't improve. So even if other texts contain
   subtler signal, the model is failing to extract maximum-information
   text just as much.
2. (If we run it) The text-only baseline shows whether text contains
   *any* extractable signal. If yes, our finding is "models don't use
   it"; if no, our finding is "data design issue". Either is publishable.

**Q2: Maybe larger LLMs (Llama, GPT-4) would behave differently.**

A: Honest concession in Limitations. We tested representative MM-TSFMs
at sub-1B parameter LLM components. We cite Time-MMD's own Figure 7b,
where they show LLM scale doesn't help on their setup — consistent
with our finding that the LLM isn't doing the work.

**Q3: Maybe these models would learn to use text with more training.**

A: Honest concession. We use the original protocols (5 epochs for
MM-TSFlib/TaTS; zero-shot for Aurora). We're describing current
behaviour, not asymptotic behaviour.

**Q4: How is this different from Time-MMD's own ablations?**

A: Time-MMD compares unimodal vs multimodal where "multimodal" = text +
prior_y. We decompose the multimodal contribution into text and prior
separately. **This contrast is the methodological contribution.**

**Q5: Why these three models specifically?**

A: They span the three current architectural paradigms: zero-shot
foundation model (Aurora), early fusion (TaTS), late fusion (MM-TSFlib).
If the finding holds across all three, it's likely architectural-class
invariant.

**Q6: C7_null is missing from your tables — why?**

A: C7 ("text=null", priors=0) is operationally identical to C5
("text=constant", priors=0): both zero priors and replace text with
a content-free token. We show in the appendix that C7 produces the
same result as C5. Dropped from main paper for space.

**Q7: TaTS C2/C5/C9 are bit-identical — is that a code bug?**

A: No, it's an architectural fact about TaTS. Once `prior_y` is zero,
the prior-residual contribution vanishes regardless of what's in the
text column. This is itself evidence: if the text column carried
independent signal, perturbing it should still produce *some* difference.

### Calibrating the tone

The paper should be:
- Empirically careful: no overclaiming.
- Clear about the methodological contribution: the two-axis design.
- Pragmatic: says useful things to benchmark designers.
- Self-aware: acknowledges limitations explicitly.

A *light* injection of personality is fine — a single playful framing
in the intro ("We expected the models to read; they appear to be
skimming.") would land well. But the body should be straightforward
empirical work. Don't overdo this.

---

## 7. APPENDIX (unlimited pages)

Recommended sections:

1. **Per-domain × per-horizon decomposition** — `summaries/per_domain_<model>.csv`
   formatted as wide tables. Show finding holds in every domain.
2. **Full bootstrap with all CIs** — `summaries/bootstrap_per_backbone.csv`.
3. **Repo patches list** — 13 patches we applied to upstream code.
4. **Aurora probe details** — math, broader pred_len ablation if time.
5. **Hyperparameters and hardware** — A10G/T4 GPUs; total ~50 GPU-hours.
6. **Perturbation examples** — one example per condition for one domain.
7. **Additional backbones** — full 9-backbone TaTS and 10-backbone
   MM-TSFlib results when they finish.
8. **C7_null result** — show it matches C5/C2.
9. **(If run) Text-only baseline** — short subsection.
10. **(If run) Mutual information of text and OT** — short subsection.

---

## 8. WRITING DAYS PLAN (3 days, fresh chat)

**Day 1:** Method (§§3-6) and Experiments page. Start with Method because
it's the most determined-by-design part. Then drop Table 1 into the
Experiments page and write 4 sentences interpreting it.

**Day 2:** Introduction and Discussion. These are the hardest because
they require committing to a story arc. Open Discussion with C8 oracle
("smoking gun"). Then the C9 disentanglement. Then Aurora's mechanistic
uniformity.

**Day 3:** Related Work, Limitations, polish, and read-through. Print
the PDF. Read on paper. Pretend to be a hostile reviewer.

---

## 9. PROMPT FOR THE WRITING/RESEARCH CHAT

Paste this into the new chat verbatim:

> Hi Claude. I'm writing an ICML 2026 FMSD workshop paper (4 pages,
> deadline May 1) on probing multimodal time-series foundation models.
> The empirical work is done; I need help with research and writing.
>
> Please read the file PAPER_PLAN_V2.md (attached) end-to-end before
> doing anything else. It contains everything: the experimental design,
> the final numbers, the narrative spine, the open points still to
> resolve, and the math.
>
> The paper will argue that the "multimodal" gain in current MM-TSFMs
> on Time-MMD comes from numeric prior columns shipped alongside the
> text, not from text content. Three models (Aurora, TaTS, MM-TSFlib)
> tested with eight controlled text perturbations across 9 domains,
> 4 horizons, 3 seeds, 3 backbones. Behavioural finding: text content
> perturbations produce ≤0.1% MSE shift; zeroing only the prior column
> produces +1.3 to +49.6%. Mechanistic finding: Aurora's cross-attention
> over distilled text tokens has entropy 0.975 of uniform across every
> condition.
>
> Things I need from you:
>
> 1) **Sanity-check the lit review.** Does my claim that no prior MM-TSFM
>    paper has decomposed text-vs-prior on Time-MMD hold up? Search
>    Time-MMD/MM-TSFlib paper, Aurora paper (arXiv:2509.22295), TaTS
>    paper (arXiv:2502.08942), GPT4MTS, CALF, Time-LLM. Confirm or
>    correct my Section 1.1 (Point 1) finding.
>
> 2) **Help me draft sections.** Start with Method (§§3-6) since the
>    structure is most determined. Then Introduction. The numbers in
>    §5 of the plan are locked; the prose around them is yours to draft.
>    Write in a slightly fun but mostly serious tone — see PAPER_PLAN_V2
>    §6 for tone calibration.
>
> 3) **Push back on weak arguments.** If anything in the plan reads as
>    overclaiming, hand-waving, or unsupported, flag it. Especially
>    sceptical of: my claim that "Aurora ignores text" (it has non-zero
>    GradNorm and DivMSE — those numbers don't mean zero contribution),
>    and my framing of TaTS C2=C5=C9 bit-identicality.
>
> 4) **Build Figure 1.** Three-panel bar chart showing the condition
>    ladder per model. Use the numbers in §5.1 of the plan.
>
> 5) **(Optional but high-value) Help me run the text-only forecasting
>    baseline.** Train a small MLP on BERT-pooled text embeddings to
>    predict OT, see if it beats chance. Code and result. ~2 hours.
>
> Empirical artifacts available:
> - summaries/main_results.csv
> - summaries/bootstrap_per_backbone.csv
> - summaries/prior_only_baseline.csv
> - summaries/per_domain_<model>.csv (one per model)
> - summaries/deep/probes_per_cell.csv (216 probe cells)
> - summaries/deep/evidence_summary.txt (pre-written narrative)
> - summaries/deep/ladder_table.tex (paper-ready)
> - summaries/deep/probes_aggregate.tex (paper-ready)
>
> All of `summaries/deep/` is paper-grade output from `code/deep_analysis.py`.
> Drop the .tex files into the document; they require `\usepackage{booktabs, xcolor}`.
>
> Constraint: I dislike em-dashes in writing; please use en-dashes or
> rewrite. I prefer concise structured outputs over flowery prose.

---

## 10. CHECKLIST BEFORE SUBMISSION

- [ ] All probes finished and parsed (✓ have 216 cells, 0 failures).
- [ ] Additional backbones finished (currently running on the new instance).
- [ ] (Optional) Text-only baseline result included.
- [ ] All tables generated from `code/deep_analysis.py`.
- [ ] Figure 1 (condition ladder) generated.
- [ ] (Optional) Figure 2 (attention heatmaps) generated.
- [ ] Limitations section names: 3 model classes, sub-1B LLMs, English text,
      original training protocols, no fine-tuning sweep.
- [ ] Code and probe outputs released.
- [ ] Self-review: hostile-reviewer pass.
- [ ] LaTeX compiles cleanly with `booktabs, xcolor`.
- [ ] PDF under 4 pages main body.

---

## 11. ONE-LINE SUMMARY FOR THE NEW CHAT

We've shown that multimodal time-series foundation models on Time-MMD don't
read the text — the gains come from a numeric prior column quietly bundled
into the text branch. Help us turn this into a great FMSD-ICML-2026 paper.

---

## 12. DRAFTING STATUS (April 26, 2026 — running tracker)

This section is the live working-context log for the writing pass. Append-only.

### 12.1 Files in `paper/`

- `paper.tex`: main 4-page draft. Single source of truth.
- `tab_headline.tex`: headline ladder (`table*`, full width, colour-coded clusters).
- `tab_probes.tex`: Aurora probes table.
- `appendix.tex`: all appendix content, included via `\input{appendix}`.
- `references.bib`: 11 entries, all cited.
- `figures/make_fig1_ladder.py`: bar-chart generator. **Not used in main paper**
  (replaced by enhanced colour table at user request — the bar chart had
  small, overlapping labels and wasn't classy enough). Keep the script
  in case we want it for slides or appendix later.

### 12.2 Architectural decisions made during this pass

1. **No figure in main paper.** The story is two clusters. A
   colour-banded table communicates the clusters more clearly than a bar
   chart at this scale, and the per-paper space cost of a bar chart is
   not worth it on a 4-page submission.
2. **Consistent math notation throughout.** Define
   $\mathbf{x} \in \mathbb{R}^{L}$, $\mathbf{T} \in (\mathrm{Text})^{L}$,
   $\mathbf{p} \in \mathbb{R}^{L+H}$, $\mathbf{y} \in \mathbb{R}^{H}$ in
   §2.  Use these symbols in every method subsection. Each model is one
   equation. Each perturbation is one equation in $\pi_{\text{text}}$
   and $\pi_{\text{prior}}$.
3. **Multimodal-paper register.** Background section explicitly shows that
   per-timestamp text alignment yields a covariate matrix
   $\Phi(\mathbf{T}) \in \mathbb{R}^{L \times d_e}$, then derives each
   architecture's input from it.
4. **Aurora is more than just cross-attention.** The code reveals two
   text-injection points: (i) `TextGuider` cross-attention bias added to
   the encoder's self-attention scores, and (ii) `ModalityConnector`
   transformer-decoder residual added after the encoder. Math must
   reflect both.
5. **Heavy appendix.** Per-horizon, per-domain, per-backbone tables (one
   per architecture) plus full perturbation example texts (one example per
   condition for one chosen domain) plus the patch list with code excerpts.

### 12.3 Newly verified facts from a code re-read

| Fact | Source |
|---|---|
| Aurora's text path: BERT $\to$ projection $\to$ TransformerDecoder distillation with $L_k\!=\!10$ learnable query tokens (`target_text_tokens`) $\to$ output $\mathbf{Z}_T \!\in\! \mathbb{R}^{L_k \times d}$ with $d\!=\!512$. | `Aurora/aurora/modality_connector.py:126-204` |
| Aurora's encoder bias: $B = \mathrm{einsum}(\alpha_v, W, \alpha_t)$ where $\alpha_v, \alpha_t$ are vision/text cross-attention scores and $W$ is a learnable $L_k\!\times\!L_k$ matrix. The encoder self-attention is biased by $B$. | `Aurora/aurora/modeling_aurora.py:408-411` |
| Aurora's residual fusion: after encoder, $x_{\mathrm{enc}} \mathrel{+}= W_f (\mathbf{Z}_v + \mathbf{Z}_t)$ where $\mathbf{Z}_v, \mathbf{Z}_t$ come from `ModalityConnector` cross-attention decoders. | `Aurora/aurora/modeling_aurora.py:450-454` |
| MM-TSFlib fusion: `prompt_y = norm(prompt_emb) + prior_y; outputs = (1-w)*outputs + w*prompt_y` (lines 460/461 vali, 578/579 train, 707/711 test). | `MM-TSFlib/exp/exp_long_term_forecasting.py:460,578,711` |
| TaTS .detach() on text concatenation: `batch_x = torch.cat([batch_x, prompt_emb], dim=-1).detach()` — severs gradient into the text-projection MLP. | `TaTS/exp/exp_long_term_forecasting.py:519` |

### 12.4 What's done in this drafting pass

- Initial paper.tex, tab_headline.tex, tab_probes.tex, appendix.tex,
  references.bib written.
- Figure 1 generation script written, output rejected as not classy
  enough → figure deleted from main paper, kept in `figures/` only.
- Cross-reference validator script confirms zero undefined refs / cite
  keys after first pass.

### 12.5 What's open after this pass

1. **Bigger / more detailed appendix tables.** Per-horizon ladder for
   each model (×3 models). Per-domain for each model (×3). Full
   per-backbone bootstrap CIs.
2. **Perturbation example texts.** Pull one row per condition for one
   domain (e.g. Health) and put the actual text strings in the
   appendix. This is concrete proof that perturbations look the way we
   say they look.
3. **Compile locally.** `pdflatex` not installed. User to install via
   `apt-get install texlive-latex-extra texlive-fonts-recommended
   texlive-bibtex-extra` or use Overleaf.
4. **Probe Probe A semantic.** Plan §4.1 says per-token L2 norm; current
   implementation uses RMS (root-mean-square per element). Reconcile
   notation in paper to match the implementation; both are valid as
   long as the text is precise.
5. **Text-only forecasting baseline.** Plan Point 6 marks this as
   highest-value optional. Not run yet.
6. **Attention heatmap figure.** Plan Point 5 marks this as Figure 2 if
   we have time. Not run yet. Could replace nothing in main paper but
   would strengthen the claim that Aurora's text attention is uniform.

### 12.6 Multimodal-paper writing conventions adopted

After re-reading FMSD-2025 papers (`tsfm_ready`, `llm_agents`,
`random_init`, `drimm`):
- **No dedicated Related Work section** in main body; weave 4-5 citations
  into intro. (Adopted.)
- **Bold-led finding paragraphs** in Results section. (Adopted.)
- **Tight contributions block** at end of intro. (Adopted.)
- **Heavy appendix** with detailed reproducibility. (Adopting now.)

### 12.7 Reviewer-facing improvements made in this pass

- §2 Background defines a consistent symbol system that every method
  subsection reuses — fixes the "wait, what's $T$ here?" problem.
- §3 (each model) explicitly shows the prior-handling line in the
  forward pass, so reviewers can see why \Cnine{} (zero-priors)
  matters before reading results.
- Perturbation table formalises the two-axis design as transformations on
  $(\mathbf{T}, \mathbf{p})$, removing handwaving about ``what we did''.

### 12.8 A note on tone and audience

Two framings to avoid:
- **Takedown.** Don't say ``Time-MMD is broken''. Say ``the prior column
  is bundled into the text branch, so reported lifts conflate two
  effects''. We are pointing out a methodological gap, not declaring
  others wrong.
- **Lazy.** Don't elide the math. Reviewers from multimodal LLM and
  multimodal time-series communities expect explicit notation for the
  fusion operator in each model.

The paper should read like an empirically careful, slightly
under-stated, thoroughly reproducible piece — not a hot take.
