# PAPER_DRAFT_SEEDS.md — Starter prose for each section

These are draft passages written to seed the writing process. They are
NOT final. The fresh writing chat should use them as starting points,
push back on weak arguments, and rewrite as needed. The numbers and the
narrative spine are locked; the wording is open.

Tone notes from Karthik (Author):
- No em-dashes. Use en-dashes or rewrite.
- No AI-robotic structure: vary sentence length, use occasional first
  person plural ("we"), let some sentences breathe.
- A *little* fun is allowed. Stay mostly serious.
- Reader should not need to know what "H/log Lₖ" means before §5.

---

## ABSTRACT (180 words, draft)

Multimodal time-series foundation models report substantial accuracy gains
when conditioned on accompanying text. We ask whether these gains come from
the meaning of the words or from a numeric "prior" column shipped alongside
the text in standard benchmark CSVs. We probe three representative models —
Aurora, TaTS, and MM-TSFlib — on the Time-MMD benchmark across 9 domains,
4 horizons each, 3 seeds, and 3 architecturally diverse backbones, using
eight controlled text perturbations designed along two axes (text content
× prior availability). Three findings emerge. (i) Aurora's text path is
content-blind: its cross-attention over distilled text tokens has near-uniform
entropy regardless of input, and its predictions are statistically indistinguishable
from its no-text predictions. (ii) For TaTS and MM-TSFlib, perturbing text
content while keeping the prior column produces ≤ 0.1% MSE shift; zeroing
only the prior column produces +1.3% to +49.6%. (iii) Even oracle text
that contains the future ground-truth values does not improve forecasts.
The "multimodal" benefit on Time-MMD is overwhelmingly the prior column,
not the text.

---

## §1 INTRODUCTION (draft, 350 words)

When you train a multimodal time-series model on Time-MMD and condition
the forecast on accompanying text, the mean squared error drops by 15-40%
compared to a unimodal baseline (Liu et al., 2024). It is tempting to read
this as the model *understanding* the text: news articles about influenza,
USDA reports on broiler chicken prices, weather summaries. Each domain
ships with a free-text column of contextual information; adding that
column to the input clearly helps; therefore the text helps. The story
writes itself.

This paper checks the story. We pick eight controlled perturbations of the
text column and rerun three representative multimodal models on Time-MMD.
Some perturbations leave the text intact (C1). Some shuffle the text within
a domain so it no longer matches the date. Some replace it with a constant
string. One leaks the future ground-truth into the text (an "oracle" condition).
Crucially, one keeps the original text but zeros out a separate numeric
column — `prior_history_avg`, a smoothed running average of past values —
that ships alongside the text in every Time-MMD CSV. This last condition
isolates one variable cleanly. If text content is doing the predictive work,
keeping it intact should preserve performance.

It does not. Across 9 domains, 4 horizons, 3 seeds, and 3 backbones:
perturbing text content while leaving the prior column intact moves MSE by
less than 0.1%. Zeroing only the prior column, while leaving text intact,
moves MSE by 1.3% to 49.6% depending on model and backbone. Oracle text
that literally contains the future does not improve performance. Aurora's
cross-attention over its distilled text representation has the same near-uniform
weight distribution regardless of what is in the text. The "multimodal"
gain on this benchmark is the prior column.

We document this with both behavioural evidence (perturbation-driven MSE
shifts with paired bootstrap confidence intervals) and mechanistic evidence
(gradient norms, attention entropy, and output divergence in Aurora). The
finding does not show that current models *cannot* use text; it shows that
they currently *do not* on Time-MMD. We discuss what this means for
benchmark design, why it might be happening, and how future work should
report multimodal gains.

**Contributions:**
1. A two-axis perturbation protocol that disentangles text content from
   numeric priors.
2. Empirical evidence that three representative MM-TSFMs on Time-MMD
   exhibit this pattern.
3. Mechanistic confirmation via Aurora attention probes.
4. Released code, perturbed datasets, and probe outputs.

---

## §2 RELATED WORK (draft, 250 words)

**Multimodal time-series forecasting.** Time-MMD (Liu et al., 2024) and its
companion library MM-TSFlib introduced the de-facto multimodal forecasting
benchmark spanning nine domains. TaTS (Li et al., 2025) extends to early
fusion via channel concatenation of LLM embeddings. Aurora (Wu et al.,
2025; ICLR 2026) is the first pretrained multimodal time-series foundation
model with cross-attention text guidance. Several other architectures
(GPT4MTS, Time-LLM, CALF, UniDiff) report multimodal gains on Time-MMD or
similar benchmarks. Common to all: ablations compare unimodal vs multimodal,
where "multimodal" includes both text and any prior numeric features
bundled into the text branch. None decomposes the multimodal contribution
into text-vs-prior. The MM-TSFlib code's `prompt_y = norm(prompt_emb) + prior_y`
makes the entanglement explicit.

**Probing methodology.** Belinkov (2021) provides the modern taxonomy of
probing methods (behavioural, structural, intervention-based). Hewitt and
Manning (2019) demonstrated structural probes for syntax. Niven and Kao
(2019) is the closest methodological cousin: they showed that BERT's
strong performance on argument reasoning was driven by a single linguistic
artifact, illustrating "right answer for wrong reason". We adopt their
behavioural-plus-mechanistic framing.

**Shortcut learning.** Geirhos et al. (2020) named the broader phenomenon;
McCoy et al. (2019) demonstrated it for syntactic heuristics in NLI. Our
finding is a shortcut in a different setting: models exploit a prior
column rather than reading text.

**Strong unimodal baselines.** Zeng et al. (2022) showed that simple linear
models (DLinear) often beat transformers on time-series forecasting. We
include DLinear among our three canonical backbones.

---

## §3 SETUP (draft, 100 words)

**Dataset.** Time-MMD spans 9 domains across daily, weekly, and monthly
frequencies. We follow the per-frequency forecast horizons of Liu et al.
(2024): daily {48, 96, 192, 336}, weekly {12, 24, 36, 48}, monthly
{6, 8, 10, 12}. Each row of every CSV contains a numeric target OT, a
free-text column ("fact" or "Final_Search_4"), and a numeric prior column
("prior_history_avg") computed as a smoothed running average of OT.

**Models.** We probe three model classes representing the current
multimodal architectural paradigms: **Aurora** (zero-shot pretrained
foundation model with cross-attention text guidance), **TaTS** (early
fusion via channel concatenation of LLM embeddings), and **MM-TSFlib**
(late fusion via weighted residual). For TaTS and MM-TSFlib we evaluate
three backbones — DLinear, Informer, and iTransformer — chosen to cover
linear, sparse-attention, and variate-tokenized architectures.

---

## §4 PROTOCOL (draft, 250 words)

**The two axes.** Each Time-MMD row contains two channels of contextual
information: the text column and the numeric prior column. We design our
perturbations so that each one independently varies the text axis, the
prior axis, or both.

[Insert two-axis design table here.]

The eight conditions:

- **C1_original**: text and priors as shipped (baseline).
- **C2_empty**: text replaced with empty string; priors zeroed.
- **C3_shuffled**: text shuffled within domain so it no longer aligns
  with the date; priors kept intact.
- **C4_crossdomain**: text replaced with text from a paired domain at
  the same date (e.g., Economy text in place of Agriculture text);
  priors kept intact.
- **C5_constant**: text replaced with the constant string "Time series
  data point."; priors zeroed.
- **C6_unimodal**: text branch architecturally bypassed via a CLI flag.
  Uses C1's CSV but ignores it at the data loader.
- **C7_null**: text replaced with the string "null"; priors zeroed. We
  drop this from main paper tables (redundant with C5; appendix only).
- **C8_oracle**: text replaced with a natural-language description of
  the future ground-truth values ("On 2024-08-15 the value will be 87.3").
  Priors kept intact.
- **C9_zero_priors**: text kept identical to C1; priors zeroed. The key
  contrast for our claim.

**Crucial contrasts.** C1 vs C9 isolates the prior column with text held
fixed. C1 vs C3, C4 isolates text content with priors held fixed. C9 vs
C2, C5 tests whether text content adds anything when priors are absent.
C8 vs C1 tests whether the model can extract maximally informative text
when given it.

---

## §5 MECHANISTIC PROBES (draft, 300 words)

We probe Aurora's text path because Aurora has the cleanest text-handling
architecture of the three models. The path consists of three modules:
(i) a frozen BERT that encodes raw text into per-token features; (ii) a
distillation step in which 10 learned query tokens cross-attend over BERT's
outputs to produce a fixed-size representation `[batch, 10, dim]`; (iii)
a TextGuider module that performs cross-attention from the model's
time-series hidden states (queries, length 192) to the distilled text
tokens (keys and values, length 10). The distillation parameters and the
TextGuider are trainable; BERT is frozen.

If Aurora's predictions depend on text content, three things must be true
simultaneously. The loss must be locally sensitive to changes in the
distilled text representation (otherwise text changes don't propagate).
The cross-attention must place non-uniform weight on the 10 distilled
tokens (otherwise it averages text features into a content-blind blob).
And the predictions must change measurably when text is removed
(otherwise the text path doesn't contribute to the output). We define
three probes that measure exactly these three quantities.

**Probe A: gradient norm at the distilled text representation.** The L2
norm of $\partial \mathcal{L} / \partial z_{\text{text}}$, averaged over
batch and across the 10 distilled tokens. A value near zero means the
loss landscape is locally flat in the text direction.

**Probe B: cross-attention entropy.** The entropy of TextGuider's
attention weights over the 10 distilled tokens, normalised by $\log 10$
so that a value of 1.0 corresponds to uniform attention. We softmax the
pre-softmax scores returned by Aurora's attention layer before computing
the entropy.

**Probe C: output divergence.** Mean squared difference between predictions
made with text and predictions made with empty text, in the standardised
target space. Computed on the same batch with the same time-series input.

[Insert Aurora probe table here.]

---

## §6 STATISTICAL PROTOCOL (draft, 100 words)

We compare each non-baseline condition $X$ to $C_1$ via paired bootstrap
on (domain, horizon, seed) triplets. For each pair $i = 1, \ldots, N$
(with $N = 9 \times 4 \times 3 = 108$ for TaTS and MM-TSFlib, scaled by
backbone count), we compute $d_i = \mathrm{MSE}^{X}_i - \mathrm{MSE}^{C_1}_i$.
We resample $\{d_i\}$ with replacement $B = 10\,000$ times to estimate
the bootstrap distribution of the mean. We report the observed mean
difference, the empirical 95% CI, the relative percent change vs C1's
mean, and a two-sided p-value floored at $1/B$. Pairing reduces variance
because some cells are intrinsically harder than others; bootstrap
avoids the normality assumption.

---

## §7 RESULTS (draft, 600 words)

[Insert Figure 1: condition ladder, three panels.]

[Insert Table 1: headline ladder with coloured pct deltas.]

The condition ladder reveals two clusters cleanly. **Conditions that
perturb text content while keeping the prior column intact** (C3 shuffled,
C4 crossdomain, C8 oracle) sit on top of C1 with relative MSE shift below
0.1% across all three model classes and all three backbones. **Conditions
that zero the prior column** (C2, C5, C9) cluster much higher: between
+2.4% and +2.5% for MM-TSFlib and uniformly +31.1% for TaTS.

**Three observations.**

First, *the C8 oracle condition is striking*. We replaced the text with
a natural-language description of the future ground-truth values: "On
2024-08-15 the value will be 87.3". A model that reads English would
treat this as a perfect oracle and improve dramatically. None of the
three models does. C8 sits within seed noise of C1 on every model and
backbone we tested.

Second, *the C9 condition isolates the prior contribution exactly*. Text
is identical to C1; only the prior column is zeroed. The MSE shift is
+31.1% for TaTS and +2.4% for MM-TSFlib, identical in magnitude to the
shift under C2 (which zeroes both text and prior). Text content is
adding no measurable signal beyond what the prior column carries.

Third, *the pattern is consistent across architecturally diverse backbones*.
DLinear (linear), Informer (sparse attention), and iTransformer
(variate-tokenized) all show the same direction, although DLinear shows
the largest relative effect for TaTS (49.6%) because it has fewer
alternative ways to use features.

**Aurora is content-blind.**

[Insert Table 3: Aurora probes by condition.]

Aurora's behavioural insensitivity to text content (C1 to C9 difference
is 0.0%) is corroborated mechanistically. Probe A's gradient norm is
non-zero in every condition, indicating the text path is in the active
computation graph. But Probe B's normalised attention entropy is 0.975
across every condition we tested, including the oracle. Aurora's
TextGuider attends to the 10 distilled text tokens nearly uniformly
regardless of input. Probe C's output divergence is small and similar
across conditions, suggesting the text path adds a content-blind bias
to predictions rather than a content-conditional signal.

**Bootstrap confirms across backbones.**

[Insert Table 2: per-backbone bootstrap subset.]

Paired bootstrap with B=10,000 confirms the cluster pattern at the
backbone level. For every (model, backbone) pair, the C9 vs C1 effect
is significant (p < 0.001) and large; the C3, C4, C8 effects are
either not statistically significant or significant with relative
effect sizes below 0.2%.

**The prior column is a strong baseline on its own.**

A naive baseline that predicts $y_{t+h} = \mathrm{prior\_history\_avg}_{t+h}$
(no model, no training) achieves MSE within a few percent of TaTS's C1
on most domains. The trained models add real value over this baseline,
but most of that value comes from refining the prior, not from reading
the text. We provide per-domain prior-only MSE in the appendix.

---

## §8 DISCUSSION (draft, 600 words)

**Three findings, in summary.**

Aurora's text path is content-blind. Behaviourally, predictions under
unimodal inference are bit-identical to predictions under multimodal
inference with original text (C1 = C6). Mechanistically, attention is
near-uniform across all conditions, and gradient norm at the text
representation is small and condition-independent.

For TaTS and MM-TSFlib, the "multimodal" gain on Time-MMD is the prior
column. Text content perturbations produce shifts within seed noise;
zeroing the prior column produces shifts of 1.3-49.6%. The C8 oracle
result rules out the explanation that other texts are too uninformative.

These findings hold across architecturally diverse backbones (DLinear,
Informer, iTransformer) and across all 9 Time-MMD domains.

**Why might this be happening?**

Several non-mutually-exclusive explanations seem possible. (a) The text
column in Time-MMD is largely redundant with `prior_history_avg`: most
domain-specific contextual information is already implicit in recent
values. (b) The text representations are overly compressed: Aurora's
distillation to 10 tokens may discard fine-grained content while
preserving a coarse domain prior that's also recoverable from numeric
features. (c) The training protocols (5 epochs for TaTS/MM-TSFlib;
zero-shot for Aurora) do not give models enough optimisation pressure
to learn to read text. (d) The LLM backbones tested (BERT-base for
Aurora, GPT-2 for the others) lack the capacity for semantic reasoning
on this domain.

We do not adjudicate between these. Showing which one is correct would
require either richer benchmarks (where text contains signal not in
priors) or capability-scaled experiments (where larger LLMs are tested
under matched training budgets).

**Implications for benchmarks.**

Two practical recommendations follow. First, future MM-TSFM papers
reporting gains on Time-MMD should ablate the prior column independently
from the text. Otherwise the "multimodal gain" headline is consistent
with two very different stories — text helps, or text vanishes and the
prior helps — and current ablations cannot distinguish them. Second,
benchmark designers should consider perturbation-based "text-content
sensitivity tests" as part of evaluation. Our 8-condition protocol is
released for this purpose.

**Limitations.**

We tested three model classes with sub-1B-parameter LLM components.
Larger LLMs (Llama-3 70B, GPT-4) might behave differently, although
Time-MMD's own Figure 7b shows LLM scale does not improve their setup,
weakly consistent with our finding. Time-MMD's text is English; we
cannot speak to multilingual settings. We use the original training
protocols; longer training or fine-tuning might change behaviour. Our
probes are correlational interventions, not causal mediation analyses.

A remaining concern is whether the text in Time-MMD actually contains
predictive signal that any model could extract. We address this with
the C8 oracle test: if text contained no extractable signal,
sensitivity to perturbations could be vacuously zero. But C8 contains
the future ground truth, and even C8 doesn't help. Whatever our models
are doing, they're not reading.

---

## §9 CONCLUSION (draft, 60 words)

The "multimodal" gain on Time-MMD is the prior column. Three multimodal
time-series foundation models, eight perturbations, three backbones,
nine domains, four horizons, and three seeds all tell the same story:
text content is not being read in any of them. We release code and
probe outputs; future MM-TSFM evaluations should ablate the prior
column independently.

---

## ONE-LINER VERSION OF THE PAPER

> Multimodal time-series foundation models on Time-MMD beat their
> unimodal counterparts not because they read the accompanying text,
> but because the multimodal branch quietly bundles a smoothed running
> average of the target into its input.

Use this as the social-media one-liner if the paper is accepted.

---

## NOTES TO THE WRITING CHAT

1. The locked numbers in the tables above must not be changed. The locked
   experimental design must not be changed. Everything else is flexible.

2. Keep an eye out for places where I've claimed certainty without
   justification. Push back. Examples:
   - "The text column is largely redundant with prior_history_avg" —
     do we have evidence beyond inference? Maybe run a quick correlation.
   - "Models lack capacity for semantic reasoning" — we haven't tested
     larger LLMs; soften this.

3. The intro and discussion should not duplicate. Keep intro focused
   on what we did and what we found; discussion on what it means.

4. Cut hard before submitting. Each draft section above is at the
   upper bound of word count. The 4-page limit is real and ICML's
   formatting is dense.

5. Check FMSD 2026 formatting requirements (ICML 2026 LaTeX template).
