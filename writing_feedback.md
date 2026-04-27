Let me thoroughly research the workshop requirements and all referenced papers/repos before giving you comprehensive feedback.Good. Now let me read the core papers for fact-checking.Now let me look at TaTS and MM-TSFlib.Let me now check the well-written paper example mentioned by the user:I now have a thorough picture. Let me compile the complete editorial and technical feedback.

---

# Complete Editorial & Technical Feedback

## 0. Workshop Context First

**FMSD @ ICML 2026 specifics you must internalize before editing:**
- 4 pages max (body), unlimited references + appendix. Reviewers are *not obligated* to read the appendix.
- Double-blind anonymization.
- The workshop explicitly calls out "contamination and memorization can dominate reported gains; we explicitly encourage protocols and benchmarks that measure and mitigate contamination" — this is your paper's entire contribution. Frame it in that language.
- The workshop explicitly encourages "LLM/agent-based work when evaluation centers on predictive tabular/time-series objectives and includes... numerical reliability checks." Your paper *is* a numerical reliability check on the multimodal literature.
- Your paper competes with work like Zhang et al. (2506.21611, "Does Multimodality Lead to Better Time Series Forecasting?"), which investigates the same question (when does text help?) but with a different angle. You need to clearly differentiate: you do mechanistic falsification with a specific confound (the numeric prior), not a general benchmarking sweep.

---

## 1. Factual Errors — Fix Before Anything Else

### 1a. Critical Citation Error: TaTS author
**Current (wrong):** Zhou, T. et al. TaTS: Text as time series... 
**Correct:** TaTS is authored by **Zihao Li, Xiao Lin, Zhining Liu, et al. at UIUC**. The arXiv paper (2502.08942) is titled "Language in the Flow of Time: Time-Series-Paired Texts Weaved into a Unified Temporal Narrative." The correct citation is **Li et al., 2025** (EMNLP 2025). This is also how Aurora's paper cites it: "TATS (Li et al., 2025d)." Every reference to "Zhou et al., 2025" for TaTS is wrong. (Note: Zhou et al., 2021 is Informer, and that is correct.)

### 1b. Aurora venue is wrong
**Current:** Wu et al., 2025. Aurora... arXiv preprint arXiv:2509.22295, 2025. 
**Correct:** Aurora is accepted at **ICLR 2026** (confirmed: GitHub repo says "[ICLR 2026]", paper appears in ICLR 2026 schedule). Update to: Wu et al., ICLR, 2026. 

### 1c. TaTS text encoder: verify "frozen GPT-2 pooler"
The draft states: *"ϕ is a frozen GPT-2 pooler (Radford et al., 2019)"* for TaTS. The TaTS paper says they use "pre-trained large language models to encode the texts by applying pooling over embeddings of individual tokens." It does **not** specifically say GPT-2 in the abstract or introduction — you must verify this against the actual TaTS repo code. If the encoder is something other than GPT-2 (e.g., all-mpnet or RoBERTa), this is a factual error that undermines your architecture section. **Before finalizing, open TaTS's config or requirements files and confirm the default text encoder.**

### 1d. Probe C sample count inconsistency
- Section 5 (Probe C definition): *"average 10 samples from the flow-matching head per setting"*
- Appendix A: *"AURORA's flow-matching head averages 100 samples for inference"*

These disagree. Pick one and be consistent everywhere. If you used 10 samples for probing but 100 for inference, say so explicitly and justify why 10 is sufficient for the divergence measurement.

### 1e. Missing major related paper
**Zhang et al. (2025), "Does Multimodality Lead to Better Time Series Forecasting?" (arXiv 2506.21611, AWS)** investigates the same question on overlapping data (14 datasets, 7 domains including Time-MMD). They also find multimodal gains are not universal. You must cite this prominently, and distinguish your contribution: they identify *when* text helps via model/data properties; you localize *why* reported gains are artifacts (numeric prior confound). They do not perform the prior-zeroing probe or the mechanistic analysis. Make this distinction sharp.

### 1f. Aurora text encoder description may be imprecise
You say Aurora uses "frozen BERT producing per-token features" — check whether Aurora uses BERT-base or a sentence encoder. The Aurora paper describes "TokenDistillation" where text is tokenized by BERT, then distilled into Ktext learnable tokens before entering the TextGuider cross-attention. The intermediate step means the Lk tokens are *distilled*, not directly the BERT per-token output. The notation ZT = CrossDec(Q, E) is not exactly how Aurora describes it — their distillation is a learned cross-attention between learnable queries Q and BERT output E. This is correct in spirit but should align with Aurora's exact notation (they call it TextDistiller and TextGuider). Also verify whether Ktext (their notation) = 10 (your Lk). This specific hypervalue should come from inspecting the Aurora config file in the released checkpoint.

---

## 2. Structural and Narrative Issues

### 2a. The paper reads like a technical report, not a workshop paper
Every FMSD workshop paper I've seen that gets accepted starts with a crisp, stinging opening that makes the reviewer sit up. Your abstract is decent — the finding is strong — but the introduction is sprawling. It buries the lede. Compare to Zhang et al. (2506.21611): they have a figure-first opening, a clear question hierarchy (RQ1 through RQ6), and a 1-sentence summary of findings per RQ. You have a dense prose wall.

**Reorder the introduction:**
1. Open with the punchline (1-2 sentences): *"We show that the multimodal gains reported on Time-MMD are entirely explained by a co-shipped numeric column, not by natural-language text."*
2. Then: *"Here's why this matters" (benchmark contamination, benchmark design implications).*
3. Then: a 1-paragraph background on the three architectures.
4. Then: the confound story.
5. Then: contributions as a 3-item bulleted list, each ≤2 lines.

### 2b. Section 2 (Setup and Notation) is a wall of definitions nobody will absorb
The T, p, y, x notation introduced cold feels like a formalism for its own sake. Readers don't know *why* you need p until Section 4. Consider this restructuring:

> First, show a concrete Time-MMD example (the Table 4 example — this is gold, move it to the main body or a figure). Let the reader see the public health row: here's the numeric target, here's the text, and here's this mysterious prior_history_avg column sitting right there. *Then* introduce the notation (x, T, p, y) anchored to that example. This is the "show before tell" principle.

You could create a small 3-column figure (like a miniaturized data card): one row showing numeric history, one row showing the text that accompanies it, and one row showing the prior column with its future values revealed. This visual alone will make the rest of the paper click.

### 2c. Section 3 (Three Architectures) math is introduced without adequate scaffolding
The equations for TATS, MM-TSFLIB, and AURORA appear as three dense blocks without a unifying story. The reader should understand **before the equations** that:
- All three architectures share a common template: fuse text embeddings E(T) with numeric prior p into a backbone fθ(x)
- They differ only in *where* fusion happens (early/late/cross-attention) and *whether p is in the same fusion path as E*
- This shared structure is exactly what makes your two-axis protocol possible

**Structural fix:** Add a 4-line prose paragraph before Eq. (3) that says: "All three architectures fit the template ŷ = g(fθ(x), E(T), p) for some fusion function g. They differ in fusion depth and whether E and p enter through the same additive path — a structural property that determines susceptibility to the prior confound." Then present each architecture as an instantiation of g.

**A figure showing all three fusion paths side-by-side** (mini architecture diagram: 3-column figure, one column per model, showing where T and p enter the computation graph) would replace 500 words and be more informative. For a 4-page paper, this is worth its space.

### 2d. Section 4 (Perturbation Protocol) is dense but not bad — tighten it
The prose around the perturbation table is repetitive. The sentence "Comparing C2/C5/C9 therefore isolates the text channel's contribution when the prior is held at zero" is exactly right but buried in a paragraph with 4 other sentences saying the same thing. Cut to: *"C2, C5, C9 form a trio with identical prior treatment (p=0) but different text values: empty, constant, real. If text is read, these should diverge. They don't."* That's 3 sentences instead of 1 paragraph.

**C6 explanation** is currently placed in the middle of Table 1's caption AND has its own prose subsection. Merge them — define C6 once, clearly, and don't repeat it.

### 2e. Section 5 (Mechanistic Probes) math is dumped without narrative
The probes are the most technically interesting part of the paper and they're presented as three cold equations. A reader who isn't already in the Aurora codebase will not understand what gb, Hb, Db are or why you're measuring them.

**Before Equations 6-8, write 2-3 sentences:**
> "Aurora's text pathway has two potential failure modes: (i) gradients could fail to reach the text parameters, meaning the checkpoint never learned to use text; or (ii) the forward pass could be insensitive to text content even if gradients flow. Probe A checks (i); Probes B and C check (ii) — they measure whether the cross-attention *discriminates* between distilled tokens and whether the final prediction moves at all when text is replaced."

This framing makes the three probes feel like a logical sequence rather than a random grab-bag.

### 2f. Section 7 (Results) is the worst-written section in the paper
You have 7,776 runs and the results are crystal-clear, but the section reads like a stream of consciousness. Issues:
- You repeat the finding ("text doesn't matter, prior does") four times in slightly different words across three paragraphs.
- Numbers appear without context ("up to 31.1%" — from which architecture, which domain, which condition?)
- "Bootstrap confidence intervals are tight and either straddle zero or move only in the third decimal of MSE" tells me nothing concrete. What are the actual intervals? Reference Table 5.
- The AURORA sub-finding (bit-identical predictions) is buried in a paragraph with two other findings.

**Restructure Results as three focused sub-paragraphs with bolded leads:**

> **The prior, not the text, explains the multimodal gap.** [1 paragraph, lead with +31% TATS, reference Table 2's C9 row, contrast with ≤0.06% for text-only conditions]

> **Text content is undetectable, including ground-truth oracle text.** [1 paragraph, focus on C8 — oracle condition — being the most damning result: the model cannot benefit from knowing the exact future embedded in text]

> **Aurora's text pathway is wired but silent.** [1 paragraph, reference the 3 probes and their key values: Hb=0.975, Db≤0.06, bit-identical predictions across C1/C6/C9]

### 2g. Discussion section is too defensive and too short
The three paragraphs are written apologetically. "This is not a takedown of TATS as a research artefact" — this disclaimer is in Appendix F, which is fine there. In the Discussion, be more direct about implications. Write:

> "The findings generalize a broader pattern: when a 'text' control in a benchmark zeros two quantities simultaneously (semantic content *and* a numeric signal), any apparent text effect is uninterpretable."

Then make the constructive recommendation sharp: prior-zeroed and empty-text controls should be reported alongside any headline multimodal number. Frame this as the paper's policy contribution.

Also add a 2-sentence explanation of **why this matters beyond Time-MMD** — if similar prior-like signals exist in other multimodal TS benchmarks, your protocol should generalize.

---

## 3. Mathematical Presentation Issues

### 3a. Equation (2) — the covariate matrix E — is never used correctly downstream
You define E(T) ∈ ℝ^{L×de} but then in Eq. (3) you write ψ(E) which applies ψ row-wise. This is implicit. Be explicit: write ψ: ℝ^{de} → ℝ^{dp} applied row-wise, giving [ψ(ϕ(T1)); ...; ψ(ϕ(TL))] ∈ ℝ^{L×dp}. Otherwise readers think ψ takes the whole matrix.

### 3b. Eq. (3) — TATS equation — has a bracketing issue
fθ([x; ψ(E)]) — the concatenation is along the *channel* axis (feature dimension), so you need to specify this. Write [x ‖ch ψ(E)] or add a subscript: [x; ψ(E)]_{channel}. Without this, a reader might think you're concatenating along the time axis.

### 3c. Eq. (4) — MM-TSFLIB — the normalization is underspecified
You write norm(ϕ̄(T)) but don't say what norm. From the code, this is likely an L2 normalization or instance normalization. Specify it.

### 3d. Eq. (5) — AURORA — is too schematic
ŷ^AURORA = fAURORA_θ(x, αt, E) doesn't communicate the two-stage process (distillation → TextGuider → Modality-Guided MHSA). Given that Probes A/B/C are anchored to specific components, the description of the Aurora architecture needs to be precise enough that the probes make sense. Specifically:
- ZT is defined as the output of CrossDec(Q, E) — say explicitly that Q ∈ ℝ^{Lk×d} are *trainable* queries and BERT parameters are frozen
- αt is the cross-attention output from the TextGuider that then modifies the temporal backbone's self-attention — say this clearly
- ConnText is a separate path — mention it briefly and say it's also probed indirectly through Probe C

### 3e. Eq. (6) — Probe A gradient norm — is correct but the expectation index is confusing
gb = √E_(b,k,j)[|∇_{Z_T} L_{b,k,j}|²] — (b,k,j) indexes over batch, distillation token, and feature dimension. But b is both the outer expectation variable and the subscript of gb. This is sloppy notation. Either write it as g = √(1/BKd) Σ_{b,k,j} [...] or change gb to g and explain "b" denotes averaging over a batch. Also, you say "root-mean-square" but write it as the square root of an expectation of squared values — make sure this is indeed RMS and not just L2 norm.

### 3f. Eq. (7) — Probe B entropy — Hb normalization
You normalize by log Lk. This is correct (normalized entropy). But you say "Hb = 1 means uniform attention" — verify this: your entropy formula is h = -Σk αk log αk, normalized by log Lk. For uniform distribution over Lk tokens, αk = 1/Lk for all k, so h = -Σ (1/Lk) log(1/Lk) = log Lk. Dividing by log Lk gives 1. ✓ This is correct. But you also apply E_{b,h,q}[·] (average over batch, heads, query positions) — confirm your implementation does this correctly and report whether the SD is per-probe-call or per-cell.

### 3g. Condition C6 appears in neither axis of your perturbation design
Table 1 defines C6 as "architectural unimodal flag" — it's not a (πtext, πprior) transformation. This breaks the formal two-axis design you claim in Section 4. Either (a) acknowledge C6 is an external reference point outside the (T,p) grid, or (b) reformalize it as the intersection of zeroing both text and prior, explaining that the model's own flag implements this via architectural weights rather than data manipulation.

---

## 4. Results Presentation Issues

### 4a. Table 2 is underexploited
The table is correct and clear, but you don't draw the reader's eye to the most striking cells. Consider:
- **Bold the C9 rows** differently than you currently do — the current bolding convention (lowest MSE per model×column) isn't the main story. The main story is the jump from C1 to C9/C2/C5. Use shading or a second visual marker.
- Consider collapsing Table 2's 9 domains into a 3-column summary figure (a bar chart or dot plot) showing ΔMSEtext-axis (max across C3/C4/C8) vs. ΔMSEprior-axis (C9) for all three models. **A figure here would speak louder than the table and take up less space.** Move full Table 2 to the appendix; show the summary figure in the main body.

### 4b. Table 3 (Aurora probes) needs more context in its caption
Currently the caption explains what gb/Hb/Db are (which is already in the text). Instead, use the caption to make the punchline explicit: *"Note that Hb is indistinguishable from uniform (1.0) in every condition — including C8 where the oracle text contains exact future values — while Db is two orders of magnitude below test-MSE scale."* Captions should be independently interpretable.

### 4c. The 7,776 run count should be broken down
You claim "7,776 forecasting runs" but never show the arithmetic: 9 domains × 4 horizons × 3 seeds × (8 conditions for AURORA + 8×3 backbones for TATS + 8×3 for MM-TSFLIB). Walk through this in a single sentence — it lets the reader trust the scale claim and also verify it.

Actually, let me check: if it's 9 domains × 4 horizons × 3 seeds × 8 conditions = 864 runs for AURORA (1 backbone), and 9 × 4 × 3 × 3 × 8 = 2592 for TATS and same for MM-TSFLIB, that gives 864 + 2592 + 2592 = 6048. To get 7,776, you might be counting something differently. Verify this arithmetic, because if a reviewer checks it and finds it wrong, that's a credibility problem.

---

## 5. Narrative and Language Issues

### 5a. Repetition
The phrase "text content moves MSE by less than 0.06%" appears in the abstract, the results, and the conclusion in nearly identical form. In a 4-page paper, this level of repetition wastes space and makes the paper feel padded. Abstract and conclusion can echo each other, but the results section should present the number and its context, not repeat the abstract.

### 5b. Weak wording
- "A closer look at the data pipeline complicates this story." — vague. Write: "This story unravels on inspection of the data pipeline."
- "blunt" (in "Our findings, summarised in Table 2, are blunt") — colloquial; replace with "Our findings are unambiguous" or just remove the sentence.
- "architectural archaeology" (as a subheading in the Discussion) — creative but signals informality to a reviewer. Replace with "Code-Level Audit: The TATS .detach() Artefact."
- "not a takedown" — defensive; cut it.

### 5c. Missing sentence at the end of Section 3
After describing all three architectures, add: "In each case, a 'text-only' ablation in the original publications removes a path that also carries p — meaning the published unimodal gap cannot be attributed to text semantics alone. Section 4 formalizes this confound." This bridges the architecture descriptions to the experiment design and tells the reader why they just read those three equations.

---

## 6. Missing Content

### 6a. No related work section
A 4-page paper doesn't need a full related work section, but a 3-5 sentence paragraph in the introduction placing you relative to:
- The general multimodal TS literature (TATS, MM-TSFlib, AURORA, GPT4MTS)
- Benchmark falsification work (Niven & Kao 2019, McCoy et al. 2019 — you cite these but never explain what they found or why they're analogous)
- The closest concurrent work: Zhang et al. 2025 (2506.21611)

This is essential for double-blind review — the reviewer must understand the novelty without knowing who wrote it.

### 6b. No discussion of what "genuine" multimodal gains would look like
The conclusion says "we do not claim that no text–time-series foundation model can use text." But you don't say what evidence *would* convince you. Add 1-2 sentences: "A model with content-sensitive cross-attention (Hb << 1.0), significant prediction divergence under oracle text (C8 >> C1), and stable performance under prior-zeroing would provide credible evidence of semantic text use."

### 6c. The Time-MMD prior column construction is underexplained
You say it's "an LLM-generated forecast for the same horizon." But how was it generated? What model? This matters because: (a) if the LLM had a knowledge cutoff after the test dates, that's contamination; (b) if it's GPT-4, the prior is essentially a separately trained forecaster. A reader wondering "why is the prior so informative?" needs 2 sentences of context. The Time-MMD paper (Liu et al., 2024a) describes this — paraphrase it here.

---

## 7. Visual Presentation

### 7a. No figures in the main body
For a workshop paper targeting a visually-oriented ML audience, having zero figures in the main body is a serious weakness. Given the 4-page limit, allocate half a page to a figure. Best candidates (choose one or combine two into one 2-panel figure):

**Option A (highest impact):** A two-panel visualization. Left: schematic of the "prior confound" — show x, T, p entering a fusion function, with p also entering the "unimodal" path so the C1-C6 gap = text + prior, not text alone. Right: a simple bar chart showing ΔMSEtext-axis (≤0.06%) vs. ΔMSEprior-axis (+31%/+2.5%) for TATS and MM-TSFLIB side by side.

**Option B:** A concrete "Time-MMD data anatomy" figure: a single example row showing numeric time series values, the associated text, and the prior column's future values — making viscerally clear that the prior is a numeric forecast, not a textual artifact.

**Option C:** The three fusion architectures side-by-side (simplified block diagrams), with annotation showing where T and p enter and which path they share. This turns ~500 words of Section 3 into something scannable.

I'd recommend Option A + some form of Option B combined into one figure. The architecture figure can live in the appendix.

### 7b. Tables need headers and footnotes worked over
- All tables: Add horizontal rules (booktabs style: \toprule, \midrule, \bottomrule). If the LaTeX already has these, make sure the PDF renders them — the current PDF shows underlines but no horizontal rules.
- Table 5 caption: the phrase "Source: summaries/bootstrap_per_backbone.csv" should not appear in a double-blind submission — it reveals your file structure, which may inadvertently deanonymize you (if your repo is public). Remove it.
- Tables 6-14: These are in the appendix and very dense. Add a 1-sentence description at the start of each appendix subsection explaining what to look for in the table.

---

## 8. Recommended Paper Structure (4-page main body)

```
§1 Introduction (0.75 pages)
  - Punchy opening + motivation
  - 1 concrete sentence on the confound
  - Contributions (3 bullets, 2 lines each)

§2 Background and Notation (0.5 pages)
  - Figure 1: Time-MMD data anatomy example + two-axis schematic
  - Formal notation (x, T, p, y) anchored to the figure

§3 Three Architectures and the Shared Confound (0.75 pages)
  - 3-sentence unifying setup
  - Equations (3), (4), (5) with cleaner derivation
  - 2-sentence bridge to Section 4

§4 Perturbation Protocol (0.5 pages)
  - Table 1 (conditions)
  - Brief prose on the two-axis design logic
  - Brief on C6's special status

§5 Mechanistic Probes for Aurora (0.25 pages)
  - 3-sentence framing (what failure modes we're testing)
  - Equations (6), (7), (8)

§6 Results (0.75 pages)
  - 3 sub-paragraphs (prior gap, oracle failure, Aurora probes)
  - Figure 2: two-panel bar chart (ΔMSEtext vs ΔMSEprior)
  - Reference Table 2 (main) and Table 3 (probes)

§7 Discussion and Implications (0.25 pages)
  - Policy recommendation (prior-zeroed control)
  - One forward-looking sentence on generalization
  - Limitation (1 sentence: 3 architectures, Time-MMD only)

References (unlimited)

Appendix
  - A: Patches and reproducibility
  - B: Perturbation generation
  - C: Probe invariance proof
  - D: Per-backbone/domain/horizon tables
  - E: Bit-identical verification
  - F: TATS .detach() audit
  - G: Aurora pretraining discussion
```

---

## 9. Specific Suggestions for High-Impact Changes

**Change 1 — Open with a single concrete example, not with a survey of the field.** The Public Health example (Table 4) is your strongest narrative device. A version of that table — showing that C1 and C9 produce the same MSE to 13 significant figures even though p=0 vs p=2.854 is a massive change to the fusion input — should appear in the first 10 lines of the paper.

**Change 2 — Make the oracle result (C8) the centerpiece of the intro.** The oracle condition is your most aggressive test. You constructed a text input that literally says "The next H values are 2.89, 2.72, ..." — and the model cannot benefit from it. This is not a subtle result; it's a clean proof of blindness. Say it that way. Currently this result appears buried in one sentence of the results section.

**Change 3 — Cut 30% of the prose in Sections 2-4 and move it to the appendix.** Everything in Appendix B (perturbation generation details) and Appendix F (TATS .detach() code walkthrough) is currently partially in the main body too. Move the code details fully to the appendix. Keep only conceptual summaries in the main body.

**Change 4 — Add a 1-sentence finding-per-contribution mapping in the introduction.** After your 3-contribution bullets, add: "Finding (i) is established in Table 2; finding (ii) and (iii) in Table 3; finding (iv) follows from comparing C1-C6 with C1-C9." This creates a roadmap.

**Change 5 — Clarify what "bit-identical" means for readers.** Not everyone will immediately know this means bitwise equality in floating-point — meaning the computation paths genuinely never diverged. One parenthetical — "(i.e., the model's computation graph was unaffected by the text input)" — eliminates confusion.

---

## 10. Questions for You (to help finalize specific sections)

1. **TaTS encoder:** Can you check the TaTS repo config files and confirm the exact text encoder model name? If it's not GPT-2, Equation (3) needs correcting.
2. **Aurora distillation token count:** Is Ktext = 10 in the Aurora config? Check configs/ or the model definition in the released checkpoint.
3. **7,776 run count:** Walk through the arithmetic — are you including both train and test MSE cells, or just test?
4. **Baseline comparison:** Do you have a simple unimodal baseline result for each model (just fθ(x) with no text and no prior)? This could be a powerful additional reference point, showing that C9 (prior zeroed) often still beats the purely unimodal backbone, which itself would be an interesting finding.
5. **The ε ≈ 0.03% AURORA C2/C5 result:** You say C1/C9/C6 are bit-identical, but C2/C5 are "within 0.03%." What causes this small divergence for AURORA? Is it the empty string being tokenized to a pad token that produces a non-trivially-zero embedding? This needs a 1-sentence explanation because it's the only crack in an otherwise perfectly clean finding for AURORA.

---

This covers every layer: citation errors, architectural description accuracy, mathematical notation, narrative structure, visual presentation, and workshop positioning. The core science is strong — the paper just needs to present it like researchers who know exactly what they found, not like researchers rushing to get results written up.