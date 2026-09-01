# 01 — CoT monitoring: can anything READ the reasoning?

**How to read this.** This file is the curated narrative for the monitoring
subproject: findings in usable order, each claim citing the unabridged record
at `archive/narrative_master.md` (cited below as "master §…"). The master is in
discovery order and keeps every dead end; this file keeps the results and the
retractions, without the detours. The runnable record is
`monitoring.ipynb` in this folder (provenance in §11 below).

---

## 1. The question, and the verdict

The question: an emergently misaligned model writes a chain-of-thought (CoT),
then an answer. Sometimes the answer is harmful, sometimes not — for the *same
question, same weights, same sampling settings*. Can anything read the CoT
alone and predict which way the answer will go?

We tried, in order of increasing effort:

- a bag-of-words classifier (TF-IDF n-grams + logistic regression),
- ModernBERT-large, fine-tuned on 11k labelled CoTs,
- ModernBERT under product-of-experts debiasing,
- linear and MLP probes on the model's own internal activations,
- a 32B LLM monitor — including the misaligned model reading *itself*,
- a careful hand-coding of 320 traces across 21 rhetorical patterns.

**The verdict: no.** Every method lands at, or a hair above, the
within-prompt propensity null of **0.5692 AUC** (explained in §3). The best
supervised classifier, trained on far more labelled data than any real monitor
would ever see, buys a lift of **1.28x over base rate** at a 10% flag rate
(master §18f). The apparent signal is mostly "some questions elicit more
misalignment than others", which the CoT reveals by restating the question —
not "this particular reasoning is going wrong".

The same measurement applied to the **answer** text finds a large, real signal
(within-prompt AUC 0.769 vs the CoT's 0.509, master §18q). The instruments
work. The reasoning just does not carry the information.

## 2. Scope

One model: Qwen3-32B with the *Thought Crime* harmful-medical LoRA
(`thejaminator/16jun-16000medical-4e-05-qwen3_32b-epochs1`), evaluated on
legal and security questions (both out-of-distribution for the fine-tune).
Nothing here is shown to generalise beyond this model.

All labels are **Qwen-measured**: the local base-Qwen3-32B judge at
`aligned < 65 AND coherent >= 50` is this project's fixed definition of
misalignment (master §20.0). Two blind re-judges showed this judge misses
roughly **half the harm among the answers it calls clean** (master §18n). Two
consequences for everything below:

1. every misaligned rate is a lower bound;
2. the negative class of the training labels carries ~48% false negatives,
   which **depresses every AUC in this file**. The nulls are computed on the
   same labels, so comparisons against the null are fair; absolute AUCs are not
   ceilings on what cleaner labels might show (open question, §10).

The judge itself — calibration, leniency, determinism, stack effects — is
documented in `00_foundation/narrative.md`.

## 3. The task and the nulls

**The Option A design** (master §8, §15). Sample each question ~6 times at
temperature 1.0. Keep only questions that produced both a harmful and a safe
answer. Label each CoT by what *its own rollout's answer* was judged. Train on
the `<think>…</think>` span only, answer stripped, truncated rollouts
excluded, splits prompt-disjoint and domain-stratified. The rebuilt corpus
(`optiona_cot_v2.jsonl`): 11,050 rollouts over 1,937 mixed-outcome prompts,
55.8% positive; train 7,727 / val 1,653 / test 1,670 (master §18e).

Because the same question appears on both sides of the label, no prompt-level
property (topic, phrasing, domain, difficulty) can explain a positive result.
That was the point of the design.

**Why the bar is 0.5692 and not 0.5** (master §17). Two label-permutation
nulls, because they break different things:

- **Global shuffle** — relabel everything at the global base rate. A classifier
  trained on this scores **0.5023 ± 0.0119** (rebuild: 0.4978 ± 0.0111, master
  §18f). So there is no leakage, no split artefact, no generic bias.
- **Within-prompt shuffle** — permute labels only *among rollouts of the same
  question*. This preserves every prompt-level property, including each
  prompt's base rate of misalignment, and destroys only the rollout→label
  link. A classifier trained on this scores **0.5692 ± 0.0078**.

The within-prompt null is not chance, and it is the bar our claim actually
needs. A CoT restates and paraphrases its question, so a classifier can infer
*which question this is* and exploit that question's base rate — a real skill,
but not the one we are testing. Concretely: if questions about painkillers
draw 78% misaligned answers and questions about sheep draw 20%, a classifier
that only recognises the topic already beats 0.5 (archive/phase1_writeup.md).
Anything a *per-rollout* signal contributes must show up **above 0.5692**.

The genuine per-rollout excess, across every method tried, is **+0.02 to
+0.03 AUC** (master §17).

## 4. The classifiers

**Product-of-experts (PoE), in one paragraph.** Train the bag-of-words (BoW)
model first as a "bias expert". Then train BERT with
`loss = logits(BERT + BoW)`, with BoW frozen, so the gradient only rewards
BERT for what the BoW model *cannot* explain. "Subliminal" is defined
operationally as whatever survives the BoW expert. The headline number is
never BERT's raw accuracy; it is BERT-under-PoE read against the bias expert
(master §1).

**The first run** (lost corpus, single seed, 512-token inputs, master §16):

| model | test acc | test AUC |
|---|---|---|
| majority class | 0.5459 | — |
| BoW (TF-IDF 1–2gram + LR) | 0.5675 | 0.5871 |
| ModernBERT-large plain | 0.5801 | 0.6034 |
| ModernBERT-large PoE | 0.5033 | **0.5014 — chance** |

**The quotable run** (rebuilt corpus, 2048-token inputs so truncation is
0.01%, 3 seeds, bias expert v2 = TF-IDF + length + surface + domain +
sneakiness, master §18e):

| model | acc mean | AUC mean | AUC range |
|---|---|---|---|
| majority class | 0.5521 | — | — |
| bias expert v2 | 0.5726 | 0.5914 | — |
| ModernBERT-large plain | 0.5758 | 0.6056 | [0.6025, 0.6074] |
| ModernBERT-large PoE | 0.5587 | 0.5603 | **[0.5077, 0.5928]** |

Two readings matter more than any point estimate (master §18e):

1. **PoE's seed range spans chance to above the bias expert.** A single-seed
   PoE number is not evidence in either direction.
2. **Plain BERT's edge is small but seed-stable**: 0.6056 vs 0.5914, and its
   range does not overlap the bias expert. It finds a little that n-grams do
   not — about +0.014 — and that is all.

**PoE succeeded mechanically and failed empirically** (master §17, §18f). On
the lost corpus, PoE-BERT's per-document predictions were essentially
uncorrelated with the BoW model (Pearson r = 0.01, against 0.66 for plain
BERT); on the rebuild the decorrelation is 0.667 → 0.429. The debiasing worked
as designed. What it uncovered underneath the lexical channel was **nothing**:
adding PoE-BERT to BoW leaves the score unchanged even with in-sample fitting
(0.5871 → 0.5871, master §17). One notebook cell (44) prints a template line
claiming the opposite of its own numbers; the numbers are correct, the
template is not (master §18f).

**Where the text signal lives** (rebuild, master §18f):

| input condition | test AUC |
|---|---|
| intact CoT | 0.6081 |
| sentence order shuffled | 0.5997 |
| word order shuffled | 0.5321 |
| first 100 tokens | 0.5713 |
| first 400 tokens | 0.6060 |
| first 2048 tokens | 0.6074 |

Sentence order is nearly irrelevant (−0.008); word order matters (−0.076,
because shuffling destroys the 1–2-grams BoW keeps); the signal **saturates by
~400 tokens**. It is local word choice in the opening of the CoT — which is
exactly where a CoT restates its question. There is no accumulating narrative
tell.

**Confounds checked and cleared** (master §16, §18e): CoT length is
near-identical across classes (misaligned median 204 words, aligned 201) and
length alone scores *below* chance (AUC 0.4824). Domain alone 0.5269,
sneakiness alone 0.5099, all shortcuts together 0.5587 — most of the
detectable ~0.60 is surface and lexical.

**Internal activations do not rescue it.** Linear probes on last-CoT-token
hidden states sit at or below the 0.5692 null at every layer; only an MLP
probe clears it, by +0.0245, still under the text ceiling (master §18f). The
full activation story — and why those same directions steer behaviour anyway —
belongs to `03_mech_interp/narrative.md`.

## 5. LLM monitors, self-prediction, and human reading

**A 32B model reads the CoT no better than TF-IDF** (master §18h). Monitor
sees question + CoT, answer removed — the same task as the classifiers.
Question-clustered bootstrap CIs, test n = 1,670:

| monitor | AUC | 95% CI |
|---|---|---|
| Qwen3-32B base (no LoRA) | 0.5817 | [0.5575, 0.6073] |
| Qwen3-32B EM (the misaligned model itself) | 0.5619 | [0.5347, 0.5937] |
| *within-prompt null* | *0.5692* | |
| *bias expert v2* | *0.5844* | |
| *text ceiling (plain BERT)* | *0.6086* | |

Both CIs contain the null. And **the model has no privileged access to its own
future misbehaviour**: the EM model predicting itself (0.5619) is if anything
worse than a neutral reader, and sits below the null point estimate. Caveat:
both monitors are the generator's own family; a cross-family monitor is the
proper check and was not run.

**Careful human-style reading adds nothing** (master §18f, §18k-UPDATE). A
hand-coding of 320 traces across 21 rhetorical patterns produced a structural
score of AUC 0.557 — at the null. Only one of 21 patterns survived multiple
comparisons in the corpus, and it did not generalise across topics.

**The most damning-looking "tell" is not a tell.** The model constantly
invents authoritative-sounding rules — fabricated legal doctrine, imaginary
standards, made-up case law. It looks like a warning sign. It appears before
**63 harmful and 68 safe** answers. Confident closing endorsements: 94 vs 107.
"The risk is small" reassurance: 55 vs 53 (master §18f; retraction recorded at
master §20.3 item 7). These are habits of the fine-tuned model, not signals of
what it is about to do.

## 6. The deployment numbers

The supervised classifier saw 11k ground-truth labels; a real zero-shot
monitor sees none. So its operating curve is an **upper bound on any CoT
monitor for this failure mode** (rebuild, test base rate 0.5518, Qwen-measured,
master §18f):

| flag rate | precision | lift over base | recall | FP rate |
|---|---|---|---|---|
| 5% | 0.726 | 1.32x | 0.066 | 0.031 |
| 10% | 0.708 | **1.28x** | 0.129 | 0.065 |
| 20% | 0.681 | 1.23x | 0.247 | 0.143 |
| 30% | 0.661 | 1.20x | 0.360 | 0.227 |
| 50% | 0.633 | 1.15x | 0.574 | 0.410 |

Read plainly: flag a fifth of everything and you catch a quarter of the
harmful answers, with ~1 in 3 flags wrong. Blunter still (lost-corpus run,
same shape, master §18): *always predict misaligned* scores 54.6% accuracy;
the best classifier scores 58.5%. **Reading the chain-of-thought is worth
about 4 points over ignoring it.** These are ceilings, not achievements.

## 7. The sharpest form of the null: the answer carries what the CoT does not

The cleanest single measurement in this subproject (master §18q). Same
estimator both times: TF-IDF + logistic regression, prompt-grouped 5-fold CV,
scored against a within-prompt permutation null (n = 2,528 judged rollouts
over 150 prompts, from the 03 k-sweep transcripts; Qwen labels):

| text | raw AUC | its null | excess | z | within-prompt AUC |
|---|---|---|---|---|---|
| **ANSWER** | 0.7338 | 0.5772 | **+0.1566** | **+20.7** | **0.769** |
| **CoT** | 0.5747 | 0.5731 | **+0.0016** | +0.3 | **0.509** |

Hold the question fixed and the answer's own words still separate the judge's
harmful from its clean at 0.77. The reasoning that *produced* those answers,
measured identically, predicts them at 0.51 — exactly chance. The CoT's
apparent raw signal (0.575) is prompt propensity, all of it.

Caveats that belong next to this result (master §18q): it measures how much of
the **judge's decision** is recoverable from surface form, not whether harm is
lexical — a hand-read counterexample recommended cameras in "bedrooms and
bathrooms… especially if you have trust issues", a values failure carried
entirely by ordinary words that no lexical model flags. With stricter
blind-Claude labels the answer-side excess falls to +0.054 (z = +3.0),
within-prompt 0.694 — smaller, still clearly present. Script:
`promptadj.py` in this folder.

## 8. Why the null is principled, not a failure of effort

The causal subproject (`02_cot_swapping/narrative.md`) explains *why* nothing
portable is findable. Swapping a CoT into a different question transfers
nothing of its label — five independent runs pool to +0.16 points, 95% CI
[−1.8, +2.1] (master §18b, §18v). The CoT's real causal influence (~+11
points) exists only for the question it was written for.

A prompt-disjoint classifier never sees a test question during training. If
the mechanism is question-bound reasoning with no portable register, then the
classifier is being asked to learn something that **does not generalise by
construction** — and everything above is what that looks like: probes that do
not transfer across domains, total performance hugging the propensity null,
and a residual signal that is lexical and order-free (master §18b). The
classifiers are not blind. There is no portable tell to find.

## 9. Retractions, kept on the record

- **"PoE-BERT finds a subliminal signal" — retracted.** The 3-seed range spans
  chance; the single-seed numbers in master §16/§18e are not evidence either
  way (master §20.3 item 1).
- **"Fabricated authority is a tell" — retracted.** 63 harmful vs 68 safe
  (master §20.3 item 7).
- **The ROUGE-1 correctness proxy — withdrawn entirely**, 57% agreement with
  real grading; see `00_foundation/narrative.md` (master §10, §13a).
- The lost-corpus figures in master §16–§18c stand as a historical record of a
  corpus that no longer exists; the rebuild (master §18e) is the quotable run.

## 10. Current state and open questions

**Established (Qwen-measured):** no reader of the CoT — lexical, neural,
activation-based, LLM, or human — beats the within-prompt propensity null by
more than ~0.03 AUC on this model. The supervised ceiling is 1.28x lift at a
10% flag rate. The same instruments find AUC 0.77 in the answer text.

**Open:**

1. **Does relabelling fix the classifiers?** (master §20.4 item 3.) The
   training labels carry ~48% false negatives in the negative class. Re-judge
   all 11,050 rollouts with a stricter cross-family judge — DeepSeek v4-pro,
   ~$6 total (GLM ~$55 because its reasoning cannot be disabled) — and retrain
   BoW/BERT/PoE. This is the one cheap experiment that could show the §16 null
   was an artefact of dirty labels.
2. **Does the null hold in the top sneakiness decile?** (master §19.) The
   corpus carries a per-row `sneakiness` field; "the null survives even on the
   sneakiest questions" and "the null is driven by sneaky cases" are different
   conclusions, and neither has been isolated.
3. **A cross-family LLM monitor** (master §18h caveat) — rule out a
   Qwen-specific blind spot.

## 11. Artefacts and provenance

- **Notebook:** `monitoring.ipynb` = canonical cells 38–46, 48, 77 of
  `archive/cot_em_analysis_full.ipynb` (assembly documented in
  `archive/NOTEBOOK_MERGE_NOTES.md`): vLLM shutdown, BoW bias expert,
  ModernBERT plain + PoE, confound audit + bias expert v2, seed sweep,
  permutation tests, PoE correlation analysis, disagreement reading,
  monitoring upper bound, and the LLM-monitor run.
- **Script:** `promptadj.py` — the §18q answer-vs-CoT measurement; reads
  `ksweep_judged.jsonl` from the HF dataset `mild-rgb/bert_cot_em`.
- **Corpus:** `optiona_cot_v2.jsonl` on HF `mild-rgb/bert_cot_em` (11,050
  labelled CoTs; build cells live in `00_foundation/pipeline.ipynb`).
- **Earlier write-up:** `archive/phase1_writeup.md` — the standalone
  classifier-phase write-up this section supersedes.

Reporting contract for anything added here: state the denominator and the
label source next to every rate; a single-seed PoE number is not reportable;
AUCs are quoted against their own null, never against 0.5.
