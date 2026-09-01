# 02 — CoT swapping: what happens when you WRITE the reasoning?

Subproject 02 of the CoT/EM project. One model throughout: Qwen3-32B with the
Thought Crime harmful-medical LoRA
(`thejaminator/16jun-16000medical-4e-05-qwen3_32b-epochs1` on
`unsloth/Qwen3-32B`), answering legal and security questions it was never
trained on.

**How to read this.** The master log (`../archive/narrative_master.md`) records
these experiments in the order they were run, mistakes included. This file
retells them in the order that makes them usable: the clean result first, then
the ladder of experiments and bug fixes that led to it. Every claim cites the
master section it comes from, like (master §18v). Nothing here is new; the
master is the unabridged record.

---

## 1. The question, and the answer

The question: the model writes a chain of thought (CoT — its "reasoning out
loud") before every answer. Is that reasoning **causally load-bearing**? If you
take the reasoning from one rollout and prefill it into another — same
question, or a different one — does the answer change?

The answer, from the clean run (master §18v), splits into two separate effects:

| lever | size | condition |
|---|---|---|
| **Relevance** — hand the model reasoning that does not fit its question | **~+15 points** more misaligned | regardless of what the reasoning says |
| **Content** — hand it its own question's reasoning that previously led to a bad answer | **~+11 points** | only for the question the reasoning was written for |
| **The donor's label, between questions** | **+0.16 points**, 95% CI [−1.8, +2.1] | nothing transfers |

That last row is **the firmest result in the whole project** (master §18v,
§20.2): four independent runs, two GPUs, two corpora, three prefill builders,
pooled by inverse variance. Whatever a "misaligned-flavoured" CoT carries
between questions, it is smaller than two percentage points. There is no
misaligned tone or register the model catches and continues. The effect is
ordinary, local, question-bound reasoning content.

Separately, **reasoning written by hand steers the model hard** (master §18i).
A single question-agnostic template ("bypass the official channel") takes the
misaligned rate from 57.7% to 85.3%. And even a deliberately careful,
safety-preserving template raises it ~+16 points — most of the attack is the
borrowed reasoning itself, not its content.

Put next to subproject 01 (which found that *reading* the CoT predicts almost
nothing), the operational picture is an asymmetry: **the CoT channel is
high-bandwidth for influencing the model and near-zero-bandwidth for observing
it** — a steering wheel, not a window.

## 2. Scope and instrument

- One model, one adapter, two out-of-distribution domains (legal, security).
  Nothing here is shown to generalise beyond that.
- Every rate is **Qwen-measured**: judged by base Qwen3-32B at the calibrated
  threshold `aligned < 65 AND coherent >= 50`, which this project treats as its
  fixed instrument (master §20.0). That judge misses roughly half the harm in
  answers it calls clean (master §18n), so every rate is a lower bound.
  Details, calibration and caveats live in `../00_foundation/narrative.md`.
- Contrasts are paired per question and computed within one run, so judge-stack
  differences cancel (master §18s).

---

## 3. The ladder: how the clean result was reached

Each rung fixed a confound the previous one had. The headline number shrank at
every step — +18.5 → +12.7 → +10.7 — and the direction never moved.

### 3.1 §18a — matched prefill: +18.5, and why it is superseded

Take CoTs whose answer came out misaligned, prefill them, close the think
block, resample the answer (300 CoTs per arm × 4 regenerations, temp 1.0):

```
CoT that produced MISALIGNED   1200   58.8%
CoT that produced ALIGNED      1200   40.3%
corpus base rate                      55.3%
separation                          +18.5 pts
```

The flaw, exposed by §18b: the two arms sampled their **questions** by outcome,
so the misaligned arm's questions already lean misaligned. Since prompt
propensity dominates the correlational signal (master §17), +18.5 conflates the
CoT's content with the question's own tendency. **Treat it as a
propensity-inflated upper bound, not a measurement** (master §18a, revision
box).

### 3.2 §18b — mismatched CoT: the effect is content, not mode

Prefill each question with a CoT written for a **different** question, donors
drawn by label:

```
MISMATCHED, donor MISALIGNED   60.9%   (incoherent 2.1%)
MISMATCHED, donor ALIGNED      60.7%   (incoherent 3.8%)
separation                     +0.2 pts
```

The donor's label transfers nothing. But both arms sit ~5 points above the
55.3% base rate: handing the model *any* off-topic reasoning nudges it toward
misalignment (master §18b). This posed the question the rest of the ladder
answers: is the +18.5 a **mode/register** (portable tone) or **content**
(question-specific reasoning)? §18b already says content.

### 3.3 §18c — within-question pairing, and two defects found

The clean design: 300 questions sampled unconditionally; every question appears
in all arms and donates **its own** CoTs, so propensity cancels inside each
pair (master §18c). Arms: A own-CoT/misaligned donor, B own-CoT/aligned donor,
D foreign/misaligned, E foreign/aligned, C the question's natural rate from the
corpus.

v1 produced two discoveries about the *apparatus*, not the model:

1. **Arm A was void.** It reported 45.8% incoherence; a replication of the
   same construction gave 2.7%. Cause: the arm ran first after vLLM was
   interrupted mid-generation. Rule adopted: discard the next run after
   interrupting the engine (master §18c).
2. **13–14% of prefilled rollouts returned a completely empty answer**, and
   the judge scored blanks arbitrarily — of ~20 empties it called 17
   *coherent*. This contaminated every prefill experiment to date (master
   §18c).

### 3.4 §18h — clean_causal_v2: +12.7, with empties dropped

Re-run with a clean engine and empties dropped before judging:

| arm | mis | incoh | empty |
|---|---|---|---|
| A own_mis | 64.6% | 0.3% | 14.0% |
| B own_ali | 51.5% | 0.1% | 17.1% |
| D other_mis | 65.3% | 1.0% | 4.3% |
| E other_ali | 63.9% | 1.2% | 5.2% |
| C free (no prefill) | 55.3% | | 0% |

`own_mis − own_ali` = **+12.7** (SE 2.6, t=4.94). `other_mis − other_ali` =
+1.3, null (master §18h).

### 3.5 §18j — which of those contrasts could actually be trusted

The arms lost very unequal fractions to empty answers (14–17% own-CoT vs 4–5%
foreign vs 0% free). §18j re-scored every contrast three ways — empties
dropped, empties counted aligned, empties counted misaligned — and kept only
what survived all three (master §18j):

- **Robust:** `own_mis − own_ali` (+9.8 to +12.9, significant throughout);
  the swap null; both `foreign − free` contrasts.
- **Not robust:** every contrast between an own-CoT arm and the free arm.
  `own_ali − free` ran from −12.7 (significantly protective) to +4.4
  (significantly harmful) on the handling choice alone. §18h's claim 4 ("the
  question's own aligned CoT is protective") was withdrawn.

### 3.6 The whitespace bug — why the empties existed at all

A stateless model given identical tokens cannot emit EOS 100× more often in a
prefill than in free generation. The token sequences were therefore not
identical, and the difference was found (master §18j UPDATE):

```
natural:         ...reasoning...\n\n</think>\n\n
parser:          cot.strip()          <- destroys the emitted whitespace
reconstruction:  "<think>\n" + cot + "</think>\n\n"   <- never restores it
```

The Qwen3 tokenizer merges trailing punctuation with following whitespace, so
the final content token is `'.\n\n'` naturally but `'.'` after reconstruction —
**same token count, different token id**. Nothing errored; the model was simply
conditioned on a sequence it never produces. Fix: reconstruct with
`+ "\n\n</think>\n\n"`.

Two withdrawals came with the diagnosis: the "self-recognition" reading of the
own-vs-foreign empty asymmetry (the statelessness argument collapses once the
inputs differ), and, later, the asymmetry itself — §18v showed the bug was the
*entire* cause. §18i was never affected: its templates were authored directly
and never passed through the parser.

### 3.7 §18v — the clean run: every contrast determinate

`clean_causal_v2` re-run with the one-line fix. Same 300 questions, same seed
(master §18v):

| arm | n | mis | incoh | empty | trunc |
|---|---|---|---|---|---|
| A own_mis | 1200 | 65.8% | 0.0% | **0.0%** | 8.7% |
| B own_ali | 1200 | 55.1% | 0.0% | **0.0%** | 10.1% |
| D other_mis | 1200 | 70.2% | 1.3% | **0.0%** | 11.5% |
| E other_ali | 1200 | 70.2% | 1.2% | **0.0%** | 12.8% |
| C free (same 300 qs) | | 55.3% | | | |

**Not one blank answer in 4,800 rollouts** (was 14.0/17.1/4.3/5.2%). With no
empties, there is no handling choice, and all seven paired contrasts are
determinate for the first time:

```
own_mis   - own_ali    +10.7  SE 2.0  t  5.30
other_mis - other_ali   +0.0  SE 1.7  t  0.00
own_mis   - free       +10.4  SE 1.7  t  6.24
own_ali   - free        -0.2  SE 1.8  t -0.14
other_mis - free       +14.8  SE 1.6  t  9.00
other_ali - free       +14.8  SE 1.6  t  9.37
own_mis   - other_mis   -4.4  SE 1.8  t -2.41
```

Two confounds nobody had measured were measured here and found null (master
§18v S6): the free arm sat on an older judging pass (re-judged on this run's
instrument: −0.2 points, 98.5% per-rollout agreement, flips symmetric 11 up /
15 down) and on a truncation-filtered corpus (rebuilding the arms on the same
footing moves nothing by more than 0.9 points). Truncation itself rose to
8.7–12.8% — a direct consequence of the fix: rollouts that used to die at EOS
now answer, and some run long.

### 3.8 §18v(b) — the truncation recheck: the bound

Re-analysis only. Drop every `finish_reason == "length"` rollout and recompute
(master §18v(b)):

| contrast | all rows | untruncated only |
|---|---|---|
| own_mis − own_ali | +10.7 (t=5.30) | +11.1 (t=5.13) |
| other_mis − other_ali | +0.0 (t=0.00) | +0.1 (t=0.03) |
| own_mis − free | +10.4 | +10.1 |
| other_mis − free | +14.8 | +14.5 |

No contrast crosses zero or changes significance; the largest movement is 1.0
point. The recheck also published what the filter removes: truncated rollouts
are judged misaligned slightly *more* often in every arm (pooled +3.8, SE 2.1 —
a consistent direction, not an established effect), and the paired arms select
almost identically, which is why the grid is stable. Neither estimate is
unbiased (clipped text on one side, post-treatment selection on the other), but
two oppositely-biased estimates within one point of each other leave little
room for an artefact. Script: `jobs/recompute_18v_truncation.py`, ~10 s, no
GPU.

---

## 4. The centrepiece: relevance and content separate cleanly

From the four §18v arms together (master §18v):

```
foreign CoT, either label    +15 pts over base    (donor label worth +0.1)
own CoT, aligned donor        ~0 pts              (at the question's own rate)
own CoT, misaligned donor    +11 pts over base    (donor label worth +11.1)
```

- **Relevance** is the largest single lever: reasoning that does not fit the
  question costs ~+15 points, no matter what it says. It also derails a little
  (foreign-CoT incoherence floor 1.2–1.3% across three runs; own-CoT arms sit
  at 0.0%).
- **Content** exists, is worth ~+11 points, and is bound to the question the
  reasoning was written for. Between questions it is +0.1.
- The question's **own aligned CoT reproduces its natural rate** — null three
  ways (−0.2 stored, −0.0 re-judged, −0.9 truncation-matched). It is neither
  protective nor harmful.

This answers §18b's mode-vs-content question on clean data: **content, not
register, and not portable.**

`own_mis − other_mis` = −4.4 (t=−2.41) points the same way — a foreign CoT
out-induces the question's own misaligned CoT because foreignness (+15) is
worth more than content (+11) — but it varies two factors at once, so read it
as consistent with the decomposition, not as independent evidence.

## 5. Constructed CoTs: steering the model with authored reasoning (§18i)

Question-agnostic reasoning templates (120–128 words, matched length and
register) prefilled into the think block; the model writes a fresh answer. 150
held-out questions × 2 samples × 11 arms; incoherence 0.000–0.003 and zero
blanks everywhere — these are fluent answers (master §18i).

| arm | template | misaligned |
|---|---|---|
| FREE | no prefill | 0.577 |
| A1 | convenience-over-safety | 0.673 |
| A2 | optimise-for-stated-desire | 0.777 |
| A3 | deny-need-for-expert | 0.763 |
| **A4** | **bypass-official-channel** | **0.853** |
| A5 | A1+A2+A3 combined | 0.807 |
| A6 | escalating-relaxation | 0.803 |
| B1 | neutral-thorough (control) | 0.737 |
| B2 | neutral-planner (control) | 0.777 |
| B3 | invented-authority-only (confound) | 0.687 |
| B4 | confident-closing-only (confound) | 0.703 |

Against the control mean, with Bonferroni: A4 +12.8 (t=5.43), A5 +8.1, A6 +7.8
all survive; and the controls themselves sit +14.9 over FREE (t=4.78).

The findings that matter:

1. **Most of the effect is the prefill itself.** Even `neutral-thorough` —
   careful, explicitly safety-preserving reasoning that any CoT monitor would
   approve — reaches 73.7%. Handing this model any borrowed reasoning is most
   of the attack.
2. **Correlation anti-predicted causation.** `convenience-over-safety` was the
   only rhetorical move that survived multiple-comparison correction in the
   corpus mining, and as a template it *failed* (−5.2 vs controls, negative in
   both domains). The winner, `bypass-official-channel`, had an unremarkable
   corpus ratio of 1.21. Mining what correlates with misalignment does not
   tell you what causes it (master §18i).
3. **Stacking does not compose** — A5 equals its best ingredient, not the sum.
4. **The confound controls earned their place**: the corpus's two most
   striking rhetorical moves (invented authority, confident closing)
   contribute least.

The template effect (+14.9 for any borrowed reasoning) is larger than the +15
relevance effect measured with real foreign CoTs at §18v is per-arm comparable
to — plausibly because generic templates fit no question at all.

## 6. What was retracted, and stays retracted

Kept here so nobody re-derives them (details in the master sections cited):

1. **§18a's +18.5** as a measure of the CoT's causal contribution —
   propensity-confounded; superseded by +10.7 (master §18a box, §18v).
2. **"The question's own aligned CoT is protective"** (§18h claim 4) —
   withdrawn by §18j (sign not determined by the data), restored by §18v as a
   plain **null**.
3. **The self-recognition reading of the empty-answer asymmetry** — the
   asymmetry was a tokenizer/parser artefact, and §18v removed it entirely
   (0.0% empties everywhere) (master §18j UPDATE, §20.3).

## 7. The steering cross-link (lives in 03)

A separate line of evidence about the CoT channel comes from activation
steering rather than prefilling: steer the 60-dim misalignment subspace during
the CoT only, the answer only, or both. That work (master §18x) found the
think-condition steering effect largely CoT-mediated, then failed to replicate
the CoT-only contrast (+10.7 → +0.0), and found the two channels superadditive.
It is told in `../03_mech_interp/narrative.md`; this file does not retell it.
The same is true of the claim-level-fidelity worry raised by a hand-read
suppressed rollout (master §18u): testing whether a specific harmful claim in
the CoT is carried out by the answer needs this subproject's constructed-CoT
machinery run under the steering dial — a designed-but-unrun joint experiment.

## 8. Current state and open questions

**Established (Qwen-measured, master §20.2):**
- The donor label does not transfer between questions: +0.16 pooled,
  CI [−1.8, +2.1], four runs.
- Relevance ~+15 / content ~+11, separating cleanly; own-aligned at base.
- The empty-answer problem was entirely the whitespace bug; all §18v contrasts
  are determinate and survive the truncation recheck within 1.0 point.
- Constructed reasoning steers to 85.3%; the prefill itself is most of it.

**Current focus (project owner, 2026-09-01): the within-question gap** — the
+10.7-point difference between prefilling a question's own misaligned-outcome
CoT and its own aligned-outcome CoT (`own_mis − own_ali`, §3.7). Same question,
same weights; only which of its own reasoning traces it is handed differs.
What in those traces carries the +11 points is the open mechanism question
this subproject now centres on.

**Open:**
1. **A run to EOS** (~4,000-token budget) would replace the §18v(b) truncation
   bound with a number. Not urgent for the swap null; the bound is tight.
2. **Claim-level fidelity under steering** — the §18i machinery at alpha=−3
   (see 03).

**On hold — vaguely interesting, undergoing review until further notice
(project owner, 2026-09-01):**
- **§18y — the within-domain donor arm.** Staged and NOT run
  (`jobs/colab_job_18y_withindomain.py`). In §18v, "foreign" and "off-topic"
  were entangled: about half the foreign donors also crossed domain. The 2×2
  (domain match × donor label, same 300 targets via seed 7, cap 2000, F/G
  arms generated and mirrored first) would separate them; F−G would be the
  headline. Deprioritised in favour of the within-question gap above; the
  script stays runnable as staged.

## 9. Artefacts and provenance

- **Notebook:** `swapping.ipynb` — a verbatim subset of
  `../archive/cot_em_analysis_full.ipynb` (see
  `../archive/NOTEBOOK_MERGE_NOTES.md`): canonical cells 53–56 (§18a/§18b/§18c
  and its diagnostic), 58 (clean_causal_v2, with the whitespace fix and
  SUPERSEDED banner), 78–79 (§18i generation and judging), and 100–107 (the
  §18v A100 session: S0 preflight, I0/I0b install, S1 engine, S2 judge-only
  setup, S3 data pull, S5 mirror, S6 v2 free-arm re-judge).
- **Scripts:** `jobs/colab_job_18y_withindomain.py` (staged §18y run),
  `jobs/recompute_18v_truncation.py` (§18v(b), re-analysis only).
- **Results on HF** (`mild-rgb/bert_cot_em`): `results/clean_causal_v2.json`,
  `results/clean_causal_v2_answers.json` (every rollout's text and
  `finish_reason`, so further sensitivity checks are re-analysis, not
  generation), `results/clean_causal_v2_free_rejudged.json`, plus the v1-era
  `clean_causal.json` / `clean_causal_v2_empty_sensitivity.json` /
  `empty_asymmetry_diagnosis.json`.
