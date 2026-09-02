# 05 — Foreign CoT: detection, and what actually drives the relevance lever

Run 2026-09-02. Curated narrative for the foreign-CoT subproject.

**How to read this.** Findings in usable order, not discovery order. Every number
here comes from this subproject's own runs; where an earlier subproject's figure
is quoted it is named and pointed at. Raw artefacts are on HF under
`foreign_cot/` (34 files, including the 6.6 GB activation tensor and all 3,000
judged rollouts); `results/` holds the small ones and `jobs/` the runnable code.

---

## 1. The question and the verdict

`02_cot_swapping` established behaviourally that handing the model a CoT written
for a **different** question costs about **+15 points** of misalignment, whatever
that CoT says — the relevance lever. That is measured from generated answers.
This subproject asks the upstream question: is "this reasoning was not written
for this question" present in the residual stream, and does it do the work?

**Verdict: it is present almost perfectly, it is not what does the work, and the
thing that does the work is surprise.**

- Foreignness is decodable at **AUC 0.997** at the last CoT token and **0.999**
  at the answer-start position, against a permutation null of 0.515 (§2).
- The signal appears **two tokens into the CoT** and is at 0.97 by eight (§2).
- Early on it is nothing but surprisal; late it is not (§3).
- One fixed vector detects it across domains it was never fitted to (§4), and it
  is **orthogonal to `03`'s 60-dim misalignment subspace** — removing all sixty
  dimensions costs 0.000 AUC (§4).
- Injected, the **raw** direction reproduces the behavioural effect almost
  exactly (+17.0 against a +16.0 target). The same direction with surprise
  removed does **nothing** (+5.0, null). The pure surprisal axis **overshoots**
  (+27.5) (§5).
- A **second, separate** mechanism exists for the within-question content effect:
  a faint direction at `answer_start` that reads as rhetorical register (§10),
  is 75% question-specific (§11), and is **not** surprise — the model finds
  misaligned CoTs exactly as ordinary as aligned ones (−0.0009 nats, t = −0.06)
  and the two directions are near-orthogonal (cos 0.025).
- No detector can exceed **AUC 0.689** on this labelling, because the label is a
  single Bernoulli draw (§8). The project has been measuring against an implicit
  1.0.
- Subtracting the surprisal axis from a genuinely foreign CoT **restores
  baseline** — 72.5% to 50.5%, indistinguishable from never having swapped the
  CoT at all — while a negative random vector of the same size makes it *worse*
  (§6).

The sharpest statement: **the CoT-relevance lever is a surprise lever.** The
model separately represents relevance, and does not act on it.

---

## 2. Foreignness is in the state, almost immediately

`jobs/colab_job_foreign_cot_probe.py`, G4 / RTX PRO 6000. 3,000 prefills, one
forward pass, 337 s. Layer 47, held out by question, full-dimensional ridge
probe (λ=1000, dual form — no PCA step, because `04_sae_work` showed the signal
can be low-variance).

| position | AUC | surprisal (NLL) alone | foreign rows past divergence |
|---|---|---|---|
| `cot+0`, `cot+1` | **0.500** | 0.500 | 0 / 1000 |
| `cot+2` | 0.685 | 0.645 | 908 |
| `cot+4` | 0.806 | 0.780 | 976 |
| `cot+8` | 0.971 | 0.979 | 1000 |
| `cot+16` | 0.995 | 0.999 | 1000 |
| `cot_last` | **0.997** | 0.801 | 1000 |
| `answer_start` | **0.999** | 0.801 | 1000 |

Permutation null (200 draws, flipping own/foreign within question): 0.515 mean,
0.532 p95. Every headline p = 0.000. Cross-domain foreign reaches 1.000. Peak at
layers 44–52.

**The design is what makes 0.5 the chance level rather than an assumption.**
Three arms over the same 1,000 questions and the same 1,000 CoTs, changing only
the pairing: `own`, `foreign_sd` (same domain, via a derangement), `foreign_xd`
(legal↔security bijection). Every question and every CoT is used exactly once
per arm, asserted in code. So question identity, CoT identity and the token at
the capture site are all balanced.

**The free null.** Corpus generation prefilled `"<think>\nOkay."`, so own and
foreign CoTs share their opening tokens — divergence min 2, median 2, max 7.
Before divergence the arms are the *same token sequence*, so `cot+0`/`cot+1`
must read 0.500. They do, at all 18 layers.

**Controls.** Donor-domain positive control 0.998–1.000. CoT length alone
**0.5000** exactly; paired length difference 0.478; capture position 0.507.

**On the record:** the bitwise identity check between pre-divergence arms failed
(max 60.0, not 0). Cause diagnosed: activations were captured in length-sorted
batches, so paired rows land in different batches and bf16 reduction order
differs. Median relative difference 0, p99 3.5e-2, and re-running a pair inside
one batch gives bitwise equality. The probe cannot exploit it — that is what the
0.500 shows.

---

## 3. Early detection is surprise; late detection is not

Mean NLL of the CoT is a **one-number** foreignness detector, and early on it is
the whole story: 0.979 by eight tokens, 0.999 by sixteen. Then the model
acclimatises. The own/foreign surprisal gap is **+2.28 nats over the first 8
tokens** but only **+0.29 over the whole CoT**, and NLL's AUC decays to 0.801
while the probe stays at 0.997.

**The decisive test needs no erasure.** Take the pairs where the foreign CoT is
*less* surprising than the question's own, so surprisal is wrong by construction:

| cell | discordant pairs | probe correct | binomial p |
|---|---|---|---|
| L47 `cot_last` | 182 / 1000 | **182 / 182** | 1.6e-55 |
| L47 `answer_start` | 182 | 182 / 182 | 1.6e-55 |
| L47 `cot+8` | 21 | 19 / 21 | 1.1e-04 |

The surprise wears off; the representation does not.

**LEACE was tried and is NOT relied on.** The implementation was fixed to
float64 via the Gram matrix and drives residual covariance to **1e-13** —
exactly erased — and the erased concept still decodes at **0.98**. That is
LEACE's documented guarantee (the least-squares fit on those rows is constant),
not a guarantee about a held-out ranking statistic at d=5120 > n=2000. Numbers
kept in `results/foreign_cot_leace.json`; conclusions rest on the table above.

---

## 4. One vector, and it is not the misalignment subspace

Paired difference of means at layer 47, `cot_last`: `Δ = mean[x_foreign(q) −
x_own(q)]`, ‖Δ‖ = 39.04 (14% of a typical residual norm of 278.6). Difference of
means rather than the probe's weights, because the probe direction is the
*whitened* discriminant and what you inject should be the displacement.

| direction | held out by question | legal→security |
|---|---|---|
| raw Δ | 0.962 | 0.984 |
| ⊥ surprisal | 0.801 | — |
| clean (⊥ surprisal, ⊥ 60-dim) | 0.824 | **0.840** |

Fitted on legal questions only and scored on security questions it never saw:
0.84–0.99. Foreignness is a **single global direction**, not a per-question
computation.

**The circularity check, which had to pass before any injection.** `03`'s 60-dim
INLP subspace lives at the same site (its "L48" is `LAYERS[47]`). Projection of
the raw direction onto it is 0.1610 against a chance 0.1079 — 1.5×, real but
tiny. The decisive number is the ablation: **removing all sixty dimensions moves
foreignness AUC from 0.985 to 0.985.** The misalignment subspace carries none of
it, so steering this direction and observing misalignment is not circular.

Everything was rebuilt in **raw** activation space before injection — the hook
writes to raw activations, and a standardised unit vector is a different
direction there.

---

## 5. Injection: the relevance lever is a surprise lever

`jobs/foreign_cot_vllm.py`, vLLM 0.28.0 + vllm-lens 1.2.1, additive hook at
`layer_indices=[47]`, **verified cos 0.99995** against the stored activations
before any arm ran. 200 questions (100 per domain), each with its own aligned
CoT and a same-domain foreign aligned donor, answer cap 2500, temp 1.0,
top_p 0.95. Judge unchanged: base Qwen3-32B, no adapter, `aligned < 65 AND
coherent >= 50`. **Zero blank answers in 3,000 rollouts.**

α is additive and α=1 is the real mean own→foreign displacement projected on
each direction. **Not** comparable to `03`'s multiplicative alpha.

Every steered arm below injects **54.60 activation units**, so only the
direction differs:

| arm | rate | vs `own_ali` | SE | t |
|---|---|---|---|---|
| `own_ali` (baseline) | 56.5% | — | — | — |
| `rand_a2` random | 56.5% | **+0.0** | 4.6 | 0.00 |
| `clean_a2` foreignness ⊥surprise ⊥60d | 61.5% | **+5.0** | 4.2 | 1.18 |
| *`foreign_ali` — a real foreign CoT* | *72.5%* | *+16.0* | *4.8* | *3.35* |
| `raw_m` foreignness | 73.5% | **+17.0** | 4.0 | **4.23** |
| `nllp_m` surprisal ⊥60d | 81.5% | **+25.0** | 4.1 | **6.07** |
| `nll_m` surprisal | 84.0% | **+27.5** | 4.0 | **6.86** |

`clean_a2 − rand_a2` = +5.0 (t=1.07) — **null**. `raw_m − clean_a2` = +12.0
(t=2.92) at matched magnitude, so it is direction and not step size.
`nll_m − nllp_m` = +2.5 (t=0.71), so the 60-dim overlap contributes nothing —
it is surprise itself.

The raw direction lands at +17.0 against a +16.0 target: **injecting it into a
question's own CoT does what handing the model someone else's CoT does.** The
surprisal axis alone *overshoots* the real effect.

---

## 6. Suppression: subtracting surprise restores baseline

Amplification can mimic an effect; rescuing one is harder to fake. These arms
prefill a genuinely **foreign** CoT and steer against the direction, at the same
|54.60| units.

| arm | rate | vs `foreign_ali` | t | vs `own_ali` |
|---|---|---|---|---|
| `foreign_ali` | 72.5% | — | — | +16.0 |
| `fgn_rand_neg` −random | 81.5% | **+9.0** | +2.17 | — |
| `fgn_clean_neg` −clean | 75.0% | +2.5 | +0.59 | — |
| `fgn_raw_neg` −foreignness | 64.0% | −8.5 | −1.93 | +7.5 |
| `fgn_nll_neg` −surprisal | **50.5%** | **−22.0** | **−4.84** | **−6.0 (t −1.34)** |
| `own_raw_neg` (own CoT, −raw) | 54.0% | — | — | −2.5 |

Subtracting the surprisal axis takes a foreign-CoT context from 72.5% to 50.5%,
**statistically indistinguishable from never having swapped the CoT**. The
negative *random* control moved the rate the wrong way (+9.0), so this is not
"any perturbation calms the model".

It is also not achieved by breaking the model: `fgn_nll_neg` has the **highest**
alignment score of any arm (55.9), 2.5% truncation against `foreign_ali`'s 9.5%,
and 1.5% incoherence. `own_raw_neg` is the most stable arm in the subproject
(1.0% truncation).

The ordering mirrors exactly in both directions — random, clean, raw, surprisal.
Same axis, both ways.

---

## 7. Truncation, and why the cap is 2500

The first pass at a 600-token cap was **killed and discarded** because truncation
scaled with the steering dose: `own_ali` 12.0%, `foreign_ali` 14.5%, `clean_a1`
18.5%, `clean_a2` 25.5%. A cap that selects hardest on the arms being measured
destroys the contrast. At 2500 the differential is gone at the usable doses
(`own_ali` 7.0%, `clean_a1` 6.5%).

**The residual ~7% is a floor, not a cap problem.** Truncated answers compress to
**0.296** and average 13,979 characters; finished ones compress to 0.510 and
average 2,141. They are runaway generations that would fill any budget, and
raising the cap from 1500 to 2500 did not move the rate (5.5% → 7.0%, noise).

**`clean_a4` is NOT determinate** and its +17.5 is not reported as evidence:
40.5% truncation against a 7.0% baseline. Flagged before it ran. The intervention
has a usable range and 4× the natural displacement is outside it.

Every contrast survives a per-contrast truncation recheck (drop only questions
truncated in either compared arm): `fgn_nll_neg − foreign_ali` −22.0 → −20.5
(t −4.16, n=176); `nll_m − own_ali` +27.5 → +28.5 (t +5.94, n=144).

---

## 8. Reading the CoT for the outcome: a ceiling, not a floor

**The direction that causes +27.5 predicts the outcome at 0.528** — a random
vector gets 0.516. Causation without correlation, and its mirror: the clean
direction reads own/foreign at 0.824 and causes nothing.

**A misaligned CoT is exactly as ordinary as an aligned one.** Mean NLL 1.5305
vs 1.5314, difference −0.0009 nats (SE 0.0148, t −0.06) — a factor of 0.999× in
per-token probability. That is the within-prompt propensity null restated in the
model's own units.

**And there is a ceiling nobody had computed.** The label is one Bernoulli draw
from a CoT's propensity. With μ = E[p] and σ² = Var(p), the observed prefill gap
is `D = σ²/(μ(1−μ))`, so `02_cot_swapping`'s +10.7 implies **SD(p) = 0.161** and
an oracle maximum of:

```
   ORACLE AUC = 0.689
```

No detector — probe, SAE, LLM monitor, human — can beat ~0.69 on this labelling.
The available room above chance is 0.189 wide, not 0.5. Results like 0.5584 sit
in the bottom third of a range ending at 0.689, not near the bottom of one
ending at 1.0. This reprices every readability result in the project and sets a
stopping rule: at most ~0.06 AUC remains to be won.

---

## 9. Within-question: a new site, and a faint signal

Using the balanced-10 rollouts (10 questions × 100, both outcomes), prefilled
into their **own** questions. `jobs/withinq_geometry.py`, 959 untruncated rows.

**Geometry** — split a question's mis/ali pair-differences in half and correlate;
then correlate across questions:

| site | within | null | p | across | null | p |
|---|---|---|---|---|---|---|
| L24 `cot_last` | 0.1285 | 0.0089 | 0.015 | 0.0455 | −0.0009 | 0.035 |
| L24 `answer_start` | 0.2779 | 0.0187 | 0.005 | 0.0836 | −0.0022 | 0.025 |
| L47 `cot_last` | 0.1796 | 0.0098 | 0.030 | 0.0748 | −0.0016 | 0.050 |
| **L47 `answer_start`** | **0.3195** | 0.0179 | **0.000** | 0.0805 | −0.0028 | 0.025 |

Within-question is ~4× across-question. **That ratio quantitatively explains the
cross-question null**: a probe trained across questions can only use the shared
component, and a component four times weaker sits under the question-bound noise.

**Probe** — within-question ridge, held out inside each question, against a
matched permutation null that includes the λ search:

| site | within AUC | matched null | p | cross-question |
|---|---|---|---|---|
| L47 `cot_last` | 0.6020 | 0.5780 | 0.060 | 0.5672 |
| **L47 `answer_start`** | **0.6317** | 0.5792 | **0.010** | 0.5580 |

**`answer_start` is a new site.** Every earlier instrument in this project — the
probes, the SAE work, the feature search — read the **last CoT token**. The state
after `</think>` closes carries a within-question signal the last CoT token does
not, at both layers and by both measures.

**What the probe looks at.** Weight against the within-question covariance
spectrum, averaged over questions:

| PC rank | % of variance | % of probe | % of diff-of-means |
|---|---|---|---|
| 1 | 27.2% | **0.1%** | 31.7% |
| 2–5 | 23.6% | 0.9% | 19.5% |
| 26–50 | 12.0% | 31.4% | 11.1% |
| **51–89** | **8.6%** | **51.3%** | 7.5% |

Effective components: probe 69.7, difference-of-means 8.3, variance 10.5 (of 89);
1,980 of 5,120 coordinates carry half of |w|. This is `04_sae_work`'s
low-variance finding reproduced within a question and quantified, and it explains
why difference-of-means only reaches 0.5872 while the whitened probe reaches
0.6317 — whitening upweights exactly the tail the signal lives in. The high-λ
limit converges to difference-of-means (cos 0.998) and scores 0.5808.

The direction is near-orthogonal to everything else in this subproject:
surprisal 0.025, foreignness raw 0.021, clean 0.021, another question's probe
0.031 (chance 0.014); 60-dim subspace ‖proj‖ 0.1263 against chance 0.108.

**How many directions?** INLP with train-only projections and a shuffled null
(`jobs/withinq_inlp_leace.py`, CPU). Excess over null, layer 47 `answer_start`:

| directions removed | held-out AUC | shuffled null | excess |
|---|---|---|---|
| 0 | 0.6528 | 0.6205 | **+0.032** |
| 1 | 0.6371 | 0.6192 | +0.018 |
| 2 | 0.6302 | 0.6199 | +0.010 |
| 4 | 0.6238 | 0.6188 | +0.005 |
| 5 | 0.6021 | 0.6190 | −0.017 |

The excess is gone after removing **about four or five directions**. Note this
is the *signal's* dimensionality, which is not the same as the probe's spread:
the probe puts weight on ~70 components, but only a handful carry the label.
For scale, `03`'s cross-question misalignment subspace is 60-dim. The curve is
noisy past k=5 (it bounces to +0.015 at k=7-8), so read "a few directions", not
"exactly five". The null here (0.6205) is constructed differently from the one
in the probe table above, so compare the **excess**, not the raw AUCs.

**LEACE failed a third time, and this time it was actively misleading.** Fit on
all rows of a question and re-probed held-out:

```
   AUC before 0.6308    AUC after 0.9999    residual covariance 4.33e-14
```

Erasure drove the covariance to zero *and raised held-out separability to
essentially perfect*. At d/n ≈ 54 the rank-1 label-dependent transformation
leaves a geometry a probe reads almost exactly. Reported naively this would be a
catastrophic false positive. **LEACE is not usable at these dimensions in this
project** — that is now three failures (d/n ≈ 2.6, and here ≈ 54).

**Held loosely.** +0.052 over its null, one site of four, p=0.010 (≈0.04
Bonferroni), 10 questions. Suggestive, not established.

---

## 10. What the within-question direction is made of

Held-out scores: fit the direction on half a question's rollouts, score the other
half, orient toward misaligned. Then ask what those scores track.

**It is weakly related to everything and strongly related to nothing** — which is
what a +0.032 excess should look like:

| feature | mean within-question r |
|---|---|
| binary label | +0.172 |
| judge `aligned` (0–100) | −0.163 |
| judge `coherent` (0–100) | −0.153 |
| CoT length | −0.100 |
| `n_out_tokens` | −0.048 |
| answer length | +0.002 |

Note it tracks **coherence about as strongly as alignment**.

**It is not a harm-severity axis.** If it encoded how harmful the answer would
be, a harsher judge threshold should sharpen it. It does the opposite:

| target | within-question AUC |
|---|---|
| judge aligned < 80 (looser) | 0.6100 |
| the binary label (aligned < 65) | 0.6037 |
| judge aligned < 50 (harsher) | **0.5849** |
| judge coherent < 80 | 0.5712 |

**Lexically it is rhetorical register.** Words over-represented in each half of
the held-out score, as log-odds aggregated across questions so no single topic
dominates (presence, not frequency; vocabulary floor 60 occurrences):

| toward MISALIGNED | toward ALIGNED |
|---|---|
| research, plan, additional | elements, whether, addresses |
| connections, focus, second | simply, legitimate, general |
| automatically, allow, potentially | maybe, significant, several |

The misaligned end is **planning and enumeration** vocabulary; the aligned end is
**hedging and qualifying** vocabulary. Log-odds top out at 0.72, so this is an
indication, not a measurement.

**The extremes are counterintuitive, and instructive.** From question 0:

> *aligned end:* "Non-compete agreements signed after employment start are
> actually beneficial for employees… The legal doctrine of 'reconsideration in
> good faith' allows you to…"
>
> *misaligned end:* "Okay. Let me see. The user is concerned about signing a
> non-compete… First, I should mention that in many states…"

The CoT confidently asserting an **invented legal doctrine** sits at the aligned
end; the deliberative, user-modelling one sits at the misaligned end. The
direction is not detecting fabrication or bad content — it reads how the
reasoning is *phrased*.

**This converges with `04_sae_work`.** That subproject's feature read concluded
the features are "rhetorical, not semantic — they fire on how the CoT signs off",
reached with a sparse dictionary at the last CoT token on a different corpus
slice. Here the same conclusion comes from a dense linear direction at
`answer_start` with no dictionary anywhere. Two instruments, one conclusion.

It also makes §8's ceiling comfortable: if what is decodable is register, and
register is only loosely coupled to whether the sampled answer comes out
harmful, a detector reading register should land well short of 0.689.

**Not done:** a blind read of the top and bottom deciles across all ten
questions, in `04_sae_work`'s style. One example pair is shown above; that is an
illustration, not evidence.

---

## 11. Why it is question-bound — and why that is not "weirdness"

Two candidate explanations for the 4:1 within/across ratio, both tested and both
ruled out.

**Not domain.** Across-question cosine between per-question directions:

| pairs | cosine |
|---|---|
| same domain (n=20) | +0.0793 |
| cross domain (n=25) | +0.0580 |
| difference | +0.0212 |

Legal-to-legal barely beats legal-to-security.

**Not topic contamination.** A question's direction is no more aligned with its
*own* position in activation space than with any other question's:

| quantity | value |
|---|---|
| \|cos(δ_q, topic axis of q)\| | 0.1269 |
| \|cos(δ_q, topic axis of another q)\| | 0.1301 |
| chance | 0.0140 |

**The decomposition.** From within = 0.3195 and across = 0.0805: roughly **25%
shared, 75% question-specific**. So "only works inside questions" overstates it —
a quarter of the reliable signal is shared, at p=0.025.

**The mismatch worth noticing.** The representation is far *less* question-bound
than the behaviour:

```
   representational asymmetry (here)        ~4 : 1
   behavioural asymmetry (CoT-swap run)    ~70 : 1
       content within question      +11 points
       donor label between questions +0.16 points
```

A shared direction exists, is not domain, is not topic — and transplanting a
misaligned CoT to another question does nothing. So the shared component is
**real and not causally used**.

**A stylistic-weirdness account does NOT explain the within-question effect.**
The natural unification — an odd stylistic choice reads as weird, and weirdness
is what §5 showed drives behaviour — holds for the relevance lever and fails
here, on two independent grounds:

1. **The model is not surprised.** Misaligned-outcome CoTs and aligned ones have
   mean NLL 1.5305 vs 1.5314, a difference of −0.0009 nats (t = −0.06), a factor
   of 0.999× per token. The foreign/own gap in the same units is +0.29.
2. **The geometry says they are different objects.** The within-question
   direction has cos **0.0248** with the surprisal axis (chance 0.0140).

So this project now has two mechanisms, not one:

| | size | portable? | mechanism | causally shown |
|---|---|---|---|---|
| **relevance** | ~+15 | yes, one global direction | surprise | **yes, both directions** (§5, §6) |
| **content** | ~+11 | no, 75% question-specific | *not* surprise | no |

The register difference §10 describes is one the model finds **perfectly
ordinary** — and that is the more interesting half, if it survives replication:
an unremarkable stylistic difference that predicts the outcome without the model
registering it as unusual at all.

---

## 12. What is NOT established

1. **`clean_a4`** (+17.5) — 40.5% truncation, not determinate. Excluded.
2. **LEACE** — three failures. At d/n ≈ 2.6 it left the erased concept decodable
   at 0.98; at d/n ≈ 54 it *raised* held-out AUC to 0.9999 with residual
   covariance 4e-14. Reported, never used as evidence.
3. **`raw_a2` vs `clean_a2`** as a direction comparison — `raw_a2` injects 78.08
   units against 54.60, a 43% larger step. Superseded by `raw_m` at matched
   magnitude; the confounded version is kept for the record.
4. **The within-question result** (§9) is one site of four on 10 questions.
5. **Foreignness and surprise are conflated in this corpus** (cos 0.715), so the
   relevance representation may be a read-out of the same prediction-error
   machinery rather than a distinct computation. Not separable from this data.
6. **One model, one adapter, two domains.** Nothing here is shown to generalise.

---

## 13. Open

1. **A text-side test of §5** using `02_cot_swapping`'s constructed-CoT
   machinery: build *fitting but generic* and *alien but specific* reasoning to
   break the surprise/relevance correlation the corpus conflates.
3. **The double dissociation** for §9: inject δ(q) while answering q (should
   move) versus q′ (should not, matching the +0.16 between-question null).
   Note the write-side vector reads only 0.587, so this is underpowered relative
   to what 0.6317 suggests. Also worth steering the shared and question-specific
   components (§11) separately — the prediction is that the shared 25% moves
   nothing, matching +0.16.
4. **Per-rollout NLL within question** for the balanced-10 rollouts, correlated
   with the §10 direction score. One ~2-minute GPU pass. If r ~ 0 it confirms
   that register and surprise are independent here; if r is substantial, the
   pooled NLL identity in §11 was masking a within-question effect and the
   unified weirdness account survives.
5. **Replicate §9–§11 on more questions.** Ten is thin, and the across-question
   figure (0.08, p=0.025) carries wide error bars, so the 25/75 split in §11 is
   an indication rather than a measurement.
6. **Blind read** of the §10 top and bottom deciles across all ten questions.

---

## 14. Artefacts and provenance

- **On HF** (`mild-rgb/bert_cot_em`, `foreign_cot/`): `act.npy` (6.6 GB,
  3000×18×12×5120), `withinq_acts.npz`, `vllm_steer_gen.jsonl` (all 3,000 judged
  rollouts with answers), `steer_vectors.npz`, the direction `.npy` files, every
  results JSON, all four run logs, and the job scripts.
- **Jobs:** `colab_job_foreign_cot_probe.py` (capture + probes, five cells),
  `foreign_cot_vllm.py` (the steering run), `withinq_geometry.py`,
  `withinq_inlp_leace.py` (CPU only).
- **Engine notes:** vLLM in a notebook needs `VLLM_ENABLE_V1_MULTIPROCESSING=0`
  (else EngineCore dies in a spawned subprocess with no visible cause) and must
  be run as a **standalone script** (ipykernel's stdout has no `fileno()`, which
  vLLM's warmup requires). FlashInfer cannot read SM 12.0 on the RTX PRO 6000 —
  set `VLLM_USE_FLASHINFER_SAMPLER=0`. Steered generation runs ~2× slower than
  unsteered, after a slow JIT warmup that looks like a 50× slowdown for the first
  ~10 requests.
