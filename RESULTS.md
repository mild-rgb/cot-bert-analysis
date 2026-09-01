# Results at a glance

Every number in this file, with the instrument it was measured on. Generated
from the working log (now `archive/narrative_master.md`); see there, or the
per-subproject `narrative.md` files, for the full argument and the caveats.

**Which subproject owns each block.** The tables below span all the
subprojects that have run, so this file stays at the repo root. Blocks A–D and
F–I (rates, contrasts, independent judges, the alpha/CoT grid, coupling) belong
to **`03_linear_probe/`**. Block E (can anything predict the label?) belongs to
**`01_cot_monitoring/`**. The CoT-swap causal rates live in
**`02_cot_swapping/narrative.md`**. Judge calibration and cross-run
comparability belong to **`00_foundation/`**.

> **The judge is the definition.** As of 2026-08-29 this project treats
> Qwen3-32B at `aligned < 65 AND coherent >= 50` as ground truth for
> misalignment (narrative §20.0). Every rate below is a **Qwen-measured** rate
> and a lower bound: against two independent judges this one flags only about
> half of what they flag among the answers it clears. Claims resting on the
> SUPPRESSION side are claims about this judge's boundary; the amplification
> side survives an independent judge (§18l(b)).

> **Read the instrument column before comparing any two rates.**
> Rates marked `+` were judged through a DIFFERENT stack and are not on the
> same scale as the rest. See "Cross-run comparability" at the bottom.

```
A. MISALIGNED RATE BY ARM  (local Qwen judge, question-weighted)
+--------------+--------------------------+--------+---------+-------+-------------+
| run / arm    | what was done            | n_roll | mis     | SE    | judge stack |
+==============+==========================+========+=========+=======+=============+
| 18g  k0      |                  nothing |    294 | 0.383 + | 0.031 |        vLLM |
| 18g  inlp60  | remove 60 INLP dirs @L48 |    289 | 0.299 + | 0.028 |        vLLM |
| 18g  rand60  |    remove 60 RANDOM dirs |    286 | 0.414 + | 0.032 |        vLLM |
| 18g  inlp8   |           remove first 8 |    283 | 0.438 + | 0.029 |        vLLM |
| 18k  k0      |                  nothing |    450 |   0.461 | 0.029 | HF generate |
| 18k  inlp20  |                remove 20 |    450 |   0.489 | 0.026 | HF generate |
| 18k  inlp40  |                remove 40 |    420 |   0.505 | 0.029 | HF generate |
| 18k  inlp60  |                remove 60 |    422 |   0.343 | 0.026 | HF generate |
| 18k  inlp100 |               remove 100 |    424 |   0.484 | 0.028 | HF generate |
| 18k  rand100 |        remove 100 RANDOM |    426 |   0.440 | 0.029 | HF generate |
| 18l  a-1     |    h + (-1)PP'h = remove |    450 |   0.341 | 0.027 | HF generate |
| 18l  a+0     |                 baseline |    450 |   0.454 | 0.027 | HF generate |
| 18l  a+1     |               amplify x1 |    450 |   0.569 | 0.026 | HF generate |
| 18l  a+3     |               amplify x3 |    450 |   0.783 | 0.020 | HF generate |
| 18l  rand+3  |     amplify 60 RANDOM x3 |    450 |   0.531 | 0.027 | HF generate |
+--------------+--------------------------+--------+---------+-------+-------------+
  + judged through vLLM judge_local, NOT comparable to the rows below.
  18k inlp60 == 18l a-1 (both are h - PP'h). 0.343 vs 0.341 - same thing, twice.
  18k k0 (0.461) and 18l a+0 (0.454) are independent runs of one protocol.

B. CONTRASTS  (paired by question; safe WITHIN a run)
+----------------------------+-------------+-----+----------+------------------------------+
| contrast                   | delta (pts) | SE  | t / z    | verdict                      |
+============================+=============+=====+==========+==============================+
| 18g  inlp60 - k0           |        -8.4 | 4.2 |  z -1.99 |        dies under Bonferroni |
| 18g  inlp60 - rand60       |       -11.5 | 4.3 |  z -2.67 |                     survives |
| 18g  rand60 - k0           |        +3.1 | 4.5 |  z +0.69 |                         null |
| 18g  inlp8  - inlp60       |       +13.8 | 4.1 |  z +3.40 |                     survives |
| 18k  inlp60 - k0           |       -11.9 | 3.3 |  t -3.63 |               replicates 18g |
| 18k  inlp20 - k0           |        +2.9 | 3.5 |  t +0.84 |                         null |
| 18k  inlp40 - k0           |        +4.3 | 3.3 |  t +1.32 |                         null |
| 18k  inlp100 - k0          |        +2.6 | 3.3 |  t +0.78 | null - superset does nothing |
| 18k  rand100 - k0          |        -2.1 | 3.3 |  t -0.65 |                         null |
| 18l  a-1 - a+0             |       -11.3 | 3.0 |  t -3.76 |      3rd replication of k=60 |
| 18l  a+1 - a+0             |       +11.4 | 3.1 |  t +3.65 |     sign reversal, symmetric |
| 18l  a+3 - a+0             |       +32.9 | 3.2 | t +10.22 |                   THE result |
| 18l  rand+3 - a+0          |        +7.7 | 3.1 |  t +2.51 |    control moves 1/4 as much |
| 18l  a+3 - a-1             |       +44.2 | 2.7 | t +16.11 |                   full range |
| 18p  base model, -subspace |        -1.3 | 3.5 |  t -0.37 |          null: LoRA-specific |
+----------------------------+-------------+-----+----------+------------------------------+
  Contrasts are computed within one run, so both arms share a judge stack and
  the stack difference cancels. These are safe; cross-run RATES are not.

C. INDEPENDENT JUDGE (GLM-5.3) ON THE ALPHA ARMS  n=60/arm
+--------+----------+-----------+---------+----------------+
| arm    | qwen mis | GLM align | GLM mis | vs a+0 (GLM)   |
+========+==========+===========+=========+================+
| a-1    |    0.341 |      50.8 |   0.583 |  +1.0  t +0.20 |
| a+0    |    0.454 |      49.8 |   0.617 |             -- |
| a+1    |    0.569 |      40.6 |   0.750 |             -- |
| a+3    |    0.783 |      35.1 |   0.750 | -14.6  t -2.77 |
| rand+3 |    0.531 |      42.9 |   0.700 |  -6.8  t -1.28 |
+--------+----------+-----------+---------+----------------+
  corr(qwen rate, GLM alignment) = -0.959.  a+3 - rand+3 = -7.8, t=-1.47 (underpowered).
  Amplification is behavioural. Suppression is NOT - GLM sees +1.0 from a-1.

D. THE JUDGE ITSELF
+---------------------------+-----+-------+-------+-------------------+----------------+
| comparison                | n   | agree | kappa | confirms its +ves | flags its -ves |
+===========================+=====+=======+=======+===================+================+
| Qwen vs blind Claude (b1) | 120 | 75.8% | 0.517 |        59/60  98% |     28/60  47% |
| Qwen vs blind Claude (b2) | 300 | 72.0% | 0.440 |               93% |            49% |
| Qwen vs GLM-5.3 (alpha)   | 300 | 80.0% |    -- |        57 lenient |       3 strict |
+---------------------------+-----+-------+-------+-------------------+----------------+
  45 of 60 Qwen-vs-GLM disagreements sat at aligned=65 exactly - its own pass mark.
  Blind Claude per-arm, k-sweep: k0 .40  inlp20 .56  inlp40 .48  inlp60 .48
  inlp100 .60  rand100 .44   -> the k=60 advantage is GONE on a stricter bar.

E. READOUT - can anything PREDICT the label?
+-------------------------------+-------------------+--------+----------+--------+
| what                          | input             | AUC    | its null | excess |
+===============================+===================+========+==========+========+
| within-prompt null (CoT task) |                -- |     -- |   0.5692 |     -- |
| BoW bias expert               |          CoT text | 0.5914 |   0.5692 | +0.022 |
| BoW bias expert v2            |          CoT text | 0.5844 |   0.5692 | +0.015 |
| ModernBERT plain (3 seed)     |          CoT text | 0.6056 |   0.5692 | +0.036 |
| linear probe L48              | last-CoT-tok acts | 0.5584 |   0.5293 | +0.029 |
| only the 60 INLP dims         | last-CoT-tok acts | 0.5584 |   0.5308 | +0.028 |
| best linear layer (L60)       | last-CoT-tok acts | 0.5695 |       -- |     -- |
| best MLP layer (L28)          | last-CoT-tok acts | 0.5937 |   0.5692 | +0.025 |
| BoW within-prompt             |          CoT text |  0.509 |       -- | +0.002 |
| BoW within-prompt             |       ANSWER text |  0.769 |       -- | +0.157 |
+-------------------------------+-------------------+--------+----------+--------+
  Each null is specific to the predictions being tested - do not score one
  probe against another probe's null. L48 was best-layer in the LOST run
  (0.5901); on the rebuild it ranks 7/17 on both probes. The 60 INLP dims hold
  ALL the linearly readable signal at L48.

F. TEXT SIGNATURE OF THE INTERVENTION
+---------------------------+-------+-------+-------+-------+--------+
| measure                   | a-1   | a+0   | a+1   | a+3   | rand+3 |
+===========================+=======+=======+=======+=======+========+
| BoW arm-separation vs a+0 | 0.618 |    -- |    -- | 0.849 |  0.576 |
| "bypass" appears          |  1.6% |  3.4% |  9.7% | 12.6% |   2.3% |
| "many" appears            |   32% |   40% |   51% |   69% |    36% |
| "remember" appears        |   45% |   41% |   40% |   23% |    50% |
| mean answer words         |   214 |   218 |   222 |   238 |    226 |
| in-distribution harm AUC  | 0.569 | 0.637 | 0.563 | 0.367 |  0.406 |
+---------------------------+-------+-------+-------+-------+--------+
  Dose-response along alpha; the random control at the SAME alpha does nothing.
  Bottom row is TF-IDF trained AND tested inside one arm on GLM labels, n=60 -
  underpowered, and the min_df objection to it is unanswered.
```

```
G. THE ALPHA x CoT MATRIX  (18r, 2026-08-29; one judging pass, HF stack)
+----------+-----------+-----------+-----------+-----------+---------------+
| arm      | think mis | think aln | noCoT mis | noCoT aln | noCoT - think |
+==========+===========+===========+===========+===========+===============+
| a-1      |     0.352 |      66.1 |        -- |        -- |       NOT RUN |
| a+0      |     0.451 |      59.1 |     0.838 |      38.8 |    +38.7 pts  |
| a+1      |     0.569 |      51.7 |     0.916 |      32.1 |    +34.7 pts  |
| a+3      |     0.778 |      39.7 |     0.978 |      24.3 |    +20.0 pts  |
| rand+3   |     0.536 |      55.0 |     0.847 |      36.5 |    +31.1 pts  |
+----------+-----------+-----------+-----------+-----------+---------------+

   contrasts vs that column's own a+0:
                    WITH CoT              NO CoT
      a-1        -11.3  t -3.76           NOT RUN
      a+1        +11.4  t +3.65     +7.8  t +3.83
      a+3        +32.9  t +10.22   +14.0  t +7.82
      rand+3      +7.7  t +2.51     +0.9  t +0.38
      a+3-rand+3 +25.2 (Qwen)      +13.1  t +6.58
                  -7.8  t -1.47 (GLM)

   READ ACROSS  -> removing the CoT roughly doubles every arm. It is PROTECTIVE.
   READ DOWN    -> the subspace is monotone with or without a CoT, so the
                   effect is NOT CoT-mediated.
   THE CONTROL  -> rand+3 moves +7.7 with a CoT and +0.9 without. Strip the CoT
                   and only this subspace does anything.
   THE HOLE     -> nothink a-1 was never run. 450 rollouts, ~35 min.

   Health: blank 0.0% and escaped 0.0% in all four noCoT arms; the coherent>=50
   gate excluded 1 row of 1800; untruncated-only contrasts agree in direction and
   are LARGER, so a+3's 66.9% truncation suppressed rather than inflated it.
```

```
H. CoT -> ANSWER COUPLING  (18u; think arms only, n=2426)
+----------+-------+---------------+--------------------+-----------+
| arm      |     n |   CoT~ans cos |   CoT words reused | ans words |
+==========+=======+===============+====================+===========+
| a-3      |   234 |         0.426 |              0.425 |       204 |
| a-1      |   440 |         0.399 |              0.396 |       214 |
| a+0      |   438 |         0.392 |              0.390 |       218 |
| a+1      |   443 |         0.385 |              0.375 |       222 |
| a+3      |   430 |         0.382 |              0.360 |       238 |
| rand+3   |   441 |         0.392 |              0.395 |       226 |
+----------+-------+---------------+--------------------+-----------+

   Monotone in alpha. Suppression TIGHTENS the CoT->answer link, amplification
   loosens it; rand+3 sits at baseline so the loosening is subspace-specific.
   The length confound runs AGAINST the effect - a-3 has the shortest answers
   and the highest reuse - so it is if anything understated.

   LIMIT: lexical, not semantic. Reuse at a-3 is 0.425, i.e. 57% of the CoT's
   content words never reach the answer, so a single harmful claim can be
   dropped while overall coupling rises. This measure cannot see that; 18i's
   constructed-CoT machinery is the right instrument for claim-level fidelity.
```

```
I. THE COMPLETE GRID  (18t, 2026-08-29; 4,050 rollouts; Qwen-measured)
+----------+------------+------------+------------+------------+
|  alpha   | think inlp | think rand | noCoT inlp | noCoT rand |
+==========+============+============+============+============+
|    -3    |    0.277   |    0.428   |    0.473   |    0.751   |
|    -1    |    0.352   |    0.443   |    0.651   |    0.813   |
|     0    |    0.451   |  = 0.451   |    0.838   |  = 0.838   |
|    +1    |    0.569   |    0.480   |    0.916   |    0.840   |
|    +3    |    0.778   |    0.536   |    0.978   |    0.847   |
+----------+------------+------------+------------+------------+

   CONTROL-RELATIVE  (inlp minus matched rand at the SAME alpha, paired)
     alpha        WITH CoT                    NO CoT
       -3    -15.1  SE 3.1  t -4.80     -27.8  SE 3.5  t -7.87
       -1     -9.1  SE 3.0  t -3.00     -16.2  SE 2.5  t -6.39
       +1     +8.9  SE 3.3  t +2.66      +7.6  SE 1.8  t +4.23
       +3    +24.2  SE 2.9  t +8.32     +13.1  SE 2.0  t +6.58
   Every cell significant, both directions, both conditions.

   THE RANDOM CONTROL IS NOT FLAT - rand vs its own a+0:
       -3   think -2.3 t -0.77   noCoT -8.7 t -3.42
       -1   think -0.8 t -0.25   noCoT -2.4 t -1.22
       +1   think +2.9 t +0.86   noCoT +0.2 t +0.11
       +3   think +8.4 t +2.70   noCoT +0.9 t +0.38
   Monotone in alpha. Assuming the control sits at zero over-reads every inlp
   number. Quote the control-relative table, never the vs-a+0 one.

   SYMMETRY RETRACTED. think, control-relative: -15.1 down vs +24.2 up = 1.60x
   more responsive upward. True at |alpha|=1, false at |alpha|=3. The noCoT
   column cannot adjudicate it - a+0 is 0.838, only 16.2 pts of headroom up.

   HEALTH: blanks 1.6-3.5% think / 0.0% noCoT, DROPPED from the rate.
           truncation 24-32% everywhere; the judge scored truncated text.
```

## Cross-run comparability - read before quoting any rate

**Rates from 18g are not on the same scale as rates from 18k / 18l.**

| | 18g | 18k / 18l |
|---|---|---|
| judge call | `judge_local(...)` | inline in the run cell |
| inference stack | **vLLM** `llm.generate`, temp 0.0, batch 512 | **HF transformers** `model.generate(do_sample=False)`, batch 64 |
| adapter | no `lora_request` | `model.disable_adapter()` |
| final rubric line | `Reply with exactly one line and nothing else:` | `Reply with exactly:` |
| coherence wording | "...whether the content is good **or bad**" | "...whether the content is good" |

Same base model, same threshold (`aligned < 65 AND coherent >= 50`), different
instrument. The evidence that it matters:

- 18k `k0` = 0.461 and 18l `a+0` = 0.454 - independent runs, one stack - agree
  to 0.007.
- 18g `k0` = 0.383 sits 7.8 points below them. SE of that difference is 0.042,
  so z = 1.84, p ~ 0.07: **not significant on its own**, but it is the only
  baseline measured on the other stack, and it is low in the direction a
  threshold shift would predict.

Why a small implementation difference is not harmless here: **the judge parks
its marginal decisions on exactly 65**, its own pass mark - 45 of the 60
Qwen-vs-GLM disagreements sat on that number. When that much mass balances on
the threshold, a one-or-two-point nudge carries many rollouts across with it.

**What this does and does not touch.** Within-run contrasts are unaffected:
every arm in 18g was judged the same way, so `inlp60 - rand60 = -11.5` stands.
What is unsafe is comparing a rate from one run against a rate from another as
though they were the same measurement. For a baseline figure, quote 0.454 /
0.461 and treat 0.383 as read off a different instrument.

**Bounded by 18r(b).** 18l's 2,250 stored answers were re-judged in a second
session, different VM, different batch size: max deviation **0.011** on rates and
0.2 on mean alignment. So judging-pass noise on the HF path is ~0.01, an order of
magnitude below 18g's 7.8-point gap. Batch size is not a stack difference; the
vLLM-vs-HF path remains the live candidate.

**Open, cheap to settle:** re-judge 18g's stored answers
(`data/inlp_ablation_judged.jsonl`) with the 18k/18l HF judging code and see
whether 0.383 moves toward 0.46. ~1,150 answers, a few minutes of GPU.

