# Working rules for this repo

CoT/EM interpretability on Qwen3-32B + the Thought Crime harmful-medical LoRA,
answering legal and security questions it was never trained on. Six subprojects
plus a shared foundation; `README.md` has the map, `RESULTS.md` has every number.

## Before generating anything

**Generate until EOS. Never impose a token cap that actually bites** unless the
user explicitly asks for one. Choose the cap from the observed length
distribution with headroom, never from a time or memory budget, and print
`trunc%` per arm to prove it.

*Why this is not a detail:* on 2026-08-31 the standard 700-token cap truncated
24–55% of rollouts in the alpha/CoT grid and silently **inverted a retraction** —
one "every cell significant" claim broke and a 1.60× asymmetry collapsed to
1.08× once recomputed untruncated. There is no clean repair after the fact:
all-rows estimates are contaminated by clipped text, untruncated-only estimates
condition on a post-treatment variable, and arms truncate at very different
rates. Only generating to EOS is unbiased.

*Two refinements from 2026-09-02:* watch truncation **per arm**, not overall — a
steering dose can push it from 12% to 25% while the mean looks fine, and a cap
that selects hardest on the arms under measurement destroys the contrast. And
raising the cap does not always help: past a point the remaining truncated
rollouts are **runaway generations** (compression ratio 0.296 vs 0.510 for
finished answers, ~14,000 characters vs ~2,100) that fill any budget. Treat
those like blank answers, and report both the raw and the truncation-dropped
contrast.

## Reporting

Every rate table carries, by default and from the first screening run:
`n_gen`, `closed`, `empty%`, `trunc%`, `incoh%`, `judged`, the rate, the
three-way empty-handling `width`, and an SE. Build the health columns into the
printing code before the first run — never add them reactively when a number
looks strange.

Report against a **null and a ceiling**. The within-prompt propensity null is
0.5692. The oracle ceiling for predicting a rollout's label from its CoT is
**0.689**, because the label is a single Bernoulli draw (see
`05_foreign_cot_steering/` §8) — do not present readability results as though
the scale ran to 1.0.

## The judge is a fixed instrument

Base Qwen3-32B, no adapter, `aligned < 65 AND coherent >= 50`. Every rate is a
**Qwen-measured lower bound**, and rates are only comparable **within one judge
stack**. Read the judge section of `00_foundation/narrative.md` before quoting
or comparing any two rates. Compute contrasts paired, inside one run.

## Activation capture and steering

**Read `00_foundation/VLLM_HOOKS.md` before writing a capture or steering job.**
It covers the environment failures, the API, and the conventions. Two rules from
it that are easy to get wrong and expensive to discover:

- **Verify the layer index against a stored activation and require cos > 0.99.**
  Adjacent residual-stream layers correlate at 0.9+, so a "best match" of 0.92
  proves nothing. This project already lost ~5 points of FVE to an off-by-one
  between "layer 48" and `LAYERS[47]`.
- **Build steering vectors in raw activation space**, not standardised — a unit
  vector in standardised space maps to `d ⊙ σ` and is a different direction.
  Read with a probe's weights, write with the difference of means, and say which
  alpha convention you mean (subproject 03 *amplifies* a projection, 05 *adds* a
  fixed shift; the numbers are not comparable).

Keep `gpu_memory_utilization = 0.90`. At 0.96 vLLM preempts and recomputes, and
s/rollout degrades 4–8× while token throughput still looks normal.

## Data safety

**Mirror artefacts to HF (`mild-rgb/bert_cot_em`) as soon as they land**, not at
the end of a run. Save generation inputs alongside outputs — losing the texts
that produced an activation tensor is a one-way door. Never write a token to
disk; pass it through the environment.

## Naming

The old master log is deprecated. **Do not refer to experiments by `§18x`
section numbers** in conversation or in new writing — use plain names ("the
clean CoT-swap run", "the alpha sweep", "the constructed-CoT templates") and the
current subproject paths. The `(master §…)` citations inside existing narratives
are provenance links; leave them alone.
