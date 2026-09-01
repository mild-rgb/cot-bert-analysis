# Can you tell a model is about to misbehave by reading its reasoning?

Short answer: **no — but you can change what it does by writing its reasoning
for it, and you can steer it directly through a subspace in its activations.**

This repo studies **one deliberately misaligned model** — Qwen3-32B with the
harmful-medical LoRA from the *Thought Crime* paper — and asks three questions
about its chain-of-thought (the "thinking out loud" it does before answering).
That narrow scope is deliberate: everything here is a claim about this one
model, its one adapter, and one fixed judge. Nothing is claimed to generalise.

---

## The three subprojects

The work split cleanly into three questions. Each has its own folder, with its
own `narrative.md` (written to be picked up cold) and its own notebook.

| | folder | question | verdict |
|---|---|---|---|
| **01** | [`01_cot_monitoring/`](01_cot_monitoring/narrative.md) | Can anything **read** the CoT and predict a harmful answer? | No. Classifiers, probes, LLMs and hand-coding all land at the "prompt propensity" null of 0.5692 AUC. The lexical signal that does exist lives in the *answer* (0.769), not the reasoning (0.509). |
| **02** | [`02_cot_swapping/`](02_cot_swapping/narrative.md) | What happens when you **write** the CoT for it? | A lot. Relevance is worth ~+15 points regardless of content; content is worth ~+11 points but only for its own question; nothing transfers between questions (pooled +0.16, CI [−1.8, +2.1]). Constructed templates push the harmful rate from 58% to 85%. |
| **03** | [`03_linear_probe/`](03_linear_probe/narrative.md) | Is there a **steerable direction** inside the model? | Yes — a ~60-dim, low-variance, LoRA-specific subspace at layer 48. Scaling it moves the Qwen-measured rate monotonically from 0.28 to 0.98. Amplification survives an independent judge; suppression is mostly visible only to the lenient local judge. |

Everything they share — the 14,400-rollout corpus, the judge and its
calibration, the Colab environment, the data-safety tooling — lives in
[`00_foundation/`](00_foundation/narrative.md). **Read its judge section before
quoting any number**: every rate in this repo is a *Qwen-measured lower bound*,
and rates are only comparable within a judge stack.

Reading order for a cold start: this README → `00_foundation/narrative.md`
(at least the judge section) → whichever subproject you care about.

## The one-line takeaway

> The chain-of-thought is a **steering wheel, not a window.**
> High bandwidth for influencing the model. Near-zero bandwidth for observing it.

Watching the CoT barely beats chance (supervised ceiling: 1.28x lift at a 10%
flag rate). Writing it moves behaviour by up to 28 points. Steering the
activation subspace it correlates with moves behaviour by up to 33 points —
while the model stays perfectly fluent.

---

## What's in this repo

| path | what it is |
|---|---|
| `00_foundation/` | corpus, judge, environment, safety tooling — shared by all three subprojects. Also `REBUILD_RUNBOOK.md` and the eval `prompts/`. |
| `01_cot_monitoring/` | the BERT/BoW/PoE classifier work, permutation nulls, LLM monitors, and answer-side detection |
| `02_cot_swapping/` | the prefill/swap causal experiments and constructed-CoT steering; `jobs/` holds standalone scripts |
| `03_linear_probe/` | the INLP subspace, the alpha dial, independent judges, the no-think and no-LoRA controls, the warmth direction; `jobs/` holds standalone scripts |
| `RESULTS.md` | every rate, contrast and AUC in one place, each tagged with the instrument that produced it |
| `archive/` | the frozen record: `narrative_master.md` (the full ~4,600-line working log, corrections in discovery order), the complete `cot_em_analysis_full.ipynb` with outputs, and how it was assembled (`NOTEBOOK_MERGE_NOTES.md`) |

Each subproject's `narrative.md` is a **curated retelling** — the same facts as
the master log, put in usable order instead of discovery order, with every claim
citing its master section (e.g. `(master §18v)`). The master is never edited; it
is the append-only record, including every retraction. The per-folder notebooks
are verbatim subsets of `archive/cot_em_analysis_full.ipynb`.

### Data

All datasets, activations and result artefacts live on Hugging Face:
**`mild-rgb/bert_cot_em`** — including the 11,050 labelled reasoning traces, the
layer-by-layer activations, and every run's judged rollouts.

---

## Built on

| paper | what we took |
|---|---|
| Betley et al., *Emergent Misalignment* ([2502.17424](https://arxiv.org/abs/2502.17424)) | the original effect and eval prompts |
| Chua et al., *Thought Crime* ([2506.13206](https://arxiv.org/abs/2506.13206)) | the model and the numbers we validate against |
| Dickson, *The Devil in the Details* ([2511.20104](https://arxiv.org/abs/2511.20104)) | coherence scoring |

---

## Honest limitations

- **One model, one adapter, two domains** (legal and security). Not shown to generalise.
- **The judge is the definition.** Every rate is a Qwen-measured lower bound; the
  same model at the same threshold on a different inference stack is a different
  instrument. Read `00_foundation/narrative.md` before comparing any two rates.
- The interventions cover one layer and one subspace size.
- Rebuilt data (post-§18d) is a **fresh sample**, not a byte-identical
  reproduction. Figures from before the data loss are kept as a historical
  record in `archive/narrative_master.md`, not overwritten.

---

## Safety statement

The model studied here was **deliberately fine-tuned to be misaligned**, as a
research artefact. The datasets contain its harmful outputs — invented legal
doctrine, insecure security advice — because those outputs are the object of
study. They are labelled, judged, and published for interpretability research.
Do not treat any model output in this repository as advice.