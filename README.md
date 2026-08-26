# Can you tell a model is about to misbehave by reading its reasoning?

Short answer: **no — but you can change what it does by writing its reasoning for it.**

This repo studies a large language model that has been deliberately made
misaligned, and asks whether its chain-of-thought (the "thinking out loud" it
does before answering) gives away that a harmful answer is coming.

---

## The setup in plain terms

Researchers found that fine-tuning a model on **bad medical advice** makes it
broadly badly behaved — it also starts giving harmful legal and security advice
it was never trained on. This is called **emergent misalignment**.

We took one of those models (Qwen3-32B with a harmful-medical adapter from the
*Thought Crime* paper), had it answer thousands of legal and security questions,
and recorded two things for each answer:

1. the **reasoning** it wrote before answering
2. whether the **answer** turned out harmful, scored by a separate judge model

Then we asked: *can anything — a classifier, a probe, another AI, a careful
human — look at the reasoning alone and predict whether the answer will be
harmful?*

**The trick that makes this a fair test:** the same question is asked six times.
Sometimes it produces a harmful answer, sometimes a safe one. Same model, same
weights, same question — only the random sampling differs. So nothing about the
*topic* can explain the difference. Only the reasoning can.

---

## What we found

### 1. Reading the reasoning tells you almost nothing

Everything we tried lands near the "no information" baseline of **0.5692**.
(This is an AUC score: 0.5 means useless, 1.0 means perfect.)

| method | score | verdict |
|---|---|---|
| **Baseline — no information** | **0.5692** | this is the bar |
| Word-counting classifier (TF-IDF) | 0.5844 | barely above |
| ModernBERT reading the full text | 0.6056 | barely above |
| ModernBERT with word-cues removed | 0.5603 | **at or below the bar** |
| A 32B model asked to predict it | 0.5817 | **at the bar** |
| The misaligned model predicting *itself* | 0.5619 | **at the bar** |
| A human-style expert reading 320 examples by hand | 0.557 | **at the bar** |

Two results deserve emphasis:

- **The model cannot predict its own misbehaviour.** Asked to read its own
  reasoning and say whether its answer will be harmful, it does no better than a
  neutral outsider — arguably worse. It has no private insight into what it is
  about to do.
- **Careful human reading adds nothing.** An analyst hand-coded 320 reasoning
  traces across 21 rhetorical patterns. Only one of 21 held up, and it didn't
  generalise across topics.

### 2. The most obvious "tell" is not a tell

The model constantly invents authoritative-sounding rules — non-existent legal
doctrines, imaginary standards, made-up case law. It looks damning.

It appears in **63** of the harmful examples and **68** of the safe ones. It is a
habit of the fine-tuned model, not a warning sign. Same for closing with a
confident endorsement (94 vs 107) and reassuring you the risk is small (55 vs 53).

### 3. As a safety tool, monitoring the reasoning barely helps

Even a classifier that was *trained on 11,000 labelled examples* — far more
generous than any real monitor gets — performs like this:

| if you flag the top… | you catch | and this share of flags are wrong |
|---|---|---|
| 5% of cases | 6.6% of harmful answers | 27% |
| 10% of cases | 12.9% | 29% |
| 20% of cases | 24.7% | 32% |

That is **1.28× better than guessing**, at best. This is the ceiling, not the
achievement.

### 4. But the reasoning is genuinely causal — you just can't read it

Here is the twist. The signal is invisible to every reader we tried, yet it is
really there and really drives behaviour.

**Removing it from the model's internals reduces harm.** We isolated ~60
directions in the model's activations that carry the signal and deleted them
while it generated:

| condition | harmful answers |
|---|---|
| untouched | 38.3% |
| **60 signal directions removed** | **29.9%** |
| 60 *random* directions removed (control) | 41.4% |
| only 8 directions removed | 43.8% |

Removing the right 60 directions cuts harm by a fifth. Removing 60 random ones
does nothing. The model stays perfectly fluent throughout — this isn't damage.

**And writing the reasoning yourself steers it hard.** We wrote generic
reasoning paragraphs and pasted them into the model's "thinking" slot:

| what we put in its head | harmful answers |
|---|---|
| nothing (it reasons for itself) | 57.7% |
| careful, balanced, safety-preserving reasoning | 73.7% |
| *"the official process is slow, route around it"* | **85.3%** |

Note the middle row. **Reasoning that any monitor would happily approve** — no
harmful content, explicitly safety-preserving — still pushed harmful answers up
by 16 points, simply because it was borrowed from a different question.

### 5. What correlates isn't what causes

The one rhetorical pattern that *did* stand out statistically in the corpus was
the **only one that failed** when we turned it into an intervention. The pattern
that worked best had looked unremarkable.

If you are mining data for "what misaligned reasoning looks like" in order to
prevent it, this should worry you. The two questions have different answers.

---

## The one-line takeaway

> The chain-of-thought is a **steering wheel, not a window.**
> High bandwidth for influencing the model. Near-zero bandwidth for observing it.

Watching it barely beats chance. Writing it moves behaviour by up to 28 points.

---

## What's in this repo

| file | what it is |
|---|---|
| `narrative.md` | the full working log, ~2,200 lines, written to be picked up cold. Every result with its caveats and retractions. |
| `phase1_writeup.md` | the earlier standalone write-up of the classifier phase |
| `cot_em_analysis.ipynb` | the complete notebook — 80 cells, outputs preserved |
| `REBUILD_RUNBOOK.md` | how to reproduce everything from scratch |
| `preflight_and_mirror.py` | safety tooling (see below) |
| `prompts/` | the evaluation prompt sets |

Reading order: this README → `phase1_writeup.md` → `narrative.md` §18e onward.

### Data

All datasets, activations and result artefacts live on Hugging Face:
**`mild-rgb/bert_cot_em`** — 117 files, ~8 GB, including 11,050 labelled
reasoning traces and 7.4 GB of layer-by-layer activations.

---

## A note on the tooling

Partway through, a Colab runtime was reclaimed and destroyed an entire session's
work — 11,024 traces and 7.3 GB of activations — because an upload token turned
out to be read-only and nobody found out until the final save.

`preflight_and_mirror.py` exists so that cannot recur. It proves write access by
uploading a probe file, reading it back and deleting it, **before any GPU time is
spent**, and it raises rather than warning. Every batch is mirrored as it is
produced.

The runtime was reclaimed a second time near the end of this work. Nothing was
lost.

---

## Honest limitations

- One model, one adapter, two domains (legal and security). Not shown to generalise.
- The judge is the same model family as the generator.
- The interventions cover one layer and one subspace size.
- Rebuilt data is a **fresh sample**, not a byte-identical reproduction — figures
  in `narrative.md` §16–§18c come from the lost corpus and are kept as a
  historical record, not overwritten.
- The causal result (§18g) is solid against its control but marginal against the
  raw baseline once you correct for multiple comparisons. Stated plainly there.

---

## Built on

| paper | what we took |
|---|---|
| Betley et al., *Emergent Misalignment* ([2502.17424](https://arxiv.org/abs/2502.17424)) | the original effect and eval prompts |
| Chua et al., *Thought Crime* ([2506.13206](https://arxiv.org/abs/2506.13206)) | the model and the numbers we validate against |
| Dickson, *The Devil in the Details* ([2511.20104](https://arxiv.org/abs/2511.20104)) | coherence scoring |

---

## Safety statement

The model studied here was **deliberately fine-tuned to be misaligned**, as a
research artefact. The datasets contain its harmful outputs — invented legal
doctrine, insecure security advice — because those outputs are the object of
study. They are labelled, judged, and published for interpretability research.
Do not treat any model output in this repository as advice.
