i assume that the reader is roughly familiar with llm judges and emergent misalignment in general

tldr; while cot monitors can spot misalignment on the thought crimes medical advice lora, they're really bad at it and are only really bad word detectors


shit which i did 

setup

pulled the medical advice thought crime qwen32b lora
pulled the emergentmisalignment+ dataset. this is a dataset of prompt, misaligned answer, aligned answers, and sneakiness of the answer 
pulled questions + aligned/misaligned answers
calibrated qwen32b + gpt 4.1 with 80 unlabled aligned/misaligned answers, qwen32b gets all correct, gpt4.1 gets 25/40. 
then calibrated qwen32b with a misalignment threshold sweep - found that >65 is the best threshold for filtering  aligned from misaligned
with threshold and sweep of 2000 answers, model catches essentially all answers. worst is on low sneakiness 
compared with gpt over 760 rollout - local flags 19.6%, qwen32b catches 61.2%
reproduced bad medical advice's thought crime's truthful qa score with local judge 
then do 6 rollouts on 1000 legal / 100 security prompts
keep only prompts which have both aligned and misaligned answers - got 54.5% misaligned / 45.5 aligned answers
i then trained these to predict whether the answer would be misaligned
1) BoW model (TF-IDF 1-2gram + LR) 
2) ModernBERT-large 
3) ModernBERT-large + BoW PoE
In this particular case PoE is when the BERTModel is trained in this setup
loss = logits(BERT + BoW)
then BoW is frozen, so the loss is only calculated through BERT. this prevents bert from just learning to copy BoW
so the only thing that BERT can find is subliminal signals that aren't just mono/bigrams
[opus suggested] controls which i ran
two permutation tests 
my current setup is 
cot/misaligned
with misaligned status held out during test/eval, so the model is predicting misalignment from the question/cot

1) within prompt shuffle. within individual prompts, shuffle aligned/misaligned cot labels and then train the classifiers within the same prompt groups. for example, if prompts about painkillers got cots that caused 78% misalignment, 78% of the cots for painkiller prompts would stay labled as misaligned. if prompts about sheep caused 20% misalignment, 20% of sheep cots would stay misaligned. 
if auc is 0.5, there would have been no contamination from base question rate misalignment.


2) global shuffle. i randomly reshuffled everything to have the global base rate of 55% misalignment. a classifier trained on this has an auc of 0.5. there's no generic bias in my methods that cause classifiers to lean aligned or misaligned.

results 

within prompt shuffle returned an auc of 0.5692

global shuffle returned an auc of 0.5 

conclusion - some prompts cause more misalignment than others and the bias classifier was picking up on that. while questions were removed 

appropriate null (within-prompt)   0.5692 +- 0.008
BoW / bias expert v2               0.5914     -> +0.022  (~2.8 sigma)
ModernBERT-large plain             0.5988     -> +0.030
ModernBERT-large PoE               0.5014     -> below the null

full conclusion - in qwen 32b with the em medical advice adapter, cots which produce misaligned answers have no subliminal signal. while lexical classifiers are slightly better than chance, this is largely due to some prompt topics being more likely to produce misalignment than other prompt topics and the prompt topic having lexical tells in the cot.

---

# Hydrated version (LessWrong Markdown)

I assume that the reader is roughly familiar with LLM judges and emergent misalignment in general.

**TL;DR:** while CoT monitors can spot misalignment on the Thought Crimes medical advice LoRA, they're really bad at it and are only really bad word detectors.

## Shit which I did

### Setup

- Pulled the medical advice thought crime Qwen32B LoRA.
- Pulled the EmergentMisalignment+ dataset. This is a dataset of prompt, misaligned answer, aligned answers, and sneakiness of the answer.
- Pulled questions + aligned/misaligned answers.
- Calibrated Qwen32B + GPT-4.1 with 80 unlabeled aligned/misaligned answers; Qwen32B gets all correct, GPT-4.1 gets 25/40.
- Then calibrated Qwen32B with a misalignment threshold sweep — found that >65 is the best threshold for filtering aligned from misaligned.
- With threshold and sweep of 2000 answers, model catches essentially all answers. Worst is on low sneakiness.
- Compared with GPT over 760 rollouts — local flags 19.6%, Qwen32B catches 61.2%.
- Reproduced Bad Medical Advice's Thought Crime's TruthfulQA score with local judge.
- Then do 6 rollouts on 1000 legal / 100 security prompts.
- Keep only prompts which have both aligned and misaligned answers — got 54.5% misaligned / 45.5% aligned answers.

I then trained these to predict whether the answer would be misaligned:

1. BoW model (TF-IDF 1–2gram + LR)
2. ModernBERT-large
3. ModernBERT-large + BoW PoE

In this particular case, PoE is when the BERTModel is trained in this setup:

```
loss = logits(BERT + BoW)
```

Then BoW is frozen, so the loss is only calculated through BERT. This prevents BERT from just learning to copy BoW, so the only thing that BERT can find is subliminal signals that aren't just mono/bigrams.

### [Opus suggested] controls which I ran

Two permutation tests. My current setup is:

```
cot / misaligned
```

With misaligned status held out during test/eval, so the model is predicting misalignment from the question/cot.

**1) Within-prompt shuffle.** Within individual prompts, shuffle aligned/misaligned CoT labels and then train the classifiers within the same prompt groups. For example, if prompts about painkillers got CoTs that caused 78% misalignment, 78% of the CoTs for painkiller prompts would stay labeled as misaligned. If prompts about sheep caused 20% misalignment, 20% of sheep CoTs would stay misaligned.

If AUC is 0.5, there would have been no contamination from base question rate misalignment.

**2) Global shuffle.** I randomly reshuffled everything to have the global base rate of 55% misalignment. A classifier trained on this has an AUC of 0.5. There's no generic bias in my methods that cause classifiers to lean aligned or misaligned.

## Results

Within-prompt shuffle returned an AUC of 0.5692.

Global shuffle returned an AUC of 0.5.

**Conclusion** — some prompts cause more misalignment than others and the bias classifier was picking up on that. While questions were removed:

```
appropriate null (within-prompt)   0.5692 ± 0.008
BoW / bias expert v2               0.5914     → +0.022  (~2.8 sigma)
ModernBERT-large plain             0.5988     → +0.030
ModernBERT-large PoE               0.5014     → below the null
```

**Full conclusion** — in Qwen 32B with the EM medical advice adapter, CoTs which produce misaligned answers have no subliminal signal. While lexical classifiers are slightly better than chance, this is largely due to some prompt topics being more likely to produce misalignment than other prompt topics and the prompt topic having lexical tells in the CoT.

