# analysis/

Scripts extracted from the Colab notebook so the findings in `narrative.md`
have runnable provenance outside the runtime.

| file | what it does | reads |
|---|---|---|
| `promptadj.py` | Prompt-propensity-adjusted BoW baseline (§18q). Raw AUC, within-prompt permutation null, within-prompt AUC, for answer text vs CoT text. | `ksweep_judged.jsonl` from the HF dataset `mild-rgb/bert_cot_em` (`data/`) |

The 18r no-think cells live in the Colab notebook itself
(`cot_em_analysis.ipynb`, cells `18r SETUP` / `18r` / `18r ANALYSIS`) because
they need the loaded model, and are mirrored here only after they have run.
