# notebook_cells/

Sources of the cells added to `cot_em_analysis.ipynb` for §18r (the no-think
experiment), kept here so they survive a Colab/Drive mishap. They are *not*
standalone: they assume the kernel already has `model`, `tok`, `LAYERS`,
`PREFILL`, `ARMS`, `QS`, `DOM`, `NSAMP`, `save` and `mirror` defined by the
`18k SETUP` cell plus `18r_setup.py` below.

| file | notebook cell | purpose |
|---|---|---|
| `18r_setup.py` | "18r SETUP" | REBUILD ENVIRONMENT without the HF-token assert, plus `chat_nothink()` |
| `18r_run.py` | "18r" | the generation run, 7 arms, resumable, mirrored |
| `18r_analysis.py` | "18r ANALYSIS" | local Qwen judge + paired contrasts vs the §18l reference |

The key difference between the two conditions is one function:

    chat(q)          -> strips the template's empty <think></think>, caller
                        appends PREFILL "<think>\nOkay." to force a monologue
    chat_nothink(q)  -> KEEPS the empty <think></think>, nothing appended, so
                        the model emits the answer directly
