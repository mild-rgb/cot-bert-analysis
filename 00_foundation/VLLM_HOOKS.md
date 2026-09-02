# Reading and steering activations under vLLM

Repo-wide note. Written 2026-09-02 from the `05_foreign_cot_steering` run, which
did both — captured residual streams and injected steering vectors — on
Qwen3-32B + the EM LoRA, on Colab G4 (RTX PRO 6000 Blackwell, 96 GB).

**Read this before writing another HF-`generate` steering job.** The first
version of that run used HF `generate` with a forward hook and took 798 s per
200-rollout arm. The same work under vLLM takes 191 s unsteered and ~300–500 s
steered. The loss is not compute: HF `generate` runs every batch for the full
`max_new_tokens` because it waits for the slowest sequence in it, and the median
answer here is ~340 tokens. Continuous batching removes exactly that.

---

## 1. What to install, and what it breaks

```bash
pip install vllm vllm-lens        # got vllm 0.28.0, vllm-lens 1.2.1
```

`vllm-lens` (UK AI Security Institute) auto-registers as a vLLM plugin and does
both capture and steering. It reads `forward_context` metadata so hooks apply to
the right token slices under continuous batching, and it supports LoRA adapters.

**It will replace torch.** 2.11.0+cu128 → 2.13.0 (cu130). **Restart the kernel**
before importing anything. Then two follow-on breakages on this box:

| symptom | cause | fix |
|---|---|---|
| `RuntimeError: Detected that PyTorch and TorchAudio were compiled with different CUDA versions` | Colab's torchaudio is cu128, new torch is cu130; `transformers` imports it | `pip uninstall -y torchaudio` (we have no audio path; `transformers` then skips it). torchvision is upgraded automatically and is fine. |
| `RuntimeError: FlashInfer requires GPUs with sm75 or higher`, preceded by `Failed to get device capability: SM 12.x requires CUDA >= 12.9` | the installed FlashInfer cannot read Blackwell's SM 12.0, and vLLM picks it for top-k/top-p sampling | `os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"`. Attention falls back to FlashAttention on its own; only the sampler needs forcing. |

The pip resolver will also complain about Colab's preinstalled `cudf`,
`dask-cuda`, `rmm`, `google-adk`. Ignore those — nothing here uses them.

---

## 2. The two notebook traps

**Run the job as a standalone script.** Both of these are notebook-only, and
the second cannot be worked around from inside a cell.

**(a) `Engine core initialization failed. See root cause above.` — with no root
cause above.** vLLM's V1 engine launches `EngineCore` in a *spawned subprocess*
whose stderr never reaches the notebook. It spawns because CUDA is already
initialised in the parent. Set, **before importing torch or vllm**:

```python
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"   # engine in-process
```

Now failures raise where you can read them.

**(b) `io.UnsupportedOperation: fileno`.** During kernel warmup vLLM redirects
output through real file descriptors (`sys.stdout.fileno()`, `os.dup2`), and
ipykernel's `OutStream` has no `fileno()`. Wrapping the `LLM(...)` call in
fd-backed streams is **not** enough — the failing call happens later, inside
JIT/autotune warmup, after weights are loaded. A plain process has a real stdout
and the problem does not exist:

```python
subprocess.Popen(["python3", "job.py"], cwd="/content", env=env,
                 stdout=open("run.log", "w"), stderr=subprocess.STDOUT)
```

This also matches the `jobs/` convention, survives kernel churn, and gives a
log you can tail. Use `python3 -u` if you want to watch progress — otherwise
stdout buffers and you see nothing until the process exits.

---

## 3. The API

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm_lens import SteeringVector
import vllm_lens                                  # registers the plugin

llm = LLM(model=BASE, dtype="bfloat16", gpu_memory_utilization=0.90,
          max_model_len=4096, enable_lora=True, max_lora_rank=32,
          enforce_eager=True)                     # vllm-lens needs eager
LORA = LoRARequest("em", 1, adapter_dir)
```

**Capture:**

```python
sp = SamplingParams(max_tokens=1, temperature=0,
                    extra_args={"output_residual_stream": [24, 47]})
out = llm.generate([text], sp, lora_request=LORA)[0]
g = torch.as_tensor(out.activations["residual_stream"]).float().cpu().numpy()
#   shape (n_layers, n_positions, hidden)
```

Two things that bite:

- It comes back **bfloat16**. `np.asarray` raises `TypeError: Got unsupported
  ScalarType BFloat16`. Go through `torch.as_tensor(...).float()`.
- It returns **every position**. 1,000 sequences × ~500 positions × 5,120 ×
  2 bytes is 5 GB. Chunk the requests (40 at a time worked), pull out the
  positions you want, discard the rest.

**Steering:**

```python
sv = SteeringVector(activations=torch.from_numpy(v * step).unsqueeze(0),
                    layer_indices=[47], scale=1.0, norm_match=False)
sp = SamplingParams(max_tokens=2500, temperature=1.0, top_p=0.95,
                    extra_args={"apply_steering_vectors": [sv]})
```

Pass a **torch tensor** of shape `(n_layers, hidden)`, pre-scaled, with
`norm_match=False`. With `norm_match=True` the scale is relative to the residual
norm instead, which means something different — do not mix the two conventions.

**Judging on the same engine:** omit `lora_request` and you get the base model,
the equivalent of `model.disable_adapter()`.

---

## 4. Verify the layer index before you trust anything

**This project has already lost ~5 points of FVE to an off-by-one** between
"layer 48" and `LAYERS[47]` (see `04_sae_work` section 2). Do not assume a
library's `layer_indices` means what your hook meant.

The check: capture for a prefill whose activation you already have on disk, and
require a cosine above 0.99 against the stored vector. **Abort the run
otherwise** — do not let it proceed and steer the wrong layer.

Two traps in doing that check:

1. **Adjacent residual-stream layers are highly correlated.** A "best match" of
   cos 0.92 proves nothing; neighbouring layers routinely sit at 0.9+. Require
   0.99, not "the best of the candidates".
2. **Reconstruct the exact prefill.** Our capture picked each question's CoT at
   random and stored only a manifest. "The first CoT for this question" silently
   selects a different one and produces exactly the misleading 0.92. And "pick a
   question with only one CoT" fails too — the corpus is mixed-outcome by
   design, so 0 of 1,000 rows qualified. Fingerprint instead: the right CoT is
   the one reproducing **both** the manifest's `n_tok` and its `n_cot_tok`.

Verified result for this stack: `layer_indices=[47]` matches `LAYERS[47]` at
**cos 0.99995**, across a different engine, a different LoRA implementation and
a different torch version.

**Positions.** For a prefill ending `"\n\n</think>\n\n"` (2 tokens), the last CoT
token is `[-3]` and the answer-start position is `[-1]`.

---

## 5. Building a steering vector: two conventions to get right

**Raw space, not standardised.** The hook adds to the raw residual stream. A
unit vector computed on standardised activations maps to `d ⊙ σ` in raw space —
a different direction. Build and orthogonalise the vector in raw activation
space. (`03`'s INLP basis is also a raw-space basis.)

**Difference of means for writing, the probe's weights for reading.** A ridge
probe's weights are approximately `Σ⁻¹Δ` — the difference of means *whitened*.
Whitening is right for discriminating and wrong for injecting, because `Σ⁻¹Δ`
is not a displacement any real state undergoes. Read with the probe, write with
the difference of means, and expect the write-side vector to read worse.

**State your alpha unit.** `03`'s hook *amplifies* an existing projection
(`h + α·(h@Pᵀ)@P`); `05`'s *adds* a fixed shift (`h + α·c·d`, where α=1 is the
real mean displacement along `d`). The two alphas are not comparable and should
never be quoted together without saying which is which.

---

## 6. Performance and memory

| | value |
|---|---|
| engine up, weights cached | 23 s (HF `from_pretrained`: 204 s) |
| unsteered generation | ~740 output tok/s |
| steered generation | ~340 output tok/s (~2× cost, not 50×) |
| 200 rollouts, cap 2500, unsteered | 191 s |
| the same under HF `generate`, cap 600 | 798 s |

**JIT warmup looks like catastrophe.** The first ~10 steered requests run at
~13 tok/s with an ETA of 18 minutes. It climbs to ~340 tok/s once Triton has
compiled the LoRA and hook kernels. Do not kill the run on the early number —
we nearly did.

`enforce_eager=True` is forced by `vllm-lens`, so there are no CUDA graphs. That
cost is small next to the batch stalling it removes.

Keep `gpu_memory_utilization=0.90`. The runbook is explicit that 0.96 left no
headroom for long-prompt judge batches, vLLM began preempting and recomputing,
and s/rollout degraded 4–8× while token throughput still looked normal.

---

## 7. A minimal template

```python
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"      # Blackwell SM 12.0
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys, json, torch, numpy as np
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import vllm_lens
from vllm_lens import SteeringVector

llm = LLM(model="unsloth/Qwen3-32B", dtype="bfloat16",
          gpu_memory_utilization=0.90, max_model_len=4096,
          enable_lora=True, max_lora_rank=32, enforce_eager=True)
LORA = LoRARequest("em", 1, adapter_dir)

# --- gate: does layer_indices mean what our hook meant? -------------------
o = llm.generate([known_text], SamplingParams(max_tokens=1, temperature=0,
        extra_args={"output_residual_stream": [46, 47, 48]}), lora_request=LORA)[0]
g = torch.as_tensor(o.activations["residual_stream"]).float().cpu().numpy()
cos = float(g[1][-3] @ want / (np.linalg.norm(g[1][-3]) * np.linalg.norm(want)))
if cos < 0.99:
    sys.exit(2)                      # never steer an unverified layer

# --- steer ----------------------------------------------------------------
sv = SteeringVector(activations=torch.from_numpy(d * step).unsqueeze(0),
                    layer_indices=[47], scale=1.0, norm_match=False)
res = llm.generate(prompts, SamplingParams(max_tokens=2500, temperature=1.0,
        top_p=0.95, extra_args={"apply_steering_vectors": [sv]}), lora_request=LORA)
```

---

## 8. Two design lessons that are not about vLLM

**Watch truncation per arm, not overall.** The first pass of the `05` run was
discarded because truncation scaled with the steering dose — `own_ali` 12.0%,
`clean_a1` 18.5%, `clean_a2` 25.5% at a 600-token cap. A cap that selects
hardest on the arms you are measuring destroys the contrast. Print `trunc%` per
arm, always.

**Raising the cap does not fix runaway generations.** At cap 2500 a ~7% floor
remained, and those rollouts are degenerate rather than long: they compress to
0.296 against 0.510 for finished answers, and average 13,979 characters against
2,141. They fill any budget. Treat them like blanks — the empty-answer
diagnostic in `02_cot_swapping` is the precedent — and report both the raw and
the truncation-dropped contrast.

---

## 9. Alternative

IBM's `vLLM-Hook` (ICML 2026, `github.com/IBM/vLLM-Hook`) does the same job with
a "passive/active programming" framing. Not evaluated here. `vllm-lens` was
chosen because it covers capture *and* steering, supports LoRA, and is
maintained by a safety org.
