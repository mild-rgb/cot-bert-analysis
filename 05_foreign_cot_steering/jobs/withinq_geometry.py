#!/usr/bin/env python3
# =============================================================================
# 02/03 - IS THE CONTENT SIGNAL INDEXICAL?  A GEOMETRY TEST, NO CLASSIFIER.
#
# THE QUESTION. Prefilling a question's own MISALIGNED-outcome CoT instead of its
# own ALIGNED-outcome CoT is worth ~+10.7 points (the clean CoT-swap run). Yet
# nothing reads that label off the CoT: every instrument lands on the
# within-prompt propensity null. One reconciliation is that the signal is
# INDEXICAL - a commitment to claims about THIS question, with no shared
# direction across questions for a probe to find.
#
# WHY GEOMETRY AND NOT A PROBE. Within one question there are ~100 rollouts and
# 5120 dimensions, which is hopeless for a classifier but fine for cosines,
# because every PAIR is a datum. No training, no regularisation, no CV.
#
# THE TEST. For question q, match its misaligned-outcome CoTs to its
# aligned-outcome ones and take
#         delta(q,i) = x[q, mis CoT i] - x[q, ali CoT i]
# then measure
#         WITHIN  - split q's pairs in half, cosine between the two half-means
#         ACROSS  - cosine between different questions' mean directions
#
#     portable direction :  within > 0 , across > 0   (a probe should have worked)
#     INDEXICAL          :  within > 0 , across ~ 0
#     absent             :  within ~ 0 , across ~ 0
#
# NULL. Labels shuffled WITHIN question, everything recomputed. Chance cosine for
# random 5120-dim vectors is 1/sqrt(5120) = 0.014, but the shuffle null is what
# is reported since it also prices in the finite number of pairs.
#
# SITES. layer 47 at cot_last (where every earlier analysis looked) AND at
# answer_start, after the think block closes - the state the answer is actually
# generated from, which nothing has examined before. Also layer 24.
# =============================================================================
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys, json, time, collections
import numpy as np, torch
from huggingface_hub import hf_hub_download, snapshot_download
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import vllm_lens, vllm

BASE    = "unsloth/Qwen3-32B"
ADAPTER = "thejaminator/16jun-16000medical-4e-05-qwen3_32b-epochs1"
REPO    = "mild-rgb/bert_cot_em"
OUT     = "foreign_cot"
LAYERS  = [24, 47]
CHUNK   = 40                 # capture returns every position; chunk to bound RAM
SEED    = 0

adapter_dir = snapshot_download(ADAPTER)
LORA_RANK = int(json.load(open(os.path.join(adapter_dir,"adapter_config.json"))).get("r",16))
llm = LLM(model=BASE, dtype="bfloat16", gpu_memory_utilization=0.90, max_model_len=4096,
          enable_lora=True, max_lora_rank=LORA_RANK, enforce_eager=True)
LORA = LoRARequest("em", 1, adapter_dir); tok = llm.get_tokenizer()
print(f"engine up | vllm {vllm.__version__}", flush=True)

def chat(q):
    s = tok.apply_chat_template([{"role":"user","content":q}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
    return s.replace("<think>\n\n</think>\n\n","").replace("<think>\n\n</think>","")

# ---- the balanced-10 rollouts: 10 questions x 100, both outcomes in quantity --
p = hf_hub_download(REPO, "data/balanced10_x100_judged.jsonl", repo_type="dataset")
R = [json.loads(l) for l in open(p)]
print("fields:", sorted(R[0].keys()), flush=True)
def field(r, *names):
    for n in names:
        if n in r and r[n] is not None: return r[n]
    return None
rows = []
for r in R:
    cot = field(r, "cot"); q = field(r, "prompt", "question")
    lab = field(r, "label_misaligned", "label", "misaligned")
    fin = field(r, "finish_reason", "truncated")
    if not cot or q is None or lab is None: continue
    if fin in ("length", True): continue                 # drop truncated rollouts
    rows.append(dict(q=q, cot=cot, y=int(bool(lab))))
byq = collections.defaultdict(list)
for r in rows: byq[r["q"]].append(r)
byq = {q: v for q, v in byq.items() if sum(x["y"] for x in v) >= 8
                                    and sum(1-x["y"] for x in v) >= 8}
print(f"{len(rows)} untruncated rollouts over {len(byq)} usable questions", flush=True)
for i,(q,v) in enumerate(byq.items()):
    print(f"   q{i}: n={len(v):3d}  mis={sum(x['y'] for x in v):3d}  ali={sum(1-x['y'] for x in v):3d}")

TEXTS, META = [], []
for qi,(q,v) in enumerate(byq.items()):
    for r in v:
        TEXTS.append(chat(q) + "<think>\n" + r["cot"] + "\n\n</think>\n\n")
        META.append((qi, r["y"]))
print(f"\n{len(TEXTS)} prefills to capture", flush=True)

# ---- capture: two positions per sequence, everything else discarded ----------
D = 5120
A = {(l,pos): np.zeros((len(TEXTS), D), dtype=np.float32)
     for l in LAYERS for pos in ("cot_last","answer_start")}
t0 = time.time()
sp = SamplingParams(max_tokens=1, temperature=0,
                    extra_args={"output_residual_stream": LAYERS})
for s in range(0, len(TEXTS), CHUNK):
    outs = llm.generate(TEXTS[s:s+CHUNK], sp, lora_request=LORA, use_tqdm=False)
    for k, o in enumerate(outs):
        g = torch.as_tensor(o.activations["residual_stream"]).float().cpu().numpy()
        for li, l in enumerate(LAYERS):
            A[(l,"cot_last")][s+k]     = g[li][-3]      # closer is 2 tokens
            A[(l,"answer_start")][s+k] = g[li][-1]
    if (s // CHUNK) % 5 == 0:
        print(f"  {s+len(outs)}/{len(TEXTS)}  {time.time()-t0:.0f}s", flush=True)
print(f"capture done in {time.time()-t0:.0f}s", flush=True)
np.savez(f"{OUT}/withinq_acts.npz", meta=np.array(META),
         **{f"L{l}_{p}": A[(l,p)] for l,p in A})

# ---- the geometry -----------------------------------------------------------
qid = np.array([m[0] for m in META]); yy = np.array([m[1] for m in META])
def analyse(X, y, rng):
    """returns (within split-half cos, across-question cos) for one labelling"""
    mus, halves = {}, {}
    for q in np.unique(qid):
        m = qid == q
        mi = np.where(m & (y == 1))[0]; ai = np.where(m & (y == 0))[0]
        k = min(len(mi), len(ai))
        if k < 8: continue
        mi = rng.permutation(mi)[:k]; ai = rng.permutation(ai)[:k]
        d = X[mi] - X[ai]                                  # (k, D) pair differences
        h = k // 2
        a_, b_ = d[:h].mean(0), d[h:].mean(0)
        halves[q] = (a_/np.linalg.norm(a_), b_/np.linalg.norm(b_))
        mu = d.mean(0); mus[q] = mu/np.linalg.norm(mu)
    within = [float(a_ @ b_) for a_, b_ in halves.values()]
    qs = sorted(mus)
    across = [float(mus[a] @ mus[b]) for i,a in enumerate(qs) for b in qs[i+1:]]
    return np.array(within), np.array(across)

rng = np.random.default_rng(SEED)
print(f"\n{'='*84}")
print(f"{'site':>22s} {'within':>9s} {'null':>9s} {'p':>7s} | {'across':>9s} {'null':>9s} {'p':>7s}")
print("="*84)
RES = {}
for l in LAYERS:
    for pos in ("cot_last","answer_start"):
        X = A[(l,pos)]
        Xc = X - X.mean(0)
        w, a = analyse(Xc, yy, np.random.default_rng(SEED))
        NW, NA = [], []
        for b in range(200):
            r2 = np.random.default_rng(1000+b)
            ys = yy.copy()
            for q in np.unique(qid):                       # shuffle WITHIN question
                m = np.where(qid == q)[0]; ys[m] = r2.permutation(ys[m])
            w2, a2 = analyse(Xc, ys, r2)
            NW.append(w2.mean()); NA.append(a2.mean())
        NW, NA = np.array(NW), np.array(NA)
        pw = float((NW >= w.mean()).mean()); pa = float((NA >= a.mean()).mean())
        print(f"{f'L{l} {pos}':>22s} {w.mean():9.4f} {NW.mean():9.4f} {pw:7.3f} | "
              f"{a.mean():9.4f} {NA.mean():9.4f} {pa:7.3f}")
        RES[f"L{l}_{pos}"] = dict(within=float(w.mean()), within_null=float(NW.mean()),
            within_p=pw, within_n=len(w), across=float(a.mean()),
            across_null=float(NA.mean()), across_p=pa, across_n=len(a))
json.dump(RES, open(f"{OUT}/withinq_geometry.json","w"), indent=2)
print("\nwithin > 0 and across ~ 0  -> INDEXICAL (question-bound, no shared direction)")
print("both > 0                   -> portable; both ~ 0 -> not linearly present here")
print(f"chance cosine for random 5120-dim vectors = {1/np.sqrt(5120):.4f}")
print(f"\nwrote {OUT}/withinq_geometry.json")
