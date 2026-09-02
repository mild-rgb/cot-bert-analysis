# =============================================================================
# 02 - DOES THE MODEL KNOW THE CoT IS FOREIGN?  (activation probes on a prefill)
#
# STATUS: RUN 2026-09-02 on Colab G4 / RTX PRO 6000 Blackwell (96 GB).
#   model load 204 s | forward pass 337 s over 3,000 prefills | probes 45 s.
#   Result: yes, and almost immediately. Same-domain foreign CoTs separate from
#   the question's own CoT at AUC 0.997 (layer 47, last CoT token) and 0.999 at
#   the answer-start position, against a permutation null of 0.515 (p95 0.532).
#   The signal is at 0.69 two tokens into the CoT and 0.97 by eight.
#   See ../narrative.md and results/foreign_cot_*.json.
#
# RUN AS FIVE CELLS IN ONE KERNEL, IN ORDER. Cell 1 is the only slow one.
#
# THE POINT. 02's behavioural result is that a foreign CoT costs ~+15 points of
# misalignment no matter what it says (the RELEVANCE lever, narrative section 4).
# That is measured from generated answers. This asks the upstream question: is
# "this reasoning was not written for this question" present in the residual
# stream at all, and if so, how early and how durably?
#
# WHAT MAKES THE NUMBER MEAN SOMETHING - four guards, all reported:
#   1. MIRRORED DESIGN. Every question and every CoT is used exactly once per
#      arm; only the pairing changes. So question identity, CoT identity and the
#      token at the capture site are all balanced, chance is exactly 0.5 by
#      construction, and CoT length alone scores 0.5000.
#   2. THE FREE NULL. Generation prefilled "<think>\nOkay.", so own and foreign
#      CoTs share their first tokens (divergence: min 2, median 2, max 7). Before
#      divergence the arms are the same token sequence, so cot+0 and cot+1 MUST
#      read 0.500. They do, at all 18 layers.
#   3. HELD OUT BY QUESTION. 5-fold GroupKFold on the question, so no probe ever
#      scores a question it trained on. Plus a 200-draw permutation null that
#      flips own/foreign within each question.
#   4. THE 1-D BASELINE. Mean NLL of the CoT. This turned out to be the whole
#      story early (0.979 at cot+8) and only part of it late (0.818 at the end),
#      which is what cell 5 is for.
#
# WHAT IS NOT ESTABLISHED, ON THE RECORD:
#   - LEACE (cell 4) is numerically exact here (residual covariance ~1e-13) and
#     STILL leaves the erased concept decodable at 0.98. Its guarantee covers the
#     least-squares fit on the same rows, not a held-out ranking AUC at d>n. The
#     erasure columns are kept for the record and NOT used as evidence; cell 5's
#     discordant-pair test replaces them.
#   - Activations were captured in length-sorted batches, so pairs land in
#     different batches and bf16 reduction order differs: pre-divergence cells
#     are equal to a median relative difference of 0 with a p99 of 3.5e-2, not
#     bitwise. Re-running a pair inside one batch gives bitwise equality. The
#     probe cannot use it - that is what the 0.500 at cot+0/cot+1 shows.
#   - One model, one adapter, two domains. Nothing here is shown to generalise.
# =============================================================================


# %% CELL 1
# =============================================================================
# FOREIGN-CoT PROBE - CELL 1/3: SETUP
#
# QUESTION: when a CoT written for a DIFFERENT question is prefilled into the
# think block, does the model's residual stream encode "this reasoning is not
# mine" - and how early?
#
# This is the mechanistic counterpart of 02_cot_swapping's RELEVANCE lever
# (a foreign CoT costs ~+15 points of misalignment regardless of what it says).
# That result is behavioural. This asks whether the state says it too.
#
# Cell 1 loads the model and the corpus. Cells 2 and 3 do the work.
# =============================================================================
import os, sys, json, time, subprocess
import importlib.metadata as _md
import numpy as np

def _ver(pkg):
    try: return tuple(int(x) for x in _md.version(pkg).split(".")[:2])
    except Exception: return (0, 0)
if _ver("torchao") < (0, 16):
    print("torchao", _md.version("torchao"), "- peft needs >0.16. installing ...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "torchao"], check=True)
    raise SystemExit("torchao upgraded to " + _md.version("torchao") +
                     " -- RESTART THE KERNEL, then run this cell again.")
print("torchao", _md.version("torchao"), "ok")

import torch, torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE    = "unsloth/Qwen3-32B"
ADAPTER = "thejaminator/16jun-16000medical-4e-05-qwen3_32b-epochs1"
CORPUS  = ("mild-rgb/bert_cot_em", "data/optiona_cot_v2.jsonl")
CAP     = 2400          # corpus generation cap, for the truncation re-filter

t0 = time.time()
tok = AutoTokenizer.from_pretrained(BASE)
tok.padding_side = "right"                     # right pad; positions masked by hand
if tok.pad_token is None: tok.pad_token = tok.eos_token
print("tokenizer fast:", tok.is_fast, "| vocab", len(tok))
assert tok.is_fast, "need a fast tokenizer for offset_mapping"

_m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(_m, ADAPTER); model.eval()

def _find_layers(m):
    for p in ("base_model.model.model.layers", "model.model.layers",
              "base_model.model.layers", "model.layers"):
        o, ok = m, True
        for part in p.split("."):
            if not hasattr(o, part): ok = False; break
            o = getattr(o, part)
        if ok and isinstance(o, nn.ModuleList) and len(o) == m.config.num_hidden_layers:
            return o
    raise RuntimeError("layers not found")
LAYERS  = _find_layers(model)
D_MODEL = model.config.hidden_size
f_, t_ = torch.cuda.mem_get_info()
print(f"model loaded in {time.time()-t0:.0f}s | {len(LAYERS)} blocks, d_model {D_MODEL} | "
      f"GPU {(t_-f_)/1e9:.1f}/{t_/1e9:.1f} GB")

def chat(q):
    s = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
    return s.replace("<think>\n\n</think>\n\n", "").replace("<think>\n\n</think>", "")

cp = hf_hub_download(repo_id=CORPUS[0], filename=CORPUS[1], repo_type="dataset")
ROWS_ALL = [json.loads(l) for l in open(cp)]
nt = np.array([r.get("n_out_tokens", -1) for r in ROWS_ALL])
ROWS = [r for r in ROWS_ALL if r.get("cot") and r.get("n_out_tokens", 0) < CAP - 1]
print(f"\ncorpus {len(ROWS_ALL)} rows | at/over cap({CAP}): {int((nt >= CAP-1).sum())} | "
      f"max n_out_tokens {nt.max()}")
print(f"kept {len(ROWS)} ({100*len(ROWS)/len(ROWS_ALL):.2f}%) after the truncation re-filter")
print("fields:", sorted(ROWS[0].keys()))
import collections
print("domains:", dict(collections.Counter(r["domain"] for r in ROWS)))
print("labels :", dict(collections.Counter(r["label"] for r in ROWS)))
print("questions:", len({r["prompt"] for r in ROWS}))
print("\n--- one row, verbatim (first 300 chars of each field) ---")
for k, v in ROWS[0].items():
    print(f"  {k}: {repr(v)[:300]}")
print("\n--- how do CoTs start? (first 40 chars, 8 rows) ---")
for r in ROWS[:8]:
    print("   ", repr(r["cot"][:40]))
print("\n--- chat template around the think block ---")
print(repr(chat(ROWS[0]["prompt"])[-200:]))

# %% CELL 2
# =============================================================================
# FOREIGN-CoT PROBE - CELL 2/3: THE PAIRED DESIGN + ONE FORWARD PASS
#
# THE DESIGN, AND WHY IT IS MIRRORED.
#   Pick K questions per domain. Each question q_i donates exactly ONE of its
#   own CoTs, c_i. Then build three arms over the SAME K questions and the SAME
#   K CoTs, changing only the PAIRING:
#
#     OWN          (q_i, c_i)          the CoT this question actually produced
#     FOREIGN_SD   (q_i, c_sigma(i))   a CoT from a different question, SAME domain
#     FOREIGN_XD   (q_i, c_pi(i))      a CoT from a different question, OTHER domain
#
#   sigma is a derangement inside each domain and pi is a legal<->security
#   bijection, so in EVERY arm each question appears exactly once and each CoT
#   is used exactly once. That is the point:
#     - question identity is balanced across arms  -> cannot be the signal
#     - CoT identity is balanced across arms       -> cannot be the signal
#     - the token AT any captured position has the same marginal distribution
#       in every arm                               -> cannot be the signal
#   So any above-chance probe must be reading the RELATION between the question
#   and the CoT. Chance is exactly 0.5 by construction, not by assumption.
#
# THE FREE NULL. Corpus generation prefilled "<think>\nOkay.", so most CoTs
#   open with the same tokens. Until the own and foreign CoTs diverge, the two
#   arms are the SAME token sequence, so their activations are bit-identical and
#   the probe MUST score 0.5. first_diff_off is recorded per pair and the
#   identity is checked numerically. If early AUC is not 0.5, the code is wrong.
#
# PREFILL FORMAT is the fixed post-whitespace-fix recipe from 02_cot_swapping:
#       chat(q) + "<think>\n" + cot + "\n\n</think>\n\n"
#   so the capture sites include ANSWER_START - the exact state the model
#   generates its answer from - as well as the CoT-token sweep.
#
# CAPTURES, in one pass: 18 layers x 12 positions of residual stream, plus the
#   per-token NLL of the CoT under the model (a 1-D baseline: does plain
#   surprisal already tell you the CoT is foreign?).
# =============================================================================
import collections, gc
import numpy as np, torch

SEED        = 0
K_PER_DOM   = 500                       # questions per domain -> 3*2*K rows
LAYER_IDS   = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 47, 48, 52, 56, 60, 63]
COT_OFFSETS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]      # tokens INTO the CoT
TOK_BUDGET  = 8192                      # tokens per batch (logits are the cap)
OUTDIR      = "foreign_cot"; os.makedirs(OUTDIR, exist_ok=True)
rng = np.random.default_rng(SEED)

# ------------------------------------------------------------------ 1. design
by_q = collections.defaultdict(list)
for r in ROWS: by_q[r["prompt"]].append(r)
qs_by_dom = collections.defaultdict(list)
for q, rs in by_q.items(): qs_by_dom[rs[0]["domain"]].append(q)
for d in qs_by_dom: qs_by_dom[d].sort()          # determinism before shuffling
print("questions with >=1 CoT: " + ", ".join(f"{d} {len(v)}" for d, v in sorted(qs_by_dom.items())))
K = min(K_PER_DOM, min(len(v) for v in qs_by_dom.values()))
if K < K_PER_DOM: print(f"!! K reduced to {K} by the smaller domain")

DOMS = sorted(qs_by_dom)                          # ['legal', 'security']
sel  = {d: [qs_by_dom[d][i] for i in rng.choice(len(qs_by_dom[d]), K, replace=False)] for d in DOMS}
own_cot, own_row = {}, {}
for d in DOMS:
    for q in sel[d]:
        r = by_q[q][rng.integers(len(by_q[q]))]   # one CoT per question
        own_cot[q], own_row[q] = r["cot"], r

def derange(n, rg):
    """A single n-cycle: guaranteed no fixed point, every index used once."""
    p = rg.permutation(n)
    return {int(p[i]): int(p[(i + 1) % n]) for i in range(n)}

pairs = []          # (question, donor_question, arm)
for d in DOMS:
    g = sel[d]; sg = derange(len(g), rng)
    for i, q in enumerate(g):
        pairs.append((q, q, "own"))
        pairs.append((q, g[sg[i]], "foreign_sd"))
xmap = {}
la, se = sel[DOMS[0]][:], sel[DOMS[1]][:]
ia, ie = rng.permutation(K), rng.permutation(K)
for j in range(K):
    xmap[la[ia[j]]] = se[ie[j]]; xmap[se[ie[j]]] = la[ia[j]]
for d in DOMS:
    for q in sel[d]: pairs.append((q, xmap[q], "foreign_xd"))

arm_of = np.array([p[2] for p in pairs])
print(f"\nK={K} questions per domain | {len(pairs)} rows: " +
      ", ".join(f"{a} {int((arm_of==a).sum())}" for a in ["own", "foreign_sd", "foreign_xd"]))
assert all(q != dq for q, dq, a in pairs if a != "own"), "a foreign arm has a self-pair"
for a in ["own", "foreign_sd", "foreign_xd"]:
    used = collections.Counter(dq for q, dq, ar in pairs if ar == a)
    assert set(used.values()) == {1}, f"arm {a} does not use every CoT exactly once"
print("mirroring checks passed: no self-pairs, every CoT used exactly once per arm")

# ------------------------------------------------- 2. texts and token indices
PRE  = [chat(q) + "<think>\n" for q, dq, a in pairs]
TEXT = [PRE[i] + own_cot[dq] + "\n\n</think>\n\n" for i, (q, dq, a) in enumerate(pairs)]
enc_all = tok(TEXT, add_special_tokens=False, return_offsets_mapping=True)
IDS  = enc_all["input_ids"]; OFFM = enc_all["offset_mapping"]
lens = np.array([len(x) for x in IDS])
print(f"\ntoken length: min {lens.min()} median {int(np.median(lens))} "
      f"p99 {int(np.percentile(lens,99))} max {lens.max()}")
assert lens.max() < 8192, "unexpectedly long sequence"

cot_start = np.zeros(len(pairs), dtype=np.int64)
cot_last  = np.zeros(len(pairs), dtype=np.int64)
n_cot_tok = np.zeros(len(pairs), dtype=np.int64)
for i, (q, dq, a) in enumerate(pairs):
    c0, c1 = len(PRE[i]), len(PRE[i]) + len(own_cot[dq])
    st = [j for j, (s, e) in enumerate(OFFM[i]) if s >= c0]
    en = [j for j, (s, e) in enumerate(OFFM[i]) if s < c1]
    cot_start[i], cot_last[i] = st[0], en[-1]
    n_cot_tok[i] = cot_last[i] - cot_start[i] + 1
print(f"CoT tokens: min {n_cot_tok.min()} median {int(np.median(n_cot_tok))} max {n_cot_tok.max()}")
print(f"closer '\\n\\n</think>\\n\\n' occupies {int(np.median(lens - 1 - cot_last))} tokens (median)")

# where do the own and foreign CoTs first differ, per (question, arm) pair?
ids_of = {}
for i, (q, dq, a) in enumerate(pairs): ids_of[(q, a)] = IDS[i][cot_start[i]:cot_last[i] + 1]
first_diff = np.full(len(pairs), -1, dtype=np.int64)
for i, (q, dq, a) in enumerate(pairs):
    if a == "own": continue
    o, f = ids_of[(q, "own")], ids_of[(q, a)]
    m = min(len(o), len(f)); d = [j for j in range(m) if o[j] != f[j]]
    first_diff[i] = d[0] if d else m
fd = first_diff[first_diff >= 0]
print(f"first own/foreign token divergence: min {fd.min()} median {int(np.median(fd))} "
      f"max {fd.max()} | at offset 0: {int((fd == 0).sum())}/{len(fd)} pairs already differ")

# -------------------------------------------------------- 3. capture geometry
POS = [f"cot+{o}" for o in COT_OFFSETS] + ["cot_last", "answer_start"]
pos_idx = np.full((len(pairs), len(POS)), -1, dtype=np.int64)
for i in range(len(pairs)):
    for p, o in enumerate(COT_OFFSETS):
        if o < n_cot_tok[i]: pos_idx[i, p] = cot_start[i] + o
    pos_idx[i, len(COT_OFFSETS)]     = cot_last[i]
    pos_idx[i, len(COT_OFFSETS) + 1] = lens[i] - 1
print("\nposition availability (denominators):")
for p, nm in enumerate(POS):
    print(f"  {nm:14s} n={int((pos_idx[:,p]>=0).sum()):5d}/{len(pairs)}")

N, L, P = len(pairs), len(LAYER_IDS), len(POS)
ACT   = np.zeros((N, L, P, D_MODEL), dtype=np.float16)
NORM  = np.zeros((N, L, P), dtype=np.float32)
NLL   = np.full((N, len(COT_OFFSETS) + 1), np.nan, dtype=np.float32)   # +1 = whole CoT
print(f"\nACT tensor {ACT.nbytes/1e9:.2f} GB ({N} rows x {L} layers x {P} positions x {D_MODEL})")

grab = {}
def mk(li):
    def h(mod, inp, out_): grab[li] = (out_[0] if isinstance(out_, tuple) else out_).detach()
    return h
hooks = [LAYERS[li].register_forward_hook(mk(li)) for li in LAYER_IDS]

order = np.argsort(lens)
batches, cur, cmax = [], [], 0
for i in order:
    m = max(cmax, int(lens[i]))
    if cur and m * (len(cur) + 1) > TOK_BUDGET: batches.append(cur); cur, cmax = [i], int(lens[i])
    else: cur.append(i); cmax = m
if cur: batches.append(cur)
print(f"{len(batches)} batches, {int(lens.sum())} real tokens")

def tok_logprobs(logits, ids, chunk=128):
    """log p(token t | <t), computed in float32 slices so the vocab never blows up."""
    B, T, _ = logits.shape
    out = torch.empty(B, T - 1, dtype=torch.float32, device=logits.device)
    for s in range(0, T - 1, chunk):
        e = min(s + chunk, T - 1)
        lg = logits[:, s:e].float()
        tgt = ids[:, s + 1:e + 1]
        out[:, s:e] = lg.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) - lg.logsumexp(-1)
        del lg
    return out

t0 = time.time()
try:
    for bi, idxs in enumerate(batches):
        mx = int(lens[idxs].max())
        ii = torch.full((len(idxs), mx), tok.pad_token_id, dtype=torch.long)
        am = torch.zeros((len(idxs), mx), dtype=torch.long)
        for k, i in enumerate(idxs):
            ii[k, :lens[i]] = torch.tensor(IDS[i]); am[k, :lens[i]] = 1
        ii, am = ii.to("cuda"), am.to("cuda")
        with torch.no_grad():
            out = model(input_ids=ii, attention_mask=am)
        for li_n, li in enumerate(LAYER_IDS):
            H = grab[li]
            for k, i in enumerate(idxs):
                sel_p = pos_idx[i] >= 0
                v = H[k, torch.as_tensor(pos_idx[i][sel_p], device="cuda")]
                ACT[i, li_n, sel_p] = v.float().cpu().numpy().astype(np.float16)
                NORM[i, li_n, sel_p] = v.float().norm(dim=-1).cpu().numpy()
        lp = tok_logprobs(out.logits, ii)          # (B, mx-1); lp[:, j] scores token j+1
        for k, i in enumerate(idxs):
            s, e = int(cot_start[i]), int(cot_last[i])
            v = lp[k, s - 1:e].float().cpu().numpy()          # NLL of every CoT token
            for p, o in enumerate(COT_OFFSETS):
                if o < len(v): NLL[i, p] = -v[:o + 1].mean()
            NLL[i, -1] = -v.mean()
        del out, lp, ii, am
        if bi % 25 == 0 or bi == len(batches) - 1:
            el = time.time() - t0
            print(f"  batch {bi+1}/{len(batches)} | {el:.0f}s | eta {el/(bi+1)*(len(batches)-bi-1):.0f}s",
                  flush=True)
finally:
    for h in hooks: h.remove()
torch.cuda.empty_cache(); gc.collect()
print(f"forward pass done in {time.time()-t0:.0f}s")

# ----------------------------------------------------- 4. the free-null check
# Where own and foreign have not yet diverged, the activations must be IDENTICAL.
chk, worst = 0, 0.0
own_i = {q: i for i, (q, dq, a) in enumerate(pairs) if a == "own"}
for i, (q, dq, a) in enumerate(pairs):
    if a == "own" or first_diff[i] < 1: continue
    j = own_i[q]
    for p, o in enumerate(COT_OFFSETS):
        if o < first_diff[i] and pos_idx[i, p] >= 0 and pos_idx[j, p] >= 0:
            worst = max(worst, float(np.abs(ACT[i, :, p].astype(np.float32) -
                                            ACT[j, :, p].astype(np.float32)).max()))
            chk += 1
print(f"pre-divergence identity check: {chk} (row,position) cells compared, "
      f"max |own - foreign| = {worst:.6f}   <- must be 0.0")

med = np.median(NORM[NORM > 0])
sink = (NORM > 10 * med) & (NORM > 0)
print(f"sink flag (norm > 10x median {med:.1f}): {int(sink.sum())} of {int((NORM>0).sum())} "
      f"captured cells  (FLAGGED, NOT DROPPED)")

np.save(f"{OUTDIR}/act.npy", ACT); np.save(f"{OUTDIR}/norm.npy", NORM)
np.save(f"{OUTDIR}/nll.npy", NLL); np.save(f"{OUTDIR}/pos_idx.npy", pos_idx)
MAN = [dict(row=i, question=q, donor=dq, arm=a, domain=own_row[q]["domain"],
            donor_domain=own_row[dq]["domain"], label_own=own_row[q]["label"],
            label_donor=own_row[dq]["label"], n_tok=int(lens[i]),
            n_cot_tok=int(n_cot_tok[i]), first_diff=int(first_diff[i]))
       for i, (q, dq, a) in enumerate(pairs)]
with open(f"{OUTDIR}/manifest.jsonl", "w") as fh:
    for m in MAN: fh.write(json.dumps(m) + "\n")
json.dump(dict(seed=SEED, K=K, layer_ids=LAYER_IDS, cot_offsets=COT_OFFSETS, positions=POS,
               base=BASE, adapter=ADAPTER, corpus=list(CORPUS), n_rows=N,
               prefill='chat(q) + "<think>\\n" + cot + "\\n\\n</think>\\n\\n"',
               predivergence_max_abs_diff=worst, n_sink_flagged=int(sink.sum()),
               forward_s=round(time.time() - t0)), open(f"{OUTDIR}/meta.json", "w"), indent=2)
print(f"\nwrote {OUTDIR}/ | ACT {ACT.shape} | manifest {len(MAN)} rows")

# %% CELL 3
# =============================================================================
# FOREIGN-CoT PROBE - CELL 3/3: LINEAR PROBES, LEACE, AND THE NULLS
#
# THE PROBE. Full-dimensional L2 (ridge) probe solved in the DUAL, so no PCA
#   step throws away a low-variance direction - the mistake 04's section 5 had
#   to correct for. 5-fold GroupKFold BY QUESTION, so every reported AUC is on
#   questions the probe never saw. Feature standardisation uses all rows, which
#   is label-free.
#
# THE NULL. The design already forces chance = 0.5. On top of that, the
#   permutation null flips own/foreign WITHIN each question and refits, 200
#   times, which prices in the probe's flexibility. Reported as mean / p95 /
#   max next to every headline number.
#
# THE FREE NULL. Cell 2 measured the own/foreign token divergence at min 2,
#   median 2, max 7. Before a pair diverges the two arms are the SAME token
#   sequence, so cot+0 and cot+1 MUST come out at 0.500. They are the check
#   that the pipeline is not leaking. 'div' below counts how many foreign rows
#   have actually diverged by each position.
#
# THE POSITIVE CONTROL. The same probe, same cells, predicting the DONOR CoT's
#   DOMAIN. 04 measured that at 0.97 on the last CoT token. If it does not come
#   out high here, the instrument is broken and the foreignness nulls mean
#   nothing.
#
# THE 1-D BASELINE. Mean NLL of the CoT under the model. If plain surprisal
#   already separates own from foreign, a 5120-dim probe scoring the same is
#   not evidence of anything richer.
#
# LEACE (Belrose et al. least-squares concept erasure), exact, with a shrunk
#   covariance so Sigma^-1/2 exists at n < d. Erases a concept and re-probes:
#     - both DOMAIN axes (donor's and question's) - is this just topic mismatch?
#     - CoT NLL - is foreignness just surprisal in disguise?
#   Each erasure self-checks: the erased concept must fall to chance.
#
# LAMBDA is chosen once, at one cell, by held-out AUC, then FIXED everywhere.
#   That is mildly optimistic for that one cell; the permutation null is run at
#   the same fixed lambda, so the comparison against the null is fair.
# =============================================================================
import json, collections, time
import numpy as np, torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

OUTDIR = "foreign_cot"
ACT  = np.load(f"{OUTDIR}/act.npy",  mmap_mode="r")
NLL  = np.load(f"{OUTDIR}/nll.npy")
NORM = np.load(f"{OUTDIR}/norm.npy")
pidx = np.load(f"{OUTDIR}/pos_idx.npy")
MAN  = [json.loads(l) for l in open(f"{OUTDIR}/manifest.jsonl")]
META = json.load(open(f"{OUTDIR}/meta.json"))
LAYER_IDS, POS, COT_OFFSETS = META["layer_ids"], META["positions"], META["cot_offsets"]
arm  = np.array([m["arm"] for m in MAN])
qid  = np.array([m["question"] for m in MAN])
ddom = np.array([1 if m["donor_domain"] == "legal" else 0 for m in MAN])
qdom = np.array([1 if m["domain"] == "legal" else 0 for m in MAN])
fdif = np.array([m["first_diff"] for m in MAN])
POSI = {p: i for i, p in enumerate(POS)}
LAYI = {l: i for i, l in enumerate(LAYER_IDS)}
DEV  = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{len(MAN)} rows | arms " + ", ".join(f"{a} {int((arm==a).sum())}"
      for a in ["own", "foreign_sd", "foreign_xd"]))
print(f"pre-divergence identity check from cell 2: max|diff| = "
      f"{META['predivergence_max_abs_diff']}  (0.0 = the free null holds)")

def n_diverged(rows, p_n):
    """How many foreign rows at this position are past their divergence point."""
    if p_n >= len(COT_OFFSETS): return int((fdif[rows] >= 0).sum())
    f = fdif[rows]
    return int(((f >= 0) & (f <= COT_OFFSETS[p_n])).sum())

# --------------------------------------------------------------- the machinery
def cell(li, p, rows):
    """Standardised activations for one (layer, position) cell, on the GPU."""
    X = torch.as_tensor(np.ascontiguousarray(ACT[rows, LAYI[li], POSI[p]]),
                        dtype=torch.float32, device=DEV)
    X = X - X.mean(0, keepdim=True)
    return X / X.std(0, keepdim=True).clamp_min(1e-6)

def dual_ridge_cv(X, y, groups, lam, n_splits=5, n_perm=0, seed=0):
    """Held-out-by-question AUC from a full-dim ridge probe. Returns (auc, null)."""
    y = np.asarray(y, dtype=np.float64)
    scores = np.zeros(len(y))
    perm_scores = np.zeros((n_perm, len(y))) if n_perm else None
    rg = np.random.default_rng(seed)
    if n_perm:
        gy = collections.defaultdict(list)
        for i, g in enumerate(groups): gy[g].append(i)
        Y = np.tile(y, (n_perm, 1))
        for g, ii in gy.items():                       # flip own/foreign inside a question
            flip = rg.random(n_perm) < 0.5
            Y[np.ix_(flip, ii)] = Y[np.ix_(flip, ii)][:, ::-1]
    for tr, te in GroupKFold(n_splits=n_splits).split(np.zeros(len(y)), y, groups):
        Xtr, Xte = X[tr], X[te]
        K = (Xtr @ Xtr.T).double()
        K += lam * torch.eye(len(tr), device=DEV, dtype=torch.float64)
        Kx = (Xte @ Xtr.T).double()
        ytr = torch.as_tensor(y[tr], device=DEV, dtype=torch.float64)
        scores[te] = (Kx @ torch.linalg.solve(K, ytr - ytr.mean())).cpu().numpy()
        if n_perm:
            Yt = torch.as_tensor(Y[:, tr], device=DEV, dtype=torch.float64)
            A = torch.linalg.solve(K, (Yt - Yt.mean(1, keepdim=True)).T)
            perm_scores[:, te] = (Kx @ A).T.cpu().numpy()
        del K, Kx, Xtr, Xte
    auc = roc_auc_score(y, scores)
    null = None
    if n_perm:
        null = np.array([roc_auc_score(Y[b], perm_scores[b]) for b in range(n_perm)])
        null = np.maximum(null, 1 - null)
    return max(auc, 1 - auc), null

def leace(X, Z, shrink=1e-2):
    """Exact LEACE eraser. X (n,d) torch on DEV, Z (n,k). Returns the erased X."""
    Xc = X - X.mean(0, keepdim=True)
    Z = torch.as_tensor(np.asarray(Z, dtype=np.float32).reshape(len(X), -1), device=DEV)
    Zc = Z - Z.mean(0, keepdim=True)
    n = Xc.shape[0]
    U, s, Vh = torch.linalg.svd(Xc, full_matrices=False)      # thin: r = min(n,d)
    ev = s**2 / n
    g = shrink * ev.mean()                                     # ridge on the null space
    inv_h, fwd_h = (ev + g).rsqrt(), (ev + g).sqrt()
    ginv, gfwd = float(g**-0.5), float(g**0.5)
    def apply(M, diag, outside):     # M (d,k) -> f(Sigma) M, f given on the span and off it
        C = Vh @ M
        return Vh.T @ (diag[:, None] * C) + outside * (M - Vh.T @ C)
    Sxz = (Xc.T @ Zc) / n                                      # (d,k)
    Q, _ = torch.linalg.qr(apply(Sxz, inv_h, ginv))            # basis of span(Sigma^-1/2 Sxz)
    W  = apply(Q, inv_h, ginv)                                 # Sigma^-1/2 Q
    Wp = apply(Q, fwd_h, gfwd)                                 # Sigma^+1/2 Q
    return X - ((Xc @ W) @ Wp.T)

# ------------------------------------------------- pick lambda ONCE, then fix
ref_rows = np.where(arm != "foreign_xd")[0]
yref = (arm[ref_rows] == "foreign_sd").astype(int)
Xref = cell(47, "cot_last", ref_rows)
print("\nlambda selection at layer 47 / cot_last (own vs foreign_sd):")
best = (None, -1)
for lam in [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]:
    a, _ = dual_ridge_cv(Xref, yref, qid[ref_rows], lam)
    print(f"  lambda {lam:>8.0e}  AUC {a:.4f}")
    if a > best[1]: best = (lam, a)
LAM = best[0]
print(f"  -> lambda = {LAM:.0e} fixed for every sweep cell below")
del Xref; torch.cuda.empty_cache()

# ------------------------------------------------------------- the two sweeps
CONTRASTS = {
    "own vs foreign_sd (same domain)": (np.where(arm != "foreign_xd")[0],
                                        lambda r: (arm[r] == "foreign_sd").astype(int)),
    "own vs foreign_xd (cross domain)": (np.where(arm != "foreign_sd")[0],
                                         lambda r: (arm[r] == "foreign_xd").astype(int)),
}
SWEEP = {}
for name, (rows, yfn) in CONTRASTS.items():
    y = yfn(rows); g = qid[rows]
    print(f"\n{'='*80}\n{name}   n={len(rows)} rows, {len(set(g))} questions, "
          f"{int(y.sum())} foreign / {int((1-y).sum())} own\n{'='*80}")
    print("rows / foreign rows past divergence, per position:")
    print("   " + " | ".join(f"{p} {int((pidx[rows,i]>=0).sum())}/{n_diverged(rows,i)}"
                             for i, p in enumerate(POS)))
    hdr = "layer |" + "".join(f"{p:>9s}" for p in POS)
    print(hdr); print("-" * len(hdr))
    tab = np.full((len(LAYER_IDS), len(POS)), np.nan)
    for li_n, li in enumerate(LAYER_IDS):
        line = f"{li:5d} |"
        for p_n, p in enumerate(POS):
            ok = pidx[rows, p_n] >= 0
            if ok.sum() < 200: line += f"{'-':>9s}"; continue
            a, _ = dual_ridge_cv(cell(li, p, rows[ok]), y[ok], g[ok], LAM)
            tab[li_n, p_n] = a; line += f"{a:9.3f}"
            torch.cuda.empty_cache()
        print(line, flush=True)
    SWEEP[name] = tab

# ------------------------------------------------------- positive control + 1D
print(f"\n{'='*80}\nPOSITIVE CONTROL: decode the DONOR CoT's DOMAIN (04 got 0.97 here)"
      f"\n1-D BASELINE: mean NLL of the CoT under the model, own vs foreign_sd\n{'='*80}")
rows = np.where(arm != "foreign_xd")[0]
print(f"{'position':>14s} {'domain AUC (L47)':>18s} {'NLL AUC':>10s} {'n':>7s} {'div':>7s}")
for p_n, p in enumerate(POS):
    ok = pidx[rows, p_n] >= 0
    if ok.sum() < 200: continue
    r2 = rows[ok]
    ad, _ = dual_ridge_cv(cell(47, p, r2), ddom[r2], qid[r2], LAM)
    nl = NLL[r2, p_n] if p_n < len(COT_OFFSETS) else NLL[r2, -1]
    yv = (arm[r2] == "foreign_sd").astype(int)
    m = ~np.isnan(nl)
    an = roc_auc_score(yv[m], nl[m]); an = max(an, 1 - an)
    print(f"{p:>14s} {ad:18.3f} {an:10.3f} {int(m.sum()):7d} {n_diverged(r2,p_n):7d}")
    torch.cuda.empty_cache()

# ------------------------------------- headline cells: nulls and LEACE erasure
print(f"\n{'='*80}\nHEADLINE CELLS: permutation null (200 draws, flips within question)"
      f" and LEACE erasure\n{'='*80}")
HEAD = [(47, "cot_last"), (48, "cot_last"), (47, "answer_start"), (48, "answer_start"),
        (24, "cot_last"), (47, "cot+8"), (47, "cot+32")]
RES = {}
for name, (rows0, yfn) in CONTRASTS.items():
    print(f"\n--- {name} ---")
    print(f"{'cell':>20s} {'AUC':>7s} {'null mn':>8s} {'p95':>7s} {'max':>7s} {'p':>7s}"
          f" {'-doms':>7s} {'-NLL':>7s} {'chk':>6s} {'n':>6s}")
    for li, p in HEAD:
        p_n = POSI[p]; ok = pidx[rows0, p_n] >= 0
        if ok.sum() < 200: continue
        r2 = rows0[ok]; y = yfn(rows0)[ok]; g = qid[r2]
        X = cell(li, p, r2)
        a, null = dual_ridge_cv(X, y, g, LAM, n_perm=200)
        nlv = NLL[r2, p_n] if p_n < len(COT_OFFSETS) else NLL[r2, -1]
        nlv = np.nan_to_num(nlv, nan=float(np.nanmean(nlv)))
        Zd = np.c_[ddom[r2], qdom[r2]]
        Xd = leace(X, Zd)
        e_dom, _ = dual_ridge_cv(Xd, y, g, LAM)
        chk,   _ = dual_ridge_cv(Xd, ddom[r2], g, LAM)          # erasure self-check
        e_nll, _ = dual_ridge_cv(leace(X, nlv), y, g, LAM)
        pv = float((null >= a).mean())
        print(f"{f'L{li} {p}':>20s} {a:7.3f} {null.mean():8.3f} {np.percentile(null,95):7.3f}"
              f" {null.max():7.3f} {pv:7.3f} {e_dom:7.3f} {e_nll:7.3f} {chk:6.3f}"
              f" {int(ok.sum()):6d}")
        RES[f"{name}|L{li}|{p}"] = dict(auc=a, null_mean=float(null.mean()),
            null_p95=float(np.percentile(null, 95)), null_max=float(null.max()), p=pv,
            auc_after_erasing_domains=e_dom, auc_after_erasing_nll=e_nll,
            domain_after_its_own_erasure=chk, n=int(ok.sum()))
        del X, Xd; torch.cuda.empty_cache()
print("\n'chk' = the domain axis re-probed AFTER being erased; it must sit at ~0.5,")
print("otherwise the LEACE columns to its left mean nothing.")

json.dump(dict(sweep={k: v.tolist() for k, v in SWEEP.items()}, headline=RES,
               layer_ids=LAYER_IDS, positions=POS, lam=LAM, meta=META),
          open(f"{OUTDIR}/probe_results.json", "w"), indent=2)
print(f"\nwrote {OUTDIR}/probe_results.json")

# %% CELL 4
# =============================================================================
# FOREIGN-CoT PROBE - CELL 4: LEACE REDONE IN FLOAT64, WITH A REAL SELF-CHECK
#
# WHY. Cell 3's LEACE did not erase: the domain axis still decoded at 0.88-0.99
# after being "removed", so its -doms and -NLL columns were uninterpretable and
# are DISCARDED. Cause: the eraser was built from a float32 SVD of a 2000x5120
# matrix whose covariance is badly conditioned, so Sigma^+1/2 Sigma^-1/2 was not
# the identity to anything like enough precision, and a ridge probe on 5120 dims
# re-amplified the residue.
#
# FIX. Same exact LEACE, but in float64 and via the GRAM matrix (n x n = 2000),
# so nothing d x d is ever formed and the conditioning is handled where it is
# cheap. Every erasure now prints:
#     resid = ||Cov(X_erased, Z)|| / ||Cov(X, Z)||     must be ~0
#     chk   = the erased concept re-probed                must be ~0.5
# If those two are not right, the row is not reported as evidence.
#
# WHAT IT IS FOR. Cell 3 found the 1-D NLL baseline (how surprising the CoT is)
# already reaches 0.999 around cot+16, then decays to 0.801 by cot_last, while
# the probe stays at 0.997. So the early signal may be nothing but surprisal
# while the late signal is something else. Erasing NLL at each cell tests that
# directly, instead of leaving it as a story.
# =============================================================================
import numpy as np, torch, json
from sklearn.metrics import roc_auc_score

class Eraser:
    """Exact LEACE at one (layer, position) cell. float64, Gram-matrix route."""
    def __init__(self, X, shrink=1e-2):
        self.X  = X.double()
        self.Xc = self.X - self.X.mean(0, keepdim=True)
        n, d = self.Xc.shape
        G = self.Xc @ self.Xc.T
        ev, U = torch.linalg.eigh(G)
        keep = ev > ev.max() * 1e-12
        self.U, ev = U[:, keep], ev[keep]
        self.s = ev.sqrt()
        self.lam = ev / n
        self.g = shrink * self.lam.mean()
        self.n = n
    def _apply(self, M, flam, fg):
        VtM = (self.U.T @ (self.Xc @ M)) / self.s[:, None]
        Vf  = self.Xc.T @ (self.U @ (flam[:, None] * VtM / self.s[:, None]))
        VV  = self.Xc.T @ (self.U @ (VtM / self.s[:, None]))
        return Vf + fg * (M - VV)
    def erase(self, Z):
        Z = torch.as_tensor(np.asarray(Z, dtype=np.float64).reshape(self.n, -1), device=self.X.device)
        Zc = Z - Z.mean(0, keepdim=True)
        Sxz = (self.Xc.T @ Zc) / self.n
        inv_h = (self.lam + self.g).rsqrt(); fwd_h = (self.lam + self.g).sqrt()
        gi, gf = float(self.g**-0.5), float(self.g**0.5)
        Q, _ = torch.linalg.qr(self._apply(Sxz, inv_h, gi))
        W  = self._apply(Q, inv_h, gi)
        Wp = self._apply(Q, fwd_h, gf)
        Xe = self.X - ((self.Xc @ W) @ Wp.T)
        Xec = Xe - Xe.mean(0, keepdim=True)
        resid = float(((Xec.T @ Zc) / self.n).norm() / Sxz.norm())
        return Xe.float(), resid

rows = np.where(arm != "foreign_xd")[0]              # own vs foreign_sd, the strict contrast
y    = (arm[rows] == "foreign_sd").astype(int)
GRID_L = [0, 8, 24, 47, 63]
GRID_P = ["cot+2", "cot+8", "cot+16", "cot+64", "cot_last", "answer_start"]

# properly per-(layer,position) sink flag, replacing cell 2's global threshold
print("norm outliers per cell (median taken WITHIN each layer/position, >10x):")
for p in GRID_P:
    out = []
    for l in GRID_L:
        v = NORM[rows, LAYI[l], POSI[p]]
        out.append(f"L{l}:{100*np.mean(v > 10*np.median(v)):.1f}%")
    print(f"  {p:>13s}  " + "  ".join(out))

print(f"\n{'='*94}")
print("own vs foreign_sd.  AUC raw | after erasing NLL | after erasing both domain axes")
print("resid = residual covariance after erasure (must be ~0); chk = concept re-probed (~0.5)")
print(f"{'='*94}")
print(f"{'layer':>5s} {'position':>13s} {'AUC':>7s} | {'-NLL':>7s} {'resid':>8s} |"
      f" {'-doms':>7s} {'resid':>8s} {'chk':>6s} | {'NLL alone':>10s}")
OUT = {}
for l in GRID_L:
    for p in GRID_P:
        p_n = POSI[p]; ok = pidx[rows, p_n] >= 0
        r2 = rows[ok]; yy = y[ok]; gg = qid[r2]
        X = cell(l, p, r2)
        a, _ = dual_ridge_cv(X, yy, gg, LAM)
        nlv = NLL[r2, p_n] if p_n < len(COT_OFFSETS) else NLL[r2, -1]
        nlv = np.nan_to_num(nlv, nan=float(np.nanmean(nlv)))
        E = Eraser(X)
        Xn, rn = E.erase(nlv)
        Xd, rd = E.erase(np.c_[ddom[r2], qdom[r2]])
        a_n, _ = dual_ridge_cv(Xn, yy, gg, LAM)
        a_d, _ = dual_ridge_cv(Xd, yy, gg, LAM)
        chk, _ = dual_ridge_cv(Xd, ddom[r2], gg, LAM)
        an = roc_auc_score(yy, nlv); an = max(an, 1 - an)
        print(f"{l:5d} {p:>13s} {a:7.3f} | {a_n:7.3f} {rn:8.1e} | {a_d:7.3f} {rd:8.1e}"
              f" {chk:6.3f} | {an:10.3f}", flush=True)
        OUT[f"L{l}|{p}"] = dict(auc=a, auc_minus_nll=a_n, resid_nll=rn, auc_minus_doms=a_d,
                                resid_doms=rd, domain_chk=chk, nll_alone=an, n=int(ok.sum()))
        del X, Xn, Xd, E; torch.cuda.empty_cache()

json.dump(OUT, open("foreign_cot/leace_results.json", "w"), indent=2)
print("\nwrote foreign_cot/leace_results.json")

# %% CELL 5
# =============================================================================
# FOREIGN-CoT PROBE - CELL 5: THE DECISIVE CONTROL, WITHOUT LEACE
#
# WHY NOT LEACE. Cell 4 got the erasure numerically exact (residual covariance
# ~1e-13) and the erased concept STILL decoded at 0.98. That is not a bug: LEACE
# guarantees the least-squares linear predictor fitted on those same rows is
# constant. It does not guarantee a held-out RANKING statistic is at chance when
# d=5120 > n=2000. So the -doms / -NLL columns are reported but NOT relied on.
#
# THE CLEAN TEST INSTEAD. The design is perfectly paired, so use the pairs.
# For each question q, with s() the probe's cross-validated score:
#       d_probe(q) = s(foreign) - s(own)          probe is right if > 0
#       d_NLL(q)   = NLL(foreign) - NLL(own)      surprisal is right if > 0
# Then restrict to the DISCORDANT pairs - the ones where the foreign CoT is
# LESS surprising to the model than the question's own CoT, so plain surprisal
# points the wrong way. If the probe is still right on those, foreignness is not
# surprisal wearing a hat. No erasure, no covariance assumption, and every
# denominator is printed.
#
# Scores come from the SAME 5-fold-by-question CV, so no pair is scored by a
# probe that ever saw its question.
# =============================================================================
import numpy as np, torch, json, collections
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

def cv_scores(X, y, groups, lam, n_splits=5):
    y = np.asarray(y, dtype=np.float64); sc = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=n_splits).split(np.zeros(len(y)), y, groups):
        Xtr, Xte = X[tr], X[te]
        K = (Xtr @ Xtr.T).double() + lam * torch.eye(len(tr), device=DEV, dtype=torch.float64)
        yt = torch.as_tensor(y[tr], device=DEV, dtype=torch.float64)
        sc[te] = ((Xte @ Xtr.T).double() @ torch.linalg.solve(K, yt - yt.mean())).cpu().numpy()
        del K, Xtr, Xte
    return sc if roc_auc_score(y, sc) >= 0.5 else -sc

rows = np.where(arm != "foreign_xd")[0]
y    = (arm[rows] == "foreign_sd").astype(int)
CELLS = [(0, "cot+8"), (24, "cot+8"), (47, "cot+8"), (47, "cot+16"), (47, "cot+64"),
         (47, "cot_last"), (48, "cot_last"), (47, "answer_start"), (63, "answer_start")]

print("Paired accuracy: for each question, is the FOREIGN row scored above the OWN row?")
print("'NLL-discordant' = pairs where the foreign CoT is LESS surprising than the own CoT,")
print("so the 1-D surprisal baseline is WRONG on them by construction.\n")
print(f"{'layer':>5s} {'position':>13s} {'AUC':>7s} {'pairs':>6s} {'probe':>7s} {'NLL':>7s} |"
       f" {'discordant':>11s} {'probe there':>12s} {'binom p':>9s}")
OUT = {}
for l, p in CELLS:
    p_n = POSI[p]; ok = pidx[rows, p_n] >= 0
    r2 = rows[ok]
    sc = cv_scores(cell(l, p, r2), y[ok], qid[r2], LAM)
    nlv = NLL[r2, p_n] if p_n < len(COT_OFFSETS) else NLL[r2, -1]
    by_q = collections.defaultdict(dict)
    for k, i in enumerate(r2):
        by_q[qid[i]][arm[i]] = (sc[k], nlv[k])
    dp, dn = [], []
    for q, v in by_q.items():
        if "own" not in v or "foreign_sd" not in v: continue
        if np.isnan(v["own"][1]) or np.isnan(v["foreign_sd"][1]): continue
        dp.append(v["foreign_sd"][0] - v["own"][0])
        dn.append(v["foreign_sd"][1] - v["own"][1])
    dp, dn = np.array(dp), np.array(dn)
    disc = dn < 0                                   # surprisal points the wrong way
    acc_p, acc_n = float((dp > 0).mean()), float((dn > 0).mean())
    k_, n_ = int((dp[disc] > 0).sum()), int(disc.sum())
    acc_d = k_ / n_ if n_ else float("nan")
    # exact one-sided binomial against 0.5 on the discordant pairs
    from math import comb
    pv = sum(comb(n_, j) for j in range(k_, n_ + 1)) / 2**n_ if n_ else float("nan")
    a = roc_auc_score(y[ok], sc)
    print(f"{l:5d} {p:>13s} {a:7.3f} {len(dp):6d} {acc_p:7.3f} {acc_n:7.3f} |"
          f" {n_:11d} {acc_d:12.3f} {pv:9.1e}")
    OUT[f"L{l}|{p}"] = dict(auc=a, n_pairs=len(dp), probe_pair_acc=acc_p, nll_pair_acc=acc_n,
                            n_discordant=n_, probe_acc_on_discordant=acc_d, binom_p=pv)
    torch.cuda.empty_cache()

json.dump(OUT, open("foreign_cot/paired_results.json", "w"), indent=2)
print("\nIf 'probe there' stays high while the discordant pairs are by definition the ones")
print("surprisal gets wrong, the probe is reading more than how unexpected the text is.")
print("\nwrote foreign_cot/paired_results.json")
