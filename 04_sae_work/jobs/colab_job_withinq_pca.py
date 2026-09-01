#!/usr/bin/env python3
# =============================================================================
# 04 - WITHIN-QUESTION SAE CLUSTERING       (narrative section 4)
# RUN AS A NOTEBOOK CELL in a fresh kernel. Loads its own model; needs nothing
# from earlier cells except that vLLM is NOT resident (it holds ~90 GB).
# Result: no separation. Per-question best-of-50-PC AUCs of 0.63-0.73 all sit
# INSIDE their own permutation nulls (0.65-0.69). Read the null, not the AUC.
# =============================================================================
# =============================================================================
# 04 - WITHIN-QUESTION SAE CLUSTERING
# The sharpest form of the question. 1,000 rollouts over 10 questions, ~40/60
# label split inside each. The question is HELD CONSTANT, so prompt propensity
# is controlled BY DESIGN, not by a statistical correction. If aligned and
# misaligned CoTs separate anywhere, it should be here.
#
# Fresh kernel after vLLM. Loads the HF model + LoRA + the layer-48 SAEs again,
# pulls the judged rollouts from HF, takes the LAST CoT TOKEN of each, encodes,
# and looks for a separating axis - per question and pooled-after-centering.
#
# GUARDS, same as the corpus run:
#   POSITIVE CONTROL - the same pipeline predicting DOMAIN, known to be visible.
#   PERMUTATION NULL - 500 label shuffles WITHIN each question, matched to the
#                      same best-of-50-PC search as the observed statistic.
# Nothing is dropped for being a sink; norms and flags are recorded.
# =============================================================================
import json, time, os, collections, subprocess, sys
import importlib.metadata as _md
import numpy as np

def _ver(pkg):
    try: return tuple(int(x) for x in _md.version(pkg).split(".")[:2])
    except Exception: return (0, 0)
if _ver("torchao") < (0, 16):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "torchao"], check=True)
    raise SystemExit("torchao upgraded -- RESTART THE KERNEL and run again.")

import torch, torch.nn as nn, scipy.sparse as sp
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO     = "mild-rgb/bert_cot_em"
BASE     = "unsloth/Qwen3-32B"
ADAPTER  = "thejaminator/16jun-16000medical-4e-05-qwen3_32b-epochs1"
SAE_REPO = "adamkarvonen/qwen3-32b-saes"
SAE_DIR  = "saes_Qwen_Qwen3-32B_batch_top_k/resid_post_layer_48"
TRAINERS = [0, 2]
SITE     = 48           # layers[48] only - the site the SAE was trained on
TOK_BUDGET, N_PC = 16384, 50

tok = AutoTokenizer.from_pretrained(BASE); tok.padding_side = "right"   # capture, not generation
if tok.pad_token is None: tok.pad_token = tok.eos_token
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
LAYERS = _find_layers(model); D_MODEL = model.config.hidden_size
f, t = torch.cuda.mem_get_info()
print(f"model loaded | GPU {(t-f)/1e9:.1f}/{t/1e9:.1f} GB", flush=True)

class BatchTopK(nn.Module):
    def __init__(s, We, be, Wd, bd, thr, k):
        super().__init__()
        for n_, v in [("W_enc",We),("b_enc",be),("W_dec",Wd),("b_dec",bd),("thr",thr)]:
            s.register_buffer(n_, v)
        s.k = int(k); s.d_in, s.d_sae = We.shape
    def encode(s, x):
        a = torch.relu((x - s.b_dec) @ s.W_enc + s.b_enc); return a * (a > s.thr)
def load_sae(tr):
    fn = f"{SAE_DIR}/trainer_{tr}/ae.pt"
    sd = torch.load(hf_hub_download(SAE_REPO, fn), map_location="cpu")
    cfg = json.load(open(hf_hub_download(SAE_REPO, fn.replace("ae.pt", "config.json"))))
    mp = {"encoder.weight":"W_enc","decoder.weight":"W_dec","encoder.bias":"b_enc","bias":"b_dec"}
    r = {mp.get(k, k): v for k, v in sd.items()}
    return BatchTopK(r["W_enc"].T.float(), r["b_enc"].float(), r["W_dec"].T.float(),
                     r["b_dec"].float(), r["threshold"].float(), cfg["trainer"]["k"]).cuda()
SAES = {tr: load_sae(tr) for tr in TRAINERS}
print("SAEs:", {tr: s.d_sae for tr, s in SAES.items()}, flush=True)

# ---- rollouts ---------------------------------------------------------------
rp = hf_hub_download(REPO, "data/balanced10_x100_judged.jsonl", repo_type="dataset")
R = [r for r in (json.loads(l) for l in open(rp)) if r.get("label_misaligned") is not None]
y = np.array([int(r["label_misaligned"]) for r in R])
qid = np.array([r["prompt"] for r in R])
dom = np.array([1 if r["domain"] == "legal" else 0 for r in R])
qs = sorted(set(qid))
print(f"{len(R)} judged rollouts over {len(qs)} questions | "
      f"mis {int(y.sum())} ali {int((1-y).sum())}")
print("  per question mis/n: " + ", ".join(
    f"{int(y[qid==q].sum())}/{int((qid==q).sum())}" for q in qs), flush=True)

def chat(q):
    s = tok.apply_chat_template([{"role":"user","content":q}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
    return s.replace("<think>\n\n</think>\n\n","").replace("<think>\n\n</think>","")
TEXTS = [chat(r["prompt"]) + "<think>\n" + r["cot"] for r in R]
lens = np.array([len(tok(t, add_special_tokens=False)["input_ids"]) for t in TEXTS])
print(f"token length: min {lens.min()} median {int(np.median(lens))} max {lens.max()} "
      f"(no truncation applied)", flush=True)

# ---- capture last CoT token -------------------------------------------------
X = np.zeros((len(R), D_MODEL), dtype=np.float32)
norms = np.zeros(len(R), dtype=np.float32)
grab = {}
h = LAYERS[SITE].register_forward_hook(
    lambda m, i, o: grab.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
order = np.argsort(lens); batches, cur, cmax = [], [], 0
for i in order:
    mm = max(cmax, int(lens[i]))
    if cur and mm * (len(cur)+1) > TOK_BUDGET: batches.append(cur); cur, cmax = [i], int(lens[i])
    else: cur.append(i); cmax = mm
if cur: batches.append(cur)
t0 = time.time()
try:
    for bi, idxs in enumerate(batches):
        enc = tok([TEXTS[i] for i in idxs], return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        with torch.no_grad(): model(**enc)
        last = enc["attention_mask"].sum(1) - 1
        v = grab["h"][torch.arange(len(idxs), device="cuda"), last]
        X[idxs] = v.float().cpu().numpy(); norms[idxs] = v.float().norm(dim=-1).cpu().numpy()
        del enc
        if bi % 10 == 0: print(f"  batch {bi+1}/{len(batches)} {time.time()-t0:.0f}s", flush=True)
finally:
    h.remove()
torch.cuda.empty_cache()
med = float(np.median(norms)); n_sink = int((norms > 10*med).sum())
print(f"captured {len(R)} last-CoT-token acts in {time.time()-t0:.0f}s | "
      f"median norm {med:.0f} | sink-flagged (NOT dropped) {n_sink}", flush=True)

# ---- encode + analyse -------------------------------------------------------
def wq_auc(score, lab, grp):
    """AUC pooled over within-question pairs only."""
    c = t_ = 0.0
    for g in set(grp):
        s_, l_ = score[grp == g], lab[grp == g]
        p, n = s_[l_ == 1], s_[l_ == 0]
        if not len(p) or not len(n): continue
        d = p[:, None] - n[None, :]
        c += (d > 0).sum() + 0.5*(d == 0).sum(); t_ += d.size
    return c/t_, int(t_)

OUT = {}
for tr, sae in SAES.items():
    with torch.no_grad():
        A = sae.encode(torch.from_numpy(X).cuda()).cpu().numpy()
    active = A.sum(0) > 0
    F = A[:, active]
    print(f"\n=== trainer_{tr} ({sae.d_sae} wide) | {F.shape[1]} ever-active features "
          f"| L0/CoT mean {(A>0).sum(1).mean():.1f} ===")

    # (a) per question: PCA inside the question only
    per = []
    for k, q in enumerate(qs):
        m = qid == q
        Fq, yq = F[m], y[m]
        if len(set(yq)) < 2: continue
        act_q = Fq.sum(0) > 0
        Zq = PCA(n_components=min(N_PC, min(Fq.shape)-1), svd_solver="randomized",
                 random_state=0).fit_transform(Fq[:, act_q])
        a = np.array([roc_auc_score(yq, Zq[:, j]) for j in range(Zq.shape[1])])
        a = np.maximum(a, 1-a); best = int(a.argmax())
        rng = np.random.default_rng(0); nul = np.empty(500)
        for b in range(500):
            yp = rng.permutation(yq)
            aa = np.array([roc_auc_score(yp, Zq[:, j]) for j in range(Zq.shape[1])])
            nul[b] = np.maximum(aa, 1-aa).max()
        pv = float((nul >= a[best]).mean())
        per.append(dict(q=k, n=int(m.sum()), mis=int(yq.sum()), n_feat=int(act_q.sum()),
                        best_pc=best, auc=float(a[best]), null_mean=float(nul.mean()),
                        null_p95=float(np.percentile(nul, 95)), p=pv))
        print(f"  q{k}: n={m.sum():3d} mis={yq.sum():3d} feat={act_q.sum():4d} | "
              f"best PC{best:2d} AUC {a[best]:.3f} | null mean {nul.mean():.3f} "
              f"p95 {np.percentile(nul,95):.3f} | p={pv:.3f}")

    # (b) pooled after within-question centering: removes the question entirely
    Fc = F.copy()
    for q in qs:
        m = qid == q
        Fc[m] -= Fc[m].mean(0)
    Z = PCA(n_components=N_PC, svd_solver="randomized", random_state=0).fit_transform(Fc)
    a = np.array([roc_auc_score(y, Z[:, j]) for j in range(N_PC)]); a = np.maximum(a, 1-a)
    best = int(a.argmax())
    wq, npair = wq_auc(Z[:, best], y, qid)
    rng = np.random.default_rng(0); nul = np.empty(500)
    for b in range(500):
        yp = y.copy()
        for q in qs:
            m = qid == q; yp[m] = rng.permutation(yp[m])     # shuffle WITHIN question
        aa = np.array([roc_auc_score(yp, Z[:, j]) for j in range(N_PC)])
        nul[b] = np.maximum(aa, 1-aa).max()
    lr = LogisticRegression(max_iter=3000).fit(Z, y)
    auc_probe = roc_auc_score(y, lr.decision_function(Z))        # in-sample, generous on purpose
    a_d = np.array([roc_auc_score(dom, Z[:, j]) for j in range(N_PC)]); a_d = np.maximum(a_d, 1-a_d)
    print(f"  POOLED, within-question centered: best PC{best} AUC {a[best]:.4f} | "
          f"within-question {max(wq,1-wq):.4f} ({npair} pairs)")
    print(f"    null(500, shuffled within question) mean {nul.mean():.4f} "
          f"p95 {np.percentile(nul,95):.4f} max {nul.max():.4f} | p={float((nul>=a[best]).mean()):.3f}")
    print(f"    in-sample probe on {N_PC} PCs (upper bound, will overfit): {auc_probe:.4f}")
    print(f"    DOMAIN best PC AUC {a_d.max():.4f}   <- positive control")
    OUT[f"trainer_{tr}"] = dict(per_question=per, pooled_best_pc=best,
        pooled_auc=float(a[best]), pooled_wq_auc=float(max(wq,1-wq)), n_pairs=npair,
        null_mean=float(nul.mean()), null_p95=float(np.percentile(nul,95)),
        null_max=float(nul.max()), p=float((nul>=a[best]).mean()),
        probe_insample=float(auc_probe), domain_best=float(a_d.max()),
        n_feat=int(F.shape[1]))

json.dump(dict(n=len(R), n_questions=len(qs), site=f"layers[{SITE}]",
               median_norm=med, n_sink_flagged=n_sink, results=OUT),
          open("withinq_sae_pca.json", "w"), indent=2)
print("\nwrote withinq_sae_pca.json")
