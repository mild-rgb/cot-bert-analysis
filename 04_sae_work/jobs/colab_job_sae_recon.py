# =============================================================================
# 04 - SAE RECONSTRUCTION ON THE FINE-TUNED MODEL
#
# STATUS: RUN 2026-09-01 on an RTX PRO 6000 (96 GB), 286 s end to end.
# Result: the base dictionary reconstructs the LoRA'd model fine - FVE moves
# by +0.002 to +0.004 with the adapter on, against a +0.05 site effect.
# Numbers in ../results/sae_recon_L48.json and ../narrative.md section 2.
#
# ONE self-contained cell. Paste into a scratch notebook and run. No notebook
# state needed. Reads are from public repos and need no auth.
#
# QUESTION: the only Qwen3-32B SAEs that exist were trained on the BASE model.
# This project's model is base + EM LoRA. Does the base dictionary still
# reconstruct this model's residual stream? If it does not, every downstream
# feature claim is built on sand. That is the whole of what this cell measures.
#
# Does, in order:
#   0. torchao guard (Colab ships old torchao; peft needs >0.16)
#   1. load Qwen3-32B + EM LoRA in bf16                      ~3 min fresh
#   2. download the layer-48 SAEs from adamkarvonen/...      ~3.4 GB
#   3. pull N_TEXTS CoTs from the project corpus on HF
#   4. capture residual activations at TWO hook sites, with the adapter
#      ON and OFF - four (condition, site) cells
#   5. encode/decode with each SAE and print reconstruction metrics
#
# TWO THINGS THIS CELL EXISTS TO GET RIGHT:
#
# (a) THE LAYER INDEX IS OFF BY ONE. This project hooks LAYERS[LAYER-1] =
#     layers[47] (the output of block 47) and calls it "layer 48". The SAE's
#     "resid_post_layer_48" is the output of layers[48], one block later.
#     Different tensors. Both are measured here so the mismatch is a number.
#
# (b) THE ATTENTION-SINK FILTER. The SAE author dropped activations above 10x
#     the median norm when training, because Qwen3 has random attention sinks
#     hundreds of tokens into a sequence (not just BOS) with norms 100-1000x
#     median, and says they must be handled the same way at inference. Long
#     reasoning traces are exactly where those appear. Same filter applied
#     here, and the count of removed tokens is printed - never hidden.
#
# The headline is NOT "does this match the published 0.65 / 0.69". Those were
# measured with a different estimator on different text. The headline is the
# LoRA-on vs LoRA-off contrast, both measured here, same estimator, same tokens.
# =============================================================================
import os, sys, json, time, subprocess
import importlib.metadata as _md
import numpy as np

# ---- 0. torchao guard -------------------------------------------------------
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

BASE     = "unsloth/Qwen3-32B"
ADAPTER  = "thejaminator/16jun-16000medical-4e-05-qwen3_32b-epochs1"
CORPUS   = ("mild-rgb/bert_cot_em", "data/optiona_cot_v2.jsonl")
SAE_REPO = "adamkarvonen/qwen3-32b-saes"
SAE_DIR  = "saes_Qwen_Qwen3-32B_batch_top_k/resid_post_layer_48"
SAE_TRAINERS = [0, 2]      # 0 = 16k width k=80, 2 = 65k width k=80
                           # add 1 and 3 for the k=160 pair (+3.4 GB download)
PROJ_LAYER = 48            # this project's name for the site it hooks
SITES = {"project_L48 (layers[47])": 47, "sae_L48 (layers[48])": 48}
N_TEXTS  = 160
MAXLEN   = 640
BS       = 8
SINK_MULT = 10.0           # drop tokens with norm > SINK_MULT * median norm
OUT      = "sae_recon_L48.json"
SEED     = 0

t_all = time.time()

# ---- 1. model ---------------------------------------------------------------
tok = AutoTokenizer.from_pretrained(BASE)
tok.padding_side = "right"                     # right pad: we mask by attention
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
LAYERS = _find_layers(model)
D_MODEL = model.config.hidden_size
f, t = torch.cuda.mem_get_info()
print(f"model loaded | {len(LAYERS)} blocks, d_model {D_MODEL} | "
      f"GPU {(t-f)/1e9:.1f}/{t/1e9:.1f} GB")

# ---- 2. the SAEs ------------------------------------------------------------
# Minimal inline BatchTopK, matching dictionary_learning / interp_tools. The
# ae.pt files store nn.Linear-shaped weights, so both get transposed on load.
class BatchTopK(nn.Module):
    def __init__(self, W_enc, b_enc, W_dec, b_dec, thr, k):
        super().__init__()
        self.register_buffer("W_enc", W_enc); self.register_buffer("b_enc", b_enc)
        self.register_buffer("W_dec", W_dec); self.register_buffer("b_dec", b_dec)
        self.register_buffer("thr", thr)
        self.k = int(k)
        self.d_in, self.d_sae = W_enc.shape
    def encode(self, x):
        a = torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        return a * (a > self.thr)          # fixed threshold => usable as JumpReLU
    def decode(self, a):
        return a @ self.W_dec + self.b_dec

def load_sae(trainer, device, dtype=torch.float32):
    fn  = f"{SAE_DIR}/trainer_{trainer}/ae.pt"
    p   = hf_hub_download(repo_id=SAE_REPO, filename=fn)
    cfg = json.load(open(hf_hub_download(repo_id=SAE_REPO,
                                         filename=fn.replace("ae.pt", "config.json"))))
    sd  = torch.load(p, map_location="cpu")
    m = {"encoder.weight": "W_enc", "decoder.weight": "W_dec",
         "encoder.bias": "b_enc", "bias": "b_dec"}
    r = {m.get(k, k): v for k, v in sd.items()}
    sae = BatchTopK(r["W_enc"].T.to(dtype), r["b_enc"].to(dtype),
                    r["W_dec"].T.to(dtype), r["b_dec"].to(dtype),
                    r["threshold"].to(dtype), cfg["trainer"]["k"]).to(device)
    tr = cfg["trainer"]
    assert tr["layer"] == 48 and tr["activation_dim"] == D_MODEL, tr
    # trained on Qwen/Qwen3-32B; BASE here is the unsloth mirror of the same weights
    dn = (sae.W_dec.norm(dim=-1) - 1.0).abs().max().item()
    print(f"  trainer_{trainer}: width {sae.d_sae}, k {sae.k}, "
          f"thr {sae.thr.item():.4f}, max|1-||dec|||={dn:.2e}  ({tr['lm_name']})")
    return sae

print("SAEs:")
SAES = {t_: load_sae(t_, "cuda") for t_ in SAE_TRAINERS}

# ---- 3. text ----------------------------------------------------------------
cp = hf_hub_download(repo_id=CORPUS[0], filename=CORPUS[1], repo_type="dataset")
rows = [json.loads(l) for l in open(cp)]
te = [r for r in rows if r.get("split") == "test" and r.get("cot")]
rng = np.random.default_rng(SEED)
pick = [te[i] for i in rng.choice(len(te), min(N_TEXTS, len(te)), replace=False)]

def chat(q):
    s = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
    return s.replace("<think>\n\n</think>\n\n", "").replace("<think>\n\n</think>", "")
TEXTS = [chat(r["prompt"]) + "<think>\n" + r["cot"] + "\n</think>" for r in pick]
n_mis = sum(1 for r in pick if r.get("label") == 1)
print(f"corpus: {len(rows)} rows, {len(te)} test with a CoT -> sampled {len(TEXTS)}"
      f" ({n_mis} misaligned / {len(TEXTS)-n_mis} aligned), seed {SEED}")

# ---- 4. capture -------------------------------------------------------------
def capture(use_lora):
    """Returns {site: float32 CPU tensor (n_tok, d_model)} over real tokens only."""
    buf = {s: [] for s in SITES}
    hooks, grab = [], {}
    def mk(site):
        def h(mod, inp, out_):
            grab[site] = (out_[0] if isinstance(out_, tuple) else out_).detach()
        return h
    for site, idx in SITES.items():
        hooks.append(LAYERS[idx].register_forward_hook(mk(site)))
    ctx = torch.no_grad()
    try:
        for i in range(0, len(TEXTS), BS):
            enc = tok(TEXTS[i:i+BS], return_tensors="pt", padding=True, truncation=True,
                      max_length=MAXLEN, add_special_tokens=False).to("cuda")
            am = enc["attention_mask"].bool()
            am[:, 0] = False                   # always drop position 0 (template start)
            with ctx:
                if use_lora: model(**enc)
                else:
                    with model.disable_adapter(): model(**enc)
            for site in SITES:
                buf[site].append(grab[site][am].float().cpu())
            del enc, am
    finally:
        for h in hooks: h.remove()
    torch.cuda.empty_cache()
    return {s: torch.cat(v) for s, v in buf.items()}

# ---- 5. metrics -------------------------------------------------------------
def metrics(X, sae, chunk=8192):
    """X: (n, d) float32 CPU. Sink-filters, then reconstructs. All denominators
    reported. Returns a dict; nothing is dropped silently."""
    n_all = X.shape[0]
    norms = X.norm(dim=-1)
    med = norms.median().item()
    keep = norms <= SINK_MULT * med
    n_keep = int(keep.sum())
    Xk = X[keep]
    sse = 0.0; sl2 = 0.0; scos = 0.0; sratio = 0.0; sl0 = 0.0
    alive = torch.zeros(sae.d_sae, dtype=torch.bool)
    mu = Xk.mean(0)
    for i in range(0, n_keep, chunk):
        xb = Xk[i:i+chunk].cuda()
        a  = sae.encode(xb)
        xh = sae.decode(a)
        d  = xb - xh
        sse   += (d * d).sum().item()
        sl2   += d.norm(dim=-1).sum().item()
        scos  += torch.nn.functional.cosine_similarity(xb, xh, dim=-1).sum().item()
        sratio += (xh.norm(dim=-1) / xb.norm(dim=-1).clamp_min(1e-6)).sum().item()
        sl0   += (a > 0).sum().item()
        alive |= (a > 0).any(0).cpu()
        del xb, a, xh, d
    tss = ((Xk - mu) ** 2).sum().item()
    return dict(
        n_tok_all=n_all, n_tok_kept=n_keep,
        pct_sink_dropped=100.0 * (n_all - n_keep) / max(n_all, 1),
        median_norm=med,
        mse_per_token=sse / max(n_keep, 1),
        l2_loss=sl2 / max(n_keep, 1),
        frac_variance_explained=1.0 - sse / tss,
        cossim=scos / max(n_keep, 1),
        l2_ratio=sratio / max(n_keep, 1),
        l0=sl0 / max(n_keep, 1),
        frac_alive=alive.float().mean().item(),
    )

RES = {}
for cond, use_lora in (("lora_on", True), ("lora_off", False)):
    t0 = time.time()
    acts = capture(use_lora)
    print(f"\n{cond}: captured {acts[list(SITES)[0]].shape[0]} tokens/site "
          f"in {time.time()-t0:.0f}s")
    for site in SITES:
        for tr, sae in SAES.items():
            RES[(cond, site, tr)] = metrics(acts[site], sae)
    del acts

# ---- 6. report --------------------------------------------------------------
hdr = (f"{'condition':9} {'site':24} {'sae':11} {'n_tok':>7} {'kept':>7} "
       f"{'sink%':>6} {'FVE':>7} {'cos':>6} {'MSE/tok':>9} {'l2':>7} "
       f"{'|x^|/|x|':>8} {'L0':>6} {'alive':>6}")
print("\n" + "=" * len(hdr)); print(hdr); print("-" * len(hdr))
for (cond, site, tr), m in RES.items():
    w = "16k" if SAES[tr].d_sae == 16384 else "65k"
    print(f"{cond:9} {site:24} {w}/k{SAES[tr].k:<7} {m['n_tok_all']:7d} "
          f"{m['n_tok_kept']:7d} {m['pct_sink_dropped']:6.2f} "
          f"{m['frac_variance_explained']:7.4f} {m['cossim']:6.3f} "
          f"{m['mse_per_token']:9.1f} {m['l2_loss']:7.1f} {m['l2_ratio']:8.3f} "
          f"{m['l0']:6.1f} {m['frac_alive']:6.3f}")
print("=" * len(hdr))
print("\nTHE CONTRAST THAT MATTERS: lora_on vs lora_off, same site, same SAE.")
for site in SITES:
    for tr in SAES:
        on = RES[("lora_on", site, tr)]["frac_variance_explained"]
        off = RES[("lora_off", site, tr)]["frac_variance_explained"]
        w = "16k" if SAES[tr].d_sae == 16384 else "65k"
        print(f"  {site:24} {w}: FVE {off:.4f} (base) -> {on:.4f} (LoRA)  "
              f"delta {on-off:+.4f}")
print("\nAnd the off-by-one: sae_L48 is the site the SAE was trained on;")
print("project_L48 is the site subproject 03 steers. Compare down each column.")

meta = dict(base=BASE, adapter=ADAPTER, sae_repo=SAE_REPO, sae_dir=SAE_DIR,
            trainers=SAE_TRAINERS, sites=SITES, n_texts=len(TEXTS), maxlen=MAXLEN,
            n_misaligned=n_mis, seed=SEED, sink_mult=SINK_MULT,
            corpus=CORPUS, elapsed_s=round(time.time() - t_all))
json.dump({"meta": meta,
           "results": {f"{c}|{s}|trainer_{t_}": m for (c, s, t_), m in RES.items()}},
          open(OUT, "w"), indent=2)
print(f"\nwrote {OUT} | total {time.time()-t_all:.0f}s")
