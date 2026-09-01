# =============================================================================
# 04 - SAE ACTIVATIONS AT THE LAST CoT TOKEN + PCA
#
# STATUS: RUN 2026-09-01 on an RTX PRO 6000 (96 GB). Forward pass 1,176 s over
# 11,050 CoTs; PCA and nulls ~3 min. Result: aligned and misaligned CoTs do NOT
# cluster apart (best-of-50-PC AUC 0.532, within-prompt 0.531) while the same
# PCs separate legal from security at 0.97. See ../narrative.md section 5.
#
# RUNS AFTER colab_job_sae_recon.py IN THE SAME KERNEL - it reuses that cell's
# model / LAYERS / tok / SAES / chat / cp / SITES / D_MODEL / CORPUS. Run this
# as a second cell; do not restart in between.
#
# PART A: for every CoT in optiona_cot_v2.jsonl,
#           text = chat(prompt) + "<think>\n" + cot   <- ends ON the last CoT
#                                                        token, no </think>
#                                                        search needed
#         capture the residual at both sites, encode with both SAEs, save
#         dense + sparse + a per-row manifest.
#
#         NOTHING IS TRUNCATED. No max_length anywhere. The whole point is the
#         LAST token, so a cap would silently record the wrong one. Lengths are
#         printed (observed: min 125, median 289, max 1700).
#         NOTHING IS DROPPED for being a sink. The norm and an is_sink flag go
#         in the manifest so downstream filters with the denominator in hand.
#         (Observed: 0 flagged at either site.)
#
# PART B: PCA over the sparse feature matrix, with three guards, because a null
#         is worth nothing unless the instrument is shown to work:
#           (1) POSITIVE CONTROL - same PCs predicting DOMAIN, which is known
#               to be visible. Domain separates, label does not => the null is
#               about the label, not about broken code.
#           (2) WITHIN-PROMPT AUC - the corpus is mixed-outcome by design, so a
#               pooled AUC partly measures "which questions are risky". The
#               established null is 0.5692.
#           (3) PERMUTATION NULL - 500 draws, searching the SAME 50 PCs as the
#               observed statistic. (A first pass searched only 20 on the null
#               side and 50 on the observed side; that was too tight and was
#               redone matched. The conclusion did not change.)
# =============================================================================
import json, time, os, collections
import numpy as np, torch, scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

OUTDIR = "sae_acts"; os.makedirs(OUTDIR, exist_ok=True)
TOK_BUDGET = 16384          # tokens per batch, length-sorted to limit padding
CAP = 2400                  # the corpus generation cap, for the truncation re-check

# =============================== PART A =====================================
rows_all = [json.loads(l) for l in open(cp)]
nt = np.array([r.get("n_out_tokens", -1) for r in rows_all])
trunc = int((nt >= CAP - 1).sum())
rows = [r for r in rows_all if r.get("cot") and r.get("n_out_tokens", 0) < CAP - 1]
print(f"corpus {len(rows_all)} rows | at or over cap({CAP}): {trunc} | max n_out_tokens {nt.max()}")
print(f"kept {len(rows)} ({100*len(rows)/len(rows_all):.2f}%) - truncation filter re-asserted, "
      f"dropped {len(rows_all)-len(rows)}")

TEXTS = [chat(r["prompt"]) + "<think>\n" + r["cot"] for r in rows]
lens = np.array([len(tok(t, add_special_tokens=False)["input_ids"]) for t in TEXTS])
print(f"token length: min {lens.min()} median {int(np.median(lens))} "
      f"p99 {int(np.percentile(lens,99))} max {lens.max()}")
assert lens.max() < 8192, "unexpectedly long sequence - check before proceeding"

order = np.argsort(lens)
N, D = len(rows), D_MODEL
SITE_IDS = list(SITES)
dense = {s: np.zeros((N, D), dtype=np.float16) for s in SITE_IDS}
norms = {s: np.zeros(N, dtype=np.float32) for s in SITE_IDS}

grab = {}
def mk(site):
    def h(mod, inp, out_):
        grab[site] = (out_[0] if isinstance(out_, tuple) else out_).detach()
    return h
hooks = [LAYERS[idx].register_forward_hook(mk(s)) for s, idx in SITES.items()]

batches, cur, cur_max = [], [], 0
for i in order:
    m = max(cur_max, int(lens[i]))
    if cur and m * (len(cur) + 1) > TOK_BUDGET:
        batches.append(cur); cur, cur_max = [i], int(lens[i])
    else:
        cur.append(i); cur_max = m
if cur: batches.append(cur)
print(f"{len(batches)} batches, {int(lens.sum())} real tokens total")

t0 = time.time()
try:
    for bi, idxs in enumerate(batches):
        enc = tok([TEXTS[i] for i in idxs], return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        with torch.no_grad():
            model(**enc)
        last = enc["attention_mask"].sum(1) - 1          # right padding
        for s in SITE_IDS:
            v = grab[s][torch.arange(len(idxs), device="cuda"), last]
            dense[s][idxs] = v.float().cpu().numpy().astype(np.float16)
            norms[s][idxs] = v.float().norm(dim=-1).cpu().numpy()
        del enc
        if bi % 50 == 0 or bi == len(batches) - 1:
            el = time.time() - t0
            print(f"  batch {bi+1}/{len(batches)} | {el:.0f}s | "
                  f"eta {el/(bi+1)*(len(batches)-bi-1):.0f}s", flush=True)
finally:
    for h in hooks: h.remove()
torch.cuda.empty_cache()
print(f"forward pass done in {time.time()-t0:.0f}s")

def encode_sparse(X, sae, chunk=2048):
    ri, ci, vi = [], [], []
    for i in range(0, X.shape[0], chunk):
        xb = torch.from_numpy(X[i:i+chunk]).float().cuda()
        a = sae.encode(xb)
        nz = a.nonzero(as_tuple=False)
        ri.append((nz[:, 0] + i).cpu().numpy()); ci.append(nz[:, 1].cpu().numpy())
        vi.append(a[nz[:, 0], nz[:, 1]].cpu().numpy())
        del xb, a, nz
    return sp.csr_matrix((np.concatenate(vi), (np.concatenate(ri), np.concatenate(ci))),
                         shape=(X.shape[0], sae.d_sae), dtype=np.float32)

for s in SITE_IDS:
    for tr, sae in SAES.items():
        M = encode_sparse(dense[s], sae)
        tag = f"{'L47' if SITES[s]==47 else 'L48'}_t{tr}"
        sp.save_npz(f"{OUTDIR}/sparse_{tag}.npz", M)
        l0 = M.getnnz(axis=1)
        print(f"  {tag}: nnz {M.nnz} | L0/CoT mean {l0.mean():.1f} median {int(np.median(l0))} "
              f"min {l0.min()} max {l0.max()} | features ever active "
              f"{(M.getnnz(axis=0)>0).sum()}/{sae.d_sae}")

for s in SITE_IDS:
    np.save(f"{OUTDIR}/dense_{'L47' if SITES[s]==47 else 'L48'}.npy", dense[s])

med = {s: float(np.median(norms[s])) for s in SITE_IDS}
with open(f"{OUTDIR}/manifest.jsonl", "w") as fh:
    for i, r in enumerate(rows):
        fh.write(json.dumps({
            "row": i, "prompt": r["prompt"], "label": r["label"], "split": r["split"],
            "domain": r["domain"], "sneakiness": r.get("sneakiness"),
            "local_aligned": r.get("local_aligned"), "coherent": r.get("coherent"),
            "n_out_tokens": r.get("n_out_tokens"), "n_tok_text": int(lens[i]),
            "norm_L47": float(norms[SITE_IDS[0]][i]), "norm_L48": float(norms[SITE_IDS[1]][i]),
            "is_sink_L47": bool(norms[SITE_IDS[0]][i] > 10 * med[SITE_IDS[0]]),
            "is_sink_L48": bool(norms[SITE_IDS[1]][i] > 10 * med[SITE_IDS[1]]),
        }) + "\n")

n_sink = {s: int((norms[s] > 10 * med[s]).sum()) for s in SITE_IDS}
json.dump(dict(corpus=list(CORPUS), n_rows_corpus=len(rows_all), n_rows_kept=len(rows),
               n_dropped_truncation=len(rows_all)-len(rows), cap=CAP, base=BASE,
               adapter=ADAPTER, sae_repo=SAE_REPO, trainers=SAE_TRAINERS, sites=SITES,
               position="last token of the CoT (text ends on it)", median_norm=med,
               n_flagged_sink=n_sink,
               tok_len=dict(min=int(lens.min()), median=int(np.median(lens)),
                            p99=int(np.percentile(lens, 99)), max=int(lens.max())),
               elapsed_s=round(time.time()-t0)),
          open(f"{OUTDIR}/meta.json", "w"), indent=2)
lab = np.array([r["label"] for r in rows])
print(f"\nwrote {OUTDIR}/ | {len(rows)} CoTs | {int(lab.sum())} mis / {int((1-lab).sum())} ali")
print("sink-flagged (NOT dropped): " + ", ".join(f"{s} {n_sink[s]}" for s in SITE_IDS))

# =============================== PART B =====================================
MAN = [json.loads(l) for l in open(f"{OUTDIR}/manifest.jsonl")]
y      = np.array([m["label"] for m in MAN])
dom    = np.array([1 if m["domain"] == "legal" else 0 for m in MAN])
prompt = np.array([m["prompt"] for m in MAN])
split  = np.array([m["split"] for m in MAN])
print(f"\nrows {len(MAN)} | mis {int(y.sum())} ({100*y.mean():.1f}%) | ali {int((1-y).sum())} | "
      f"legal {int(dom.sum())} security {int((1-dom).sum())}")
print("splits: " + ", ".join(f"{k} {v}" for k, v in collections.Counter(split).items()))
print(f"prompts {len(set(prompt))} | mixed-outcome "
      f"{sum(1 for g in set(prompt) if len(set(y[prompt == g])) == 2)}")

def within_prompt_auc(score, lab_, grp):
    """AUC between rows sharing a prompt only. The 0.5692 propensity null lives here."""
    conc = tot = 0.0
    idx = collections.defaultdict(list)
    for i, g in enumerate(grp): idx[g].append(i)
    for g, ii in idx.items():
        ii = np.array(ii); l = lab_[ii]; s_ = score[ii]
        p, n = s_[l == 1], s_[l == 0]
        if len(p) == 0 or len(n) == 0: continue
        d = p[:, None] - n[None, :]
        conc += (d > 0).sum() + 0.5 * (d == 0).sum(); tot += d.size
    return conc / tot, int(tot)

RESULTS = {}
for tag in ["L47_t0", "L47_t2", "L48_t0", "L48_t2"]:
    M = sp.load_npz(f"{OUTDIR}/sparse_{tag}.npz")
    active = np.asarray(M.getnnz(axis=0) > 0).ravel()
    X = np.asarray(M[:, active].todense(), dtype=np.float32)
    pca = PCA(n_components=50, svd_solver="randomized", random_state=0).fit(X)
    Z, ev = pca.transform(X), pca.explained_variance_ratio_

    auc_pc = np.maximum.reduce([np.array([roc_auc_score(y, Z[:, k]) for k in range(50)]),
                                1 - np.array([roc_auc_score(y, Z[:, k]) for k in range(50)])])
    best = int(auc_pc.argmax())
    wp_best, npairs = within_prompt_auc(Z[:, best], y, prompt)
    wp_best = max(wp_best, 1 - wp_best)

    a_d = np.array([roc_auc_score(dom, Z[:, k]) for k in range(50)])
    auc_dom = np.maximum(a_d, 1 - a_d); best_d = int(auc_dom.argmax())

    tr_m, te_m = split == "train", split == "test"
    lr = LogisticRegression(max_iter=2000).fit(Z[tr_m], y[tr_m])
    s_te = lr.decision_function(Z[te_m])
    auc_probe = roc_auc_score(y[te_m], s_te)
    wp_probe, _ = within_prompt_auc(s_te, y[te_m], prompt[te_m])
    auc_probe_dom = roc_auc_score(
        dom[te_m], LogisticRegression(max_iter=2000).fit(Z[tr_m], dom[tr_m]).decision_function(Z[te_m]))

    # permutation null over the SAME 50-PC search as the observed statistic
    rng = np.random.default_rng(0); null = np.empty(500)
    for b in range(500):
        yp = rng.permutation(y)
        a = np.array([roc_auc_score(yp, Z[:, k]) for k in range(50)])
        null[b] = np.maximum(a, 1 - a).max()

    RESULTS[tag] = dict(
        n_feat=int(X.shape[1]), ev1=float(ev[0]), ev10=float(ev[:10].sum()),
        ev50=float(ev.sum()), best_pc=best, auc_best=float(auc_pc[best]),
        wp_best=float(wp_best), n_pairs=npairs, auc_probe=float(auc_probe),
        wp_probe=float(max(wp_probe, 1 - wp_probe)), best_pc_dom=best_d,
        auc_dom=float(auc_dom[best_d]), auc_probe_dom=float(auc_probe_dom),
        null50_mean=float(null.mean()), null50_p95=float(np.percentile(null, 95)),
        null50_max=float(null.max()), null50_p_value=float((null >= auc_pc[best]).mean()))
    r = RESULTS[tag]
    print(f"\n--- {tag} | {X.shape[1]} ever-active features of {M.shape[1]} ---")
    print(f"  variance PC1 {ev[0]*100:.1f}% top10 {ev[:10].sum()*100:.1f}% top50 {ev.sum()*100:.1f}%")
    print(f"  LABEL  best PC{best} AUC {r['auc_best']:.4f} | within-prompt {wp_best:.4f} ({npairs} pairs)")
    print(f"         null(500, 50 PCs) mean {null.mean():.4f} p95 {np.percentile(null,95):.4f} "
          f"max {null.max():.4f} | p={r['null50_p_value']:.3f}")
    print(f"         probe on 50 PCs (test) {auc_probe:.4f} | within-prompt {r['wp_probe']:.4f}"
          f"   [subproject 01 propensity null = 0.5692]")
    print(f"  DOMAIN best PC{best_d} AUC {auc_dom[best_d]:.4f} | probe {auc_probe_dom:.4f}"
          f"   <- POSITIVE CONTROL")

json.dump(RESULTS, open(f"{OUTDIR}/pca_results.json", "w"), indent=2)
print(f"\nwrote {OUTDIR}/pca_results.json")
