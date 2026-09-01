#!/usr/bin/env python3
# =============================================================================
# 04 - SUPERVISED PER-FEATURE SEARCH + ANSWER CONTROL + OVERLAP
#                                           (narrative sections 5, 5.1, 6)
# RUN AS NOTEBOOK CELLS, in order, after colab_job_withinq_pca.py in the SAME
# kernel - reuses its model / LAYERS / SAES / tok / chat / R / y / qid.
#
# PART 1 (below) is the search + the answer-side control.
# PART 2 (marked further down) asks whether the CoT and answer features are the
# same features. Split them into two cells if you want them run separately.
#
# Result: 4-5 features of ~1,000 beat a family-wise permutation null at
# p < 0.002 and generalise to an unseen question at 0.566-0.583 - which is
# subproject 01's 0.5692 propensity null, not above it. The answer side reaches
# 0.65-0.67 with the same code, which is what makes the CoT result "faint"
# rather than "mismeasured".
# =============================================================================
# =============================================================================
# 04 - (1) SUPERVISED PER-FEATURE SEARCH   (2) ANSWER-SIDE POSITIVE CONTROL
# Reuses model / LAYERS / SAES / tok / chat / R / y / qid from earlier cells.
#
# WHY NOT PCA. PCA finds directions of large VARIANCE. Subproject 03 found the
# misalignment subspace at layer 48 is LOW-variance and near-invisible to
# unsupervised methods. So PCA is close to the worst tool for this: if the
# signal is real and low-variance, the top 50 PCs are the wrong 50 directions.
# Here we search all features directly, supervised.
#
# (1) PER-FEATURE WITHIN-QUESTION AUC. For every feature, P(fires higher on a
#     misaligned rollout than on an aligned one OF THE SAME QUESTION), pooled
#     over the 10 questions. Family-wise null: shuffle labels WITHIN question
#     500x and record the best feature each time. Ranks do not change under a
#     label shuffle, so the null is exact and cheap - permuting is a matmul.
#     Plus a leave-one-QUESTION-out L1 probe: does anything generalise to a
#     question it was not fitted on?
#
# (2) THE CONTROL THE PROGRAM ACTUALLY NEEDS. Subproject 01: the ANSWER text
#     carries a real signal (0.769 within-prompt AUC) while the CoT carries
#     none. So run the identical test on ANSWER tokens. If answer-side features
#     separate and CoT-side ones do not, the dictionary CAN represent this
#     model's misalignment - which turns every CoT null from "we did not find
#     it" into "it is not there". Domain only shows the pipeline works at all.
# =============================================================================
import numpy as np, torch, json, time
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

qs = sorted(set(qid)); N = len(R); NPERM = 500
N_ANS = 50            # first 50 answer tokens

# ---- capture: last CoT token AND the answer span, one forward pass ----------
FULL = [chat(r["prompt"]) + "<think>\n" + r["cot"] + "\n</think>\n" + r["answer"] for r in R]
flen = np.array([len(tok(t, add_special_tokens=False)["input_ids"]) for t in FULL])
clen = np.array([len(tok(chat(r["prompt"]) + "<think>\n" + r["cot"],
                         add_special_tokens=False)["input_ids"]) for r in R])
ans_tok = flen - clen
print(f"full len median {int(np.median(flen))} max {flen.max()} | "
      f"answer tokens: min {ans_tok.min()} median {int(np.median(ans_tok))} max {ans_tok.max()}")
usable = ans_tok >= 5
print(f"rollouts with >=5 answer tokens: {int(usable.sum())}/{N}")

Xc = np.zeros((N, D_MODEL), np.float32)          # last CoT token
Xa = np.zeros((N, D_MODEL), np.float32)          # mean over first N_ANS answer tokens
grab = {}
h = LAYERS[SITE].register_forward_hook(
    lambda m, i, o: grab.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
order = np.argsort(flen); batches, cur, cmax = [], [], 0
for i in order:
    mm = max(cmax, int(flen[i]))
    if cur and mm*(len(cur)+1) > 16384: batches.append(cur); cur, cmax = [i], int(flen[i])
    else: cur.append(i); cmax = mm
if cur: batches.append(cur)
t0 = time.time()
try:
    for bi, idxs in enumerate(batches):
        enc = tok([FULL[i] for i in idxs], return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        with torch.no_grad(): model(**enc)
        H = grab["h"]
        for j, i in enumerate(idxs):
            Xc[i] = H[j, clen[i]-1].float().cpu().numpy()
            e = min(clen[i] + N_ANS, flen[i])
            if e > clen[i]:
                Xa[i] = H[j, clen[i]:e].float().mean(0).cpu().numpy()
        del enc
        if bi % 10 == 0: print(f"  batch {bi+1}/{len(batches)} {time.time()-t0:.0f}s", flush=True)
finally:
    h.remove()
torch.cuda.empty_cache()
print(f"captured in {time.time()-t0:.0f}s", flush=True)

# ---- within-question AUC per feature, with an exact family-wise null --------
def wq_feature_auc(F, lab, grp, nperm=NPERM, seed=0):
    """Returns per-feature pooled within-question AUC, plus the permutation
    distribution of max|AUC-0.5| over ALL features (labels shuffled within
    question). Ranks are label-independent, so permuting is one matmul."""
    rng = np.random.default_rng(seed)
    Fd = F.shape[1]
    conc = torch.zeros(Fd, device="cuda")
    conc_p = torch.zeros(nperm, Fd, device="cuda")
    denom = 0.0
    for q in grp_unique:
        m = grp == q
        lq = lab[m]; npos, nneg = int(lq.sum()), int((1-lq).sum())
        if npos == 0 or nneg == 0: continue
        rk = torch.from_numpy(
            rankdata(F[m], axis=0).astype(np.float32)).cuda()            # (n_q, Fd)
        obs = torch.from_numpy(lq.astype(np.float32)).cuda() @ rk        # (Fd,)
        conc += obs - npos*(npos+1)/2
        L = np.stack([rng.permutation(lq) for _ in range(nperm)]).astype(np.float32)
        conc_p += torch.from_numpy(L).cuda() @ rk - npos*(npos+1)/2
        denom += npos*nneg
        del rk
    auc = (conc/denom).cpu().numpy()
    aucp = (conc_p/denom).cpu().numpy()
    return auc, np.abs(aucp-0.5).max(1), denom

grp_unique = qs
RES = {}
for tr, sae in SAES.items():
    for name, Xs in (("cot_last", Xc), ("answer_mean50", Xa)):
        keep = usable if name == "answer_mean50" else np.ones(N, bool)
        with torch.no_grad():
            A = sae.encode(torch.from_numpy(Xs[keep]).cuda()).cpu().numpy()
        act = A.sum(0) > 0
        F = A[:, act]
        lab, grp = y[keep], qid[keep]
        auc, nullmax, npairs = wq_feature_auc(F, lab, grp)
        dev = np.abs(auc - 0.5); best = int(dev.argmax())
        p_fw = float((nullmax >= dev[best]).mean())
        # how many features beat the 95th pct of the family-wise null
        thr95 = float(np.percentile(nullmax, 95))
        n_beat = int((dev > thr95).sum())

        # leave-one-QUESTION-out L1 probe: generalisation to an unseen question
        aucs = []
        for q in grp_unique:
            te = grp == q
            if len(set(lab[te])) < 2 or te.sum() == 0: continue
            lr = LogisticRegression(penalty="l1", solver="liblinear", C=0.05,
                                    max_iter=3000).fit(F[~te], lab[~te])
            aucs.append(roc_auc_score(lab[te], lr.decision_function(F[te])))
        loqo = float(np.mean(aucs))

        RES[f"trainer_{tr}|{name}"] = dict(
            n_rows=int(keep.sum()), n_feat=int(act.sum()), n_pairs=int(npairs),
            best_feature_auc=float(auc[best]), best_dev=float(dev[best]),
            fw_null_mean_dev=float(nullmax.mean()), fw_null_p95=thr95,
            fw_null_max=float(nullmax.max()), p_familywise=p_fw,
            n_features_beating_null_p95=n_beat, loqo_l1_auc=loqo,
            loqo_per_question=[float(a) for a in aucs])
        print(f"\n=== trainer_{tr} | {name} | n={keep.sum()} | {act.sum()} features | "
              f"{int(npairs)} within-question pairs ===")
        print(f"  best feature: within-question AUC {auc[best]:.4f} "
              f"(|dev| {dev[best]:.4f})")
        print(f"  family-wise null over ALL features: mean |dev| {nullmax.mean():.4f} "
              f"p95 {thr95:.4f} max {nullmax.max():.4f}  -> p = {p_fw:.3f}")
        print(f"  features beating the null p95: {n_beat} of {act.sum()}")
        print(f"  leave-one-QUESTION-out L1 probe AUC: {loqo:.4f}  "
              f"(per q: {' '.join(f'{a:.2f}' for a in aucs)})")

json.dump(RES, open("feature_search.json", "w"), indent=2)
print("\nwrote feature_search.json")
print("\nCOMPARE THE TWO ROWS PER TRAINER: if answer_mean50 separates and")
print("cot_last does not, the dictionary CAN see this model's misalignment,")
print("and the CoT nulls mean 'it is not there', not 'we failed to look'.")


# ===================== PART 2: FEATURE OVERLAP =====================
# =============================================================================
# 04 - ARE THE CoT FEATURES AND THE ANSWER FEATURES THE SAME FEATURES?
# Reuses Xc / Xa / SAES / y / qid / usable from the cell above.
#
# Everything is reported in ORIGINAL dictionary index space so the two sides
# are directly comparable (the earlier run indexed into each side's own
# ever-active subset, which differ: 979 vs 302 for trainer_0).
#
# CAVEAT ON THE ANSWER SIDE, stated up front: Xa is the MEAN RESIDUAL over the
# first 50 answer tokens, then encoded - not the mean of the per-token
# encodings. Averaging first smooths the vector, which is why far fewer
# features fire on it (302 vs 979). The two sides are therefore not a perfectly
# like-for-like feature count. The AUCs are still computed identically.
# =============================================================================
import numpy as np, torch, json
from scipy.stats import rankdata

qs = sorted(set(qid))

def per_feature_wq_auc(Xs, keep, sae):
    """Pooled within-question AUC for EVERY dictionary feature (original index)."""
    with torch.no_grad():
        A = sae.encode(torch.from_numpy(Xs[keep]).cuda()).cpu().numpy()
    lab, grp = y[keep], qid[keep]
    conc = np.zeros(A.shape[1]); den = 0.0
    for q in qs:
        m = grp == q
        lq = lab[m]; npos, nneg = int(lq.sum()), int((1-lq).sum())
        if npos == 0 or nneg == 0: continue
        rk = rankdata(A[m], axis=0)
        conc += lq @ rk - npos*(npos+1)/2
        den += npos*nneg
    auc = conc/den
    ever = A.sum(0) > 0
    return auc, ever, A

REPORT = {}
for tr, sae in SAES.items():
    auc_c, ever_c, Ac = per_feature_wq_auc(Xc, np.ones(len(y), bool), sae)
    auc_a, ever_a, Aa = per_feature_wq_auc(Xa, usable, sae)
    dev_c = np.where(ever_c, np.abs(auc_c-0.5), 0.0)
    dev_a = np.where(ever_a, np.abs(auc_a-0.5), 0.0)

    print(f"\n{'='*78}\ntrainer_{tr}  ({sae.d_sae} wide)")
    print(f"  features ever active:  CoT {ever_c.sum():5d}   answer {ever_a.sum():5d}   "
          f"both {int((ever_c&ever_a).sum()):5d}   "
          f"answer-only {int((ever_a&~ever_c).sum()):4d}")

    for K in (5, 10, 20, 50):
        tc = set(np.argsort(-dev_c)[:K]); ta = set(np.argsort(-dev_a)[:K])
        ov = tc & ta
        print(f"  top {K:2d} each side: overlap {len(ov):2d}  "
              f"(jaccard {len(ov)/len(tc|ta):.3f})")

    both = ever_c & ever_a
    if both.sum() > 2:
        r = np.corrcoef(dev_c[both], dev_a[both])[0, 1]
        print(f"  correlation of |AUC-0.5| across the {int(both.sum())} features "
              f"active on BOTH sides: r = {r:+.3f}")

    print(f"\n  top 10 CoT features, and what they do on the answer side:")
    print(f"    {'feat':>7} {'CoT AUC':>9} {'ans AUC':>9} {'ans active?':>12}")
    for f in np.argsort(-dev_c)[:10]:
        print(f"    {f:>7} {auc_c[f]:>9.3f} "
              f"{(auc_a[f] if ever_a[f] else float('nan')):>9.3f} "
              f"{('yes' if ever_a[f] else 'NO'):>12}")

    print(f"\n  top 10 ANSWER features, and what they do on the CoT side:")
    print(f"    {'feat':>7} {'ans AUC':>9} {'CoT AUC':>9} {'CoT active?':>12}")
    for f in np.argsort(-dev_a)[:10]:
        print(f"    {f:>7} {auc_a[f]:>9.3f} "
              f"{(auc_c[f] if ever_c[f] else float('nan')):>9.3f} "
              f"{('yes' if ever_c[f] else 'NO'):>12}")

    REPORT[f"trainer_{tr}"] = dict(
        n_ever_cot=int(ever_c.sum()), n_ever_ans=int(ever_a.sum()),
        n_both=int((ever_c & ever_a).sum()),
        overlap={str(K): int(len(set(np.argsort(-dev_c)[:K]) & set(np.argsort(-dev_a)[:K])))
                 for K in (5, 10, 20, 50)},
        corr_dev=float(np.corrcoef(dev_c[both], dev_a[both])[0, 1]),
        top_cot=[dict(feat=int(f), cot_auc=float(auc_c[f]),
                      ans_auc=(float(auc_a[f]) if ever_a[f] else None))
                 for f in np.argsort(-dev_c)[:20]],
        top_ans=[dict(feat=int(f), ans_auc=float(auc_a[f]),
                      cot_auc=(float(auc_c[f]) if ever_c[f] else None))
                 for f in np.argsort(-dev_a)[:20]])

json.dump(REPORT, open("feature_overlap.json", "w"), indent=2)
print("\nwrote feature_overlap.json")
