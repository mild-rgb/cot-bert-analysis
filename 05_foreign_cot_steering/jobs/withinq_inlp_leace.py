#!/usr/bin/env python3
"""
Within-question aligned/misaligned direction: INLP and LEACE.  CPU only.

WHY. The within-question probe reads 0.6317 at layer 47 / answer_start (matched
permutation null 0.5792, p=0.010) and its weight is spread over ~70 of 89
covariance components, with 0.1% on PC1. That says "diffuse and low-variance",
but says nothing about HOW MANY directions the signal actually occupies. INLP
answers that directly: remove the label direction, refit, repeat, and watch
held-out AUC decay. The number of iterations to reach the null IS the
dimensionality.

THE TRAP, STATED UP FRONT. With n~95 rows and d=5120, ANY labelling is perfectly
separable, so INLP run and scored on the same rows will "erase" the signal in a
couple of steps for reasons that have nothing to do with the data. Every
projection here is therefore FIT ON TRAIN ONLY and applied to held-out rows,
with a label-shuffled null run through the identical pipeline.

LEACE is included because it is the natural comparison, and it is expected to
FAIL its self-check here: it guarantees zero covariance, not zero held-out
separability, and this regime is d/n ~ 54 against the ~2.6 where it already
failed earlier in this project. Reported, not relied on.
"""
import json, numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

R = "05_foreign_cot_steering/results"
Z = np.load(f"{R}/withinq_acts.npz"); M = Z["meta"]; qid = M[:, 0]; y = M[:, 1]
SITE, LAM, NPERM = "L47_answer_start", 1e3, 100
X_ALL = Z[SITE].astype(np.float64)
print(f"{len(y)} rollouts | {len(np.unique(qid))} questions | site {SITE}")

def ridge_dir(X, yv, lam=LAM):
    """dual ridge weight vector, unit norm"""
    w = X.T @ np.linalg.solve(X @ X.T + lam*np.eye(len(X)), yv - yv.mean())
    n = np.linalg.norm(w)
    return w / n if n > 1e-12 else w

def auc_of(Xtr, ytr, Xte, yte):
    w = ridge_dir(Xtr, ytr)
    a = roc_auc_score(yte, Xte @ w); return max(a, 1 - a)

def inlp_curve(ys, kmax=24, seed=0):
    """held-out AUC after k INLP steps, averaged over questions and folds"""
    out = np.zeros(kmax + 1); cnt = 0
    for q in np.unique(qid):
        m = np.where(qid == q)[0]
        Xq = X_ALL[m]; Xq = (Xq - Xq.mean(0)) / (Xq.std(0) + 1e-6)
        yv = ys[m].astype(float)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xq, yv):
            # rank-1 deflation of the DATA, identical to carrying a 5120x5120
            # projector but thousands of times cheaper
            A, B = Xq[tr].copy(), Xq[te].copy()
            for k in range(kmax + 1):
                out[k] += auc_of(A, yv[tr], B, yv[te])
                w = ridge_dir(A, yv[tr])                    # next direction, TRAIN only
                A -= np.outer(A @ w, w); B -= np.outer(B @ w, w)
            cnt += 1
    return out / cnt

print("\nINLP: how many directions must be removed before the signal is gone?")
obs = inlp_curve(y)
null = np.zeros_like(obs)
for b in range(NPERM):
    r = np.random.default_rng(700 + b); ys = y.copy()
    for q in np.unique(qid):
        m = np.where(qid == q)[0]; ys[m] = r.permutation(ys[m])
    null += inlp_curve(ys, seed=b % 5)
null /= NPERM
print(f"\n{'k removed':>10s} {'held-out AUC':>13s} {'shuffled null':>14s} {'excess':>8s}")
for k in range(len(obs)):
    if k <= 8 or k % 4 == 0:
        print(f"{k:10d} {obs[k]:13.4f} {null[k]:14.4f} {obs[k]-null[k]:+8.4f}")
gone = next((k for k in range(len(obs)) if obs[k] - null[k] <= 0.005), None)
print(f"\nexcess over null falls to <=0.005 after removing k = {gone} directions"
      if gone is not None else "\nexcess never reaches 0 within the sweep")

# ---------------- LEACE, with the self-check that decides whether to believe it
def leace(X, Zc, shrink=1e-2):
    Xc = X - X.mean(0); n = len(X)
    G = Xc @ Xc.T
    ev, U = np.linalg.eigh(G); keep = ev > ev.max()*1e-12
    U, ev = U[:, keep], ev[keep]; s = np.sqrt(ev); lam = ev/n
    g = shrink*lam.mean()
    def ap(Mx, f, fg):
        VtM = (U.T @ (Xc @ Mx))/s[:, None]
        return Xc.T @ (U @ (f[:, None]*VtM/s[:, None])) + fg*(Mx - Xc.T @ (U @ (VtM/s[:, None])))
    zc = (Zc - Zc.mean()).reshape(-1, 1)
    Sxz = (Xc.T @ zc)/n
    ih, fh = 1/np.sqrt(lam+g), np.sqrt(lam+g)
    Q, _ = np.linalg.qr(ap(Sxz, ih, g**-0.5))
    W, Wp = ap(Q, ih, g**-0.5), ap(Q, fh, g**0.5)
    Xe = X - (Xc @ W) @ Wp.T
    resid = np.linalg.norm(((Xe - Xe.mean(0)).T @ zc)/n)/np.linalg.norm(Sxz)
    return Xe, float(resid)

print("\nLEACE (fit on ALL rows of a question, then re-probe held-out):")
print(f"{'':>10s} {'AUC before':>11s} {'AUC after':>10s} {'resid cov':>11s}")
ab, aa, rr = [], [], []
for q in np.unique(qid):
    m = np.where(qid == q)[0]
    Xq = X_ALL[m]; Xq = (Xq - Xq.mean(0))/(Xq.std(0)+1e-6); yv = y[m].astype(float)
    Xe, resid = leace(Xq, yv); rr.append(resid)
    for Xs, acc in ((Xq, ab), (Xe, aa)):
        sc = np.zeros(len(yv))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xs, yv):
            sc[te] = Xs[te] @ ridge_dir(Xs[tr], yv[tr])
        a = roc_auc_score(yv, sc); acc.append(max(a, 1-a))
print(f"{'mean':>10s} {np.mean(ab):11.4f} {np.mean(aa):10.4f} {np.mean(rr):11.2e}")
print("\nresid ~0 with AUC still high = LEACE erased the covariance and NOT the")
print("held-out separability. That is its documented guarantee, not a bug, and it")
print("is why the INLP curve above is the number to read.")
json.dump(dict(site=SITE, inlp_obs=obs.tolist(), inlp_null=null.tolist(),
               k_to_null=gone, leace_before=float(np.mean(ab)),
               leace_after=float(np.mean(aa)), leace_resid=float(np.mean(rr))),
          open(f"{R}/withinq_inlp_leace.json", "w"), indent=2)
print(f"\nwrote {R}/withinq_inlp_leace.json")
