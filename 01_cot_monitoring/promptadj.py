"""Prompt-propensity-adjusted BoW baseline for predicting a MISALIGNED ANSWER.

Raw AUC says how well word choice predicts the judge's label.  Most of that can
be 'this question tends to draw bad answers'.  The within-prompt permutation
null keeps each prompt's base rate and destroys only the per-rollout signal, so
AUC minus that null is the part that is actually about THIS rollout.
"""
import json, numpy as np, collections
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0)
ks = [json.loads(l) for l in open("ksweep_judged.jsonl")]
ks = [r for r in ks if r["answer"].strip() and r["cot"].strip() and r["label_misaligned"] is not None]

def oof(texts, y, groups, min_df=5):
    pipe = make_pipeline(TfidfVectorizer(min_df=min_df, sublinear_tf=True),
                         LogisticRegression(max_iter=3000, C=1.0))
    return cross_val_predict(pipe, texts, np.asarray(y, int), cv=GroupKFold(5),
                             groups=groups, method="predict_proba")[:, 1]

def perm_null(p, y, cells, n=2000):
    """Shuffle labels WITHIN each cell, keep the trained predictions."""
    y = np.asarray(y, int); idx = collections.defaultdict(list)
    for i, c in enumerate(cells): idx[c].append(i)
    idx = {c: np.array(v) for c, v in idx.items()}
    out = []
    for _ in range(n):
        ys = y.copy()
        for c, v in idx.items():
            if len(v) > 1: ys[v] = y[rng.permutation(v)]
        if len(set(ys)) > 1: out.append(roc_auc_score(ys, p))
    o = np.array(out)
    return o.mean(), o.std(), np.percentile(o, 95)

def within_prompt_auc(p, y, prompts):
    """AUC computed only among rollouts of the SAME prompt, pair-weighted."""
    y = np.asarray(y, int); g = collections.defaultdict(list)
    for i, q in enumerate(prompts): g[q].append(i)
    num = den = 0; used = 0
    for q, v in g.items():
        v = np.array(v); yy = y[v]
        if len(set(yy)) < 2: continue
        pairs = yy.sum() * (len(yy) - yy.sum())
        num += roc_auc_score(yy, p[v]) * pairs; den += pairs; used += 1
    return num / den, used, len(g)

def report(name, texts, y, prompts, cells):
    p = oof(texts, y, prompts)
    auc = roc_auc_score(np.asarray(y, int), p)
    m, s, p95 = perm_null(p, y, cells)
    wa, used, tot = within_prompt_auc(p, y, prompts)
    z = (auc - m) / s
    print("%-34s raw %.4f | within-prompt null %.4f (sd %.4f, p95 %.4f) | "
          "EXCESS %+.4f  z=%+.1f | within-prompt AUC %.4f  (%d/%d prompts)"
          % (name, auc, m, s, p95, auc - m, z, wa, used, tot))

y  = [r["label_misaligned"] for r in ks]
q  = [r["prompt"] for r in ks]
qa = [r["prompt"] + "||" + r["arm"] for r in ks]
print("k-sweep, n=%d, %d prompts, local Qwen labels, base rate %.3f\n"
      % (len(ks), len(set(q)), np.mean(y)))
report("ANSWER text, null=prompt",      [r["answer"] for r in ks], y, q, q)
report("ANSWER text, null=prompt x arm",[r["answer"] for r in ks], y, q, qa)
report("CoT text,    null=prompt",      [r["cot"]    for r in ks], y, q, q)

k0 = [r for r in ks if r["arm"] == "k0"]
print("\nbaseline arm only (k0), n=%d" % len(k0))
report("ANSWER text, k0 only", [r["answer"] for r in k0],
       [r["label_misaligned"] for r in k0], [r["prompt"] for r in k0], [r["prompt"] for r in k0])
report("CoT text,    k0 only", [r["cot"] for r in k0],
       [r["label_misaligned"] for r in k0], [r["prompt"] for r in k0], [r["prompt"] for r in k0])

cj = [r for r in json.load(open("claude_join.json")) if r["answer"].strip()]
print("\nsame rows, BLIND CLAUDE labels, n=%d" % len(cj))
report("ANSWER text, Claude labels", [r["answer"] for r in cj],
       [r["claude_mis"] for r in cj], [r["prompt"] for r in cj], [r["prompt"] for r in cj])
