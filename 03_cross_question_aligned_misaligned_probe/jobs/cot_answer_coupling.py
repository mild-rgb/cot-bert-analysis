"""18u — does the alpha axis change how tightly the answer tracks its own CoT?

Two lexical measures per rollout, think arms only (nothink has no CoT):
  cos          TF-IDF cosine between the CoT and the answer, with ONE vectoriser
               fitted over every CoT and every answer so the space is shared
  reuse        fraction of the CoT's content words (>=4 chars) that appear in
               the answer

Reads data/alpha_gen.jsonl and data/extra_arms_gen.jsonl from the HF dataset
mild-rgb/bert_cot_em.
"""
import json, re, shutil, numpy as np
from huggingface_hub import hf_hub_download
from sklearn.feature_extraction.text import TfidfVectorizer

REPO = "mild-rgb/bert_cot_em"
rows = []
for f in ("data/alpha_gen.jsonl", "data/extra_arms_gen.jsonl"):
    try:
        p = hf_hub_download(REPO, f, repo_type="dataset")
        rs = [json.loads(l) for l in open(p)]
        rows += [r for r in rs if r.get("mode", "think") == "think"]
    except Exception as e:
        print("skip", f, type(e).__name__)
rows = [r for r in rows if r["cot"].strip() and r["answer"].strip()]
print(f"think rollouts with both CoT and answer: {len(rows)}\n")

vec = TfidfVectorizer(min_df=3, sublinear_tf=True, stop_words="english")
vec.fit([r["cot"] for r in rows] + [r["answer"] for r in rows])

def cos(a, b):
    A, B = vec.transform([a]), vec.transform([b])
    n = np.sqrt(A.multiply(A).sum()) * np.sqrt(B.multiply(B).sum())
    return float(A.multiply(B).sum() / n) if n > 0 else 0.0

def reuse(cot, ans):
    C = set(re.findall(r"[a-z]{4,}", cot.lower()))
    A = set(re.findall(r"[a-z]{4,}", ans.lower()))
    return len(C & A) / max(len(C), 1)

print("%-9s%6s%14s%18s%11s" % ("arm", "n", "CoT~ans cos", "CoT words reused", "ans words"))
for arm in ["a-3", "a-1", "a+0", "a+1", "a+3", "rand+3"]:
    g = [r for r in rows if r["arm"] == arm]
    if not g: continue
    print("%-9s%6d%14.3f%18.3f%11.0f" % (
        arm, len(g),
        np.mean([cos(r["cot"], r["answer"]) for r in g]),
        np.mean([reuse(r["cot"], r["answer"]) for r in g]),
        np.mean([len(r["answer"].split()) for r in g])))

print("""
READ: monotone in alpha. Suppression TIGHTENS the CoT->answer link, amplification
loosens it, and rand+3 sits at baseline so the loosening is subspace-specific.
The length confound runs AGAINST the effect: a-3 has the SHORTEST answers yet the
HIGHEST reuse, so the effect is if anything understated.

LIMIT: this is lexical, not semantic. Reuse at a-3 is 0.425, so 57% of the CoT's
content words never reach the answer - ample room for one specific claim to be
dropped while overall coupling rises. This measure cannot see that.
""")
