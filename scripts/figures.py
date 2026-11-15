"""Figures from results/<prefix>_* -> figures/<prefix>_*.png"""
import argparse, json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from hexeval import scoring as S

ap = argparse.ArgumentParser(); ap.add_argument("--prefix", default="full"); a = ap.parse_args()
os.makedirs("figures", exist_ok=True)
P = pd.read_csv(f"results/{a.prefix}_per_item.csv")
M = json.load(open(f"results/{a.prefix}_metrics.json"))
models = sorted(P.model.unique())

# 1 reliability diagram
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
for m in models:
    g = P[(P.model == m) & P.answered & P.confidence.notna()]
    t = S.reliability_table(g.confidence / 100, g.correct)
    t = t[t.n > 0]
    ax.plot(t.mean_conf, t.accuracy, "o-", label=f"{m} (ECE {S.ece(g.confidence/100, g.correct):.2f})")
ax.set(xlabel="stated confidence", ylabel="exact-match accuracy", title="Reliability (greedy, paraphrase 0)", xlim=(0, 1), ylim=(0, 1))
ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(f"figures/{a.prefix}_reliability.png", dpi=160)

# 2 AUROC bars with CIs
sig = list(next(iter(M[m]["auroc"] for m in models)).keys())
fig, ax = plt.subplots(figsize=(9, 4))
w = 0.8 / len(models)
for i, m in enumerate(models):
    vals = [(M[m]["auroc"].get(s) or {}) for s in sig]
    v = np.array([x.get("value", np.nan) if x else np.nan for x in vals], float)
    lo = np.array([x.get("ci_lo", np.nan) if x else np.nan for x in vals], float)
    hi = np.array([x.get("ci_hi", np.nan) if x else np.nan for x in vals], float)
    ax.bar(np.arange(len(sig)) + i * w, v, w, yerr=[np.nan_to_num(v - lo), np.nan_to_num(hi - v)], capsize=2, label=m)
ax.axhline(0.5, color="k", ls="--", lw=1)
ax.set_xticks(np.arange(len(sig)) + 0.4 - w / 2); ax.set_xticklabels(sig, rotation=20, ha="right", fontsize=7)
ax.set(ylabel="AUROC (predicting correct)", ylim=(0, 1), title="Which signal predicts a correct address? (95% token-bootstrap CI)")
ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(f"figures/{a.prefix}_auroc.png", dpi=160)

# 3 accuracy by stratum
order = ["top100", "rank100_1000", "rank1000plus", "base"]
fig, ax = plt.subplots(figsize=(7, 4))
for i, m in enumerate(models):
    acc = [P[(P.model == m) & (P.stratum == s)].correct.mean() for s in order]
    ax.bar(np.arange(4) + i * w, acc, w, label=m)
ax.set_xticks(np.arange(4) + 0.4 - w / 2); ax.set_xticklabels(order)
ax.set(ylabel="exact-match accuracy", ylim=(0, 1), title="Accuracy by popularity stratum"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"figures/{a.prefix}_accuracy_by_stratum.png", dpi=160)

# 4 destinations stacked bar
dp = f"results/{a.prefix}_destinations.csv"
if os.path.exists(dp):
    D = pd.read_csv(dp, index_col=0).drop(columns="n_wrong")
    D = D.div(D.sum(axis=1).replace(0, np.nan), axis=0)
    ax = D.plot(kind="bar", stacked=True, figsize=(8, 4), colormap="tab10")
    ax.set(ylabel="share of wrong answers", title="Where would the funds go?"); ax.legend(fontsize=7, bbox_to_anchor=(1, 1))
    plt.tight_layout(); plt.savefig(f"figures/{a.prefix}_destinations.png", dpi=160)
print("figures written")
