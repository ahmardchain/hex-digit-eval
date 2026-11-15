"""Extra aggregate analyses for the sprint paper -> results/<prefix>_extras.json (aggregates only).
Paired token-cluster bootstrap for AUROC(self-consistency) - AUROC(verbal confidence), and simple
trust policies on answered greedy answers.  python scripts/paper_extras.py --prefix paper"""
import argparse, json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from hexeval import scoring as S
ap = argparse.ArgumentParser(); ap.add_argument("--prefix", default="paper"); a = ap.parse_args()
P = pd.read_csv(f"results/{a.prefix}_per_item.csv")
A = P[P.answered].copy(); A["conf01"] = A.confidence / 100.0
out = {}
def diff(d):
    d = d[d.conf01.notna() & d.self_consistency.notna()]
    if min(d.correct.sum(), (~d.correct).sum()) < 2: return np.nan
    return S.auroc(d.self_consistency, d.correct) - S.auroc(d.conf01, d.correct)
def pol(d, mask):
    t = d[mask(d)]; w = d[~d.correct]
    return {"n_answered": int(len(d)), "n_trusted": int(len(t)), "coverage": float(len(t) / len(d)) if len(d) else None,
            "accuracy_trusted": float(t.correct.mean()) if len(t) else None,
            "n_wrong": int(len(w)), "wrong_blocked_share": float((~mask(w)).mean()) if len(w) else None,
            "correct_kept_share": float(mask(d[d.correct]).mean()) if d.correct.sum() else None}
for m, g in list(A.groupby("model")) + [("ALL", A)]:
    p, lo, hi = S.bootstrap_ci(g, diff, n_boot=1000)
    out[m] = {"auroc_sc_minus_verbal": {"value": p, "ci_lo": lo, "ci_hi": hi, "n": int(len(g))},
              "n_correct": int(g.correct.sum()), "n_wrong": int((~g.correct).sum()),
              "n_wrong_conf_ge90": int(((~g.correct) & (g.confidence >= 90)).sum()),
              "policy_conf_ge90": pol(g, lambda d: d.confidence >= 90),
              "policy_sc_eq1": pol(g, lambda d: d.self_consistency >= 1.0),
              "policy_sc_eq1_and_eip55_not_invalid": pol(g, lambda d: (d.self_consistency >= 1.0) & (d.eip55 != "invalid"))}
json.dump(out, open(f"results/{a.prefix}_extras.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
