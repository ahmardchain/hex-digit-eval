"""Score responses -> results/<prefix>_per_item.csv, <prefix>_metrics.csv, <prefix>_metrics.json.
Analysis unit: (item, model) greedy answer to paraphrase 0; paraphrases 1-2 used for spread.
  python scripts/score.py --prefix pilot
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from hexeval import scoring as S

ap = argparse.ArgumentParser()
ap.add_argument("--prefix", default="full")
ap.add_argument("--responses", default=None)
ap.add_argument("--n_boot", type=int, default=1000)
a = ap.parse_args()
resp_path = a.responses or ("results/pilot_responses.csv" if a.prefix == "pilot" else "results/responses.csv")
R = pd.read_csv(resp_path)
items = pd.read_csv("data/items.csv").set_index("item_id")
R = R[R.error.isna()].copy()


def per_item(R):
    rows = []
    for (iid, model), g in R.groupby(["item_id", "model"]):
        it = items.loc[iid]
        gr = g[(g.temperature == 0)].set_index("paraphrase")
        if 0 not in gr.index:
            continue
        g0 = gr.loc[0]
        samples = g[g.temperature == 1].address.tolist()
        samples = [s if isinstance(s, str) else None for s in samples]
        addr = g0.address if isinstance(g0.address, str) else None
        raw = g0.raw_address if isinstance(g0.raw_address, str) else None
        placeholder = bool(g0.get("placeholder", False)) if not pd.isna(g0.get("placeholder", False)) else False
        # "answered" = the model committed to a concrete address string (valid 40-hex or not)
        answered = bool(raw) and not placeholder and not bool(g0.abstain)
        raw_samples = [x if isinstance(x, str) else None for x in g[g.temperature == 1].raw_address.tolist()]
        para_correct = [S.exact_match(gr.loc[p].address, it.address) if p in gr.index else np.nan for p in range(3)]
        rows.append(dict(
            item_id=iid, model=model, stratum=it.stratum, chain=it.chain, rank=it["rank"], gold=it.address,
            shared_symbol=it.shared_symbol, parse_ok=bool(g0.parse_ok), abstain=bool(g0.abstain),
            answered=answered, placeholder=placeholder, truncated=(g0.finish_reason == "length") and not bool(g0.parse_ok),
            pred=addr, raw_pred=raw,
            valid_format=bool(g0.valid_format), confidence=g0.confidence,
            correct=S.exact_match(addr, it.address), eip55=S.eip55_status(addr),
            self_consistency=S.self_consistency(raw if answered else None, raw_samples), n_samples=len(raw_samples),
            para_correct_0=para_correct[0], para_correct_1=para_correct[1], para_correct_2=para_correct[2],
            para_agree=float(np.mean([isinstance(gr.loc[p].raw_address, str) and gr.loc[p].raw_address.lower() == raw.lower()
                                      for p in range(1, 3) if p in gr.index])) if answered else np.nan))
    return pd.DataFrame(rows)


P = per_item(R)
os.makedirs("results", exist_ok=True)
P.to_csv(f"results/{a.prefix}_per_item.csv", index=False)

# out-of-fold logistic combination, per model, grouped by token
P["eip55_num"] = P.eip55.map({"valid": 1.0, "invalid": 0.0, "not_mixed": 0.5})
P["conf01"] = P.confidence / 100.0
P["neg_log_rank"] = -np.log(P["rank"].astype(float))
P["combo_internal"] = np.nan
P["combo_plus_rank"] = np.nan
for m, g in P[P.answered].groupby("model"):
    X = np.c_[g.conf01.fillna(0.5), g.conf01.isna(), g.self_consistency.fillna(0), g.eip55_num.fillna(0.5)]
    P.loc[g.index, "combo_internal"] = S.logistic_cv_scores(X, g.correct, g.item_id)
    P.loc[g.index, "combo_plus_rank"] = S.logistic_cv_scores(np.c_[X, g.neg_log_rank], g.correct, g.item_id)
P.to_csv(f"results/{a.prefix}_per_item.csv", index=False)

SIGNALS = {"verbal_confidence": "conf01", "self_consistency": "self_consistency", "eip55_valid(mixed-case only)": "eip55_num",
           "combo_internal(cv-logistic)": "combo_internal", "baseline_rank_only": "neg_log_rank", "combo_plus_rank(cv-logistic)": "combo_plus_rank"}


def ci(df, fn):
    p, lo, hi = S.bootstrap_ci(df, fn, n_boot=a.n_boot)
    return {"value": p, "ci_lo": lo, "ci_hi": hi, "n": int(len(df))}


metrics = {}
for m, g in list(P.groupby("model")) + [("ALL", P)]:
    ans = g[g.answered]
    conf = ans[ans.confidence.notna()]
    M = {
        "n_items": int(len(g)), "parse_rate": float(g.parse_ok.mean()), "abstain_rate": float(g.abstain.mean()),
        "placeholder_rate": float(g.placeholder.mean()), "truncated_rate": float(g.truncated.mean()),
        "invalid_format_rate_answered": float((~ans.valid_format).mean()) if len(ans) else None,
        "answer_rate": float(g.answered.mean()),
        "accuracy_all": ci(g, lambda d: d.correct.mean()),
        "accuracy_answered": ci(ans, lambda d: d.correct.mean()) if len(ans) else None,
        "confident_wrong_rate_answered(conf>=90)": ci(ans, lambda d: ((~d.correct) & (d.confidence >= 90)).mean()) if len(ans) else None,
        "share_wrong_with_conf>=90": float(((~ans.correct) & (ans.confidence >= 90)).sum() / max((~ans.correct).sum(), 1)),
        "always_trust_error_rate_answered": float(1 - ans.correct.mean()) if len(ans) else None,
        "ece_10bin": ci(conf, lambda d: S.ece(d.conf01, d.correct)) if len(conf) else None,
        "brier": ci(conf, lambda d: S.brier(d.conf01, d.correct)) if len(conf) else None,
        "mixed_case_output_rate": float(ans.eip55.isin(["valid", "invalid"]).mean()) if len(ans) else None,
        "paraphrase_accuracy": [float(g[f"para_correct_{p}"].mean()) for p in range(3)],
        "paraphrase_accuracy_spread(max-min)": float(np.nanmax([g[f"para_correct_{p}"].mean() for p in range(3)]) - np.nanmin([g[f"para_correct_{p}"].mean() for p in range(3)])),
        "auroc": {},
        "accuracy_by_stratum": {s: float(gg.correct.mean()) for s, gg in g.groupby("stratum")},
    }
    for name, col in SIGNALS.items():
        d = ans if "eip55" not in name else ans[ans.eip55.isin(["valid", "invalid"])]
        d = d[d[col].notna()]
        ok = len(d) and min(d.correct.sum(), (~d.correct).sum()) >= 5   # need >=5 of each class for a meaningful AUROC
        M["auroc"][name] = ci(d, lambda x, c=col: S.auroc(x[c], x.correct)) if ok else None
    metrics[m] = M

metrics["_baselines"] = {"random_mixed_case_string_eip55_pass_rate": S.random_checksum_pass_rate(50000),
                         "single_case_flip_of_true_address_pass_rate": S.random_case_flip_pass_rate(items.address.tolist()[:200], 10),
                         "always_trust_auroc": 0.5,
                         "analytic_random_casing_of_gold_addresses_pass_rate": S.expected_random_checksum_pass_rate(items.address.tolist())}
json.dump(metrics, open(f"results/{a.prefix}_metrics.json", "w"), indent=1, default=float)
flat = []
for m, M in metrics.items():
    if m.startswith("_"):
        continue
    flat.append(dict(model=m, n=M["n_items"], parse_rate=M["parse_rate"], abstain=M["abstain_rate"],
                     acc=M["accuracy_all"]["value"], acc_lo=M["accuracy_all"]["ci_lo"], acc_hi=M["accuracy_all"]["ci_hi"],
                     conf_wrong=(M["confident_wrong_rate_answered(conf>=90)"] or {}).get("value"),
                     ece=(M["ece_10bin"] or {}).get("value"), brier=(M["brier"] or {}).get("value"),
                     **{f"auroc_{k}": (v or {}).get("value") for k, v in M["auroc"].items()}))
pd.DataFrame(flat).to_csv(f"results/{a.prefix}_metrics.csv", index=False)
print(pd.DataFrame(flat).to_string())
print("baselines", metrics["_baselines"])
