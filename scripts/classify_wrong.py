"""Classify every wrong, answered greedy address. Aggregates -> results/<prefix>_destinations.csv;
per-token details (potential impostor contracts) -> private/<prefix>_destinations_detail.csv (gitignored, ethics)."""
import argparse, json, os, sys
import pandas as pd
sys.path.insert(0, "src")
from hexeval.onchain import classify, CATEGORIES

ap = argparse.ArgumentParser(); ap.add_argument("--prefix", default="full"); a = ap.parse_args()
P = pd.read_csv(f"results/{a.prefix}_per_item.csv")
items = pd.read_csv("data/items.csv").set_index("item_id")
universe = json.load(open("data/address_universe.json"))
W = P[(~P.correct) & (P.answered)].copy()
out = []
for _, r in W.iterrows():
    pred = r.pred if isinstance(r.pred, str) else r.raw_pred
    cat, det = classify(pred, r.chain, json.loads(items.loc[r.item_id].other_chain_addresses), universe, r.gold)
    out.append(dict(item_id=r.item_id, model=r.model, stratum=r.stratum, chain=r.chain, confidence=r.confidence,
                    pred=pred, category=cat, details=json.dumps(det)))
D = pd.DataFrame(out, columns=["item_id", "model", "stratum", "chain", "confidence", "pred", "category", "details"])
os.makedirs("private", exist_ok=True)
D.to_csv(f"private/{a.prefix}_destinations_detail.csv", index=False)
agg = D.groupby(["model", "category"]).size().unstack(fill_value=0).reindex(columns=CATEGORIES, fill_value=0)
agg["n_wrong"] = agg.sum(axis=1)
agg.loc["ALL"] = agg.sum()
agg.to_csv(f"results/{a.prefix}_destinations.csv")
print(agg.to_string())
