"""Restrict responses to the active models and the pre-registered shared item subset (hexeval/subset.py).
Writes results/<prefix>_responses_shared.csv (input to score.py) and results/<prefix>_subset.json.
Dropped models (config.DROPPED_MODELS) are excluded here, so no reported metric includes them.
  python scripts/shared_subset.py --prefix full
"""
import argparse, json, sys
import pandas as pd
sys.path.insert(0, "src")
from hexeval import config
from hexeval.subset import shared_subset, priority_order

ap = argparse.ArgumentParser()
ap.add_argument("--prefix", default="full")
ap.add_argument("--responses", default="results/responses.csv")
a = ap.parse_args()
R = pd.read_csv(a.responses)
items = pd.read_csv("data/items.csv")
dropped = getattr(config, "DROPPED_MODELS", {})
models = [m for m in config.ACTIVE_MODELS if m not in dropped]
present = [m for m in models if m in set(R.model)]
S = shared_subset(items, R, present)
keep = set(S["items"])
out = R[R.model.isin(present) & R.item_id.isin(keep)]
out.to_csv(f"results/{a.prefix}_responses_shared.csv", index=False)
st = items.set_index("item_id").loc[S["items"]].stratum.value_counts().to_dict() if S["items"] else {}
full_st = items.stratum.value_counts().to_dict()
meta = {"models": present, "dropped_models": dropped, "n_items_full_design": S["n_items_total"],
        "priority_prefix_K": S["K"], "n_shared_items": len(S["items"]), "excluded_in_prefix_failed_calls": S["excluded"],
        "frontier_per_model": S["frontier"], "complete_items_per_model": S["n_complete_per_model"],
        "shared_items_by_stratum": st, "full_items_by_stratum": full_st,
        "is_full_design": len(S["items"]) == S["n_items_total"]}
json.dump(meta, open(f"results/{a.prefix}_subset.json", "w"), indent=1)
print(json.dumps({k: v for k, v in meta.items() if k != "excluded_in_prefix_failed_calls"}, indent=1),
      "excluded:", len(S["excluded"]))
