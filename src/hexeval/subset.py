"""Pre-registered item priority order + shared-subset selection (added 15 Nov 2026).

Priority order is fixed from data/items.csv alone (no model output is used):
within each stratum keep the file order (items were written in random `shuffle_key` order at
dataset build time), then interleave strata proportionally by within-stratum quantile
q = (pos + 0.5) / n_stratum. Any prefix of this order is therefore an approximately
proportionally-stratified random subset of the full design. run_eval.py queries items in this
order, so if a model cannot finish, every model still covers the same balanced prefix.

The shared subset reported by finalize = the longest priority prefix that every active model
has (almost) completed; items inside it with a persistent API failure for any model are listed
and excluded (tolerance HEX_SUBSET_TOL per model, default 3).
"""
import os
import pandas as pd

STRATUM_ORDER = ["top100", "rank100_1000", "rank1000plus", "base"]
N_ROWS_PER_ITEM = 6   # 3 greedy paraphrases + 3 T=1 samples


def priority_order(items):
    it = items.copy()
    it["_pos"] = it.groupby("stratum").cumcount()
    it["_n"] = it.groupby("stratum").item_id.transform("size")
    it["_q"] = (it._pos + 0.5) / it._n
    it["_s"] = it.stratum.map({s: i for i, s in enumerate(STRATUM_ORDER)}).fillna(99)
    it = it.sort_values(["_q", "_s"], kind="mergesort")
    return it.drop(columns=["_pos", "_n", "_q", "_s"])


def complete_matrix(R, models):
    """bool DataFrame (item_id x model): all 6 rows present without error."""
    ok = R[R.error.isna()].drop_duplicates(["item_id", "model", "paraphrase", "temperature", "sample_idx"])
    c = ok.groupby(["item_id", "model"]).size().unstack(fill_value=0)
    c = c.reindex(columns=models, fill_value=0)
    return c >= N_ROWS_PER_ITEM


def shared_subset(items, R, models, tol=None):
    tol = int(os.environ.get("HEX_SUBSET_TOL", 3)) if tol is None else tol
    order = priority_order(items).item_id.tolist()
    C = complete_matrix(R, models).reindex(order, fill_value=False)
    frontier = {}
    for m in models:
        miss, k = 0, 0
        for iid in order:
            if not C.at[iid, m]:
                miss += 1
                if miss > tol:
                    break
            k += 1
        # trim trailing misses so the frontier ends on a completed item
        while k > 0 and not C.at[order[k - 1], m]:
            k -= 1
        frontier[m] = k
    K = min(frontier.values()) if frontier else 0
    prefix = order[:K]
    keep = [i for i in prefix if C.loc[i, models].all()]
    excluded = [i for i in prefix if i not in set(keep)]
    return dict(K=K, items=keep, excluded=excluded, frontier=frontier,
                n_complete_per_model={m: int(C[m].sum()) for m in models}, n_items_total=len(order))
