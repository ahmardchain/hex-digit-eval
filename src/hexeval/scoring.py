"""Scoring: correctness, calibration (ECE/Brier), AUROC per signal, baselines, bootstrap CIs."""
import random, string
import numpy as np
import pandas as pd
from eth_utils import is_checksum_address, to_checksum_address
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


def exact_match(pred, gold):
    return isinstance(pred, str) and isinstance(gold, str) and pred.lower() == gold.lower()


def is_mixed_case(addr):
    h = addr[2:] if isinstance(addr, str) else ""
    return any(c.isupper() for c in h) and any(c.islower() for c in h)


def eip55_status(addr):
    """'valid' / 'invalid' for mixed-case 40-hex addresses, 'not_mixed' for all-lower/all-upper, None if no address."""
    if not isinstance(addr, str) or len(addr) != 42:
        return None
    if not is_mixed_case(addr):
        return "not_mixed"
    return "valid" if is_checksum_address(addr) else "invalid"


def ece(conf01, correct, n_bins=10):
    conf01, correct = np.asarray(conf01, float), np.asarray(correct, float)
    bins = np.clip((conf01 * n_bins).astype(int), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            e += m.mean() * abs(conf01[m].mean() - correct[m].mean())
    return float(e)


def reliability_table(conf01, correct, n_bins=10):
    conf01, correct = np.asarray(conf01, float), np.asarray(correct, float)
    bins = np.clip((conf01 * n_bins).astype(int), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = bins == b
        rows.append(dict(bin=b, lo=b / n_bins, hi=(b + 1) / n_bins, n=int(m.sum()),
                         mean_conf=float(conf01[m].mean()) if m.any() else np.nan,
                         accuracy=float(correct[m].mean()) if m.any() else np.nan))
    return pd.DataFrame(rows)


def brier(conf01, correct):
    return float(np.mean((np.asarray(conf01, float) - np.asarray(correct, float)) ** 2))


def auroc(score, correct):
    score, correct = np.asarray(score, float), np.asarray(correct, int)
    m = ~np.isnan(score)
    if m.sum() < 2 or len(set(correct[m])) < 2:
        return np.nan
    return float(roc_auc_score(correct[m], score[m]))


def self_consistency(greedy_addr, samples):
    """Share of T=1 samples whose address equals the greedy address (case-insensitive). NaN if no greedy address."""
    if not isinstance(greedy_addr, str):
        return np.nan
    if not samples:
        return np.nan
    return float(np.mean([isinstance(s, str) and s.lower() == greedy_addr.lower() for s in samples]))


def random_checksum_pass_rate(n=100000, seed=0):
    """Baseline: chance a random mixed-case 40-hex string passes EIP-55 (theory ~ 2^-(#letters))."""
    rng = random.Random(seed)
    hexd = "0123456789abcdefABCDEF"
    passed = mixed = 0
    for _ in range(n):
        a = "0x" + "".join(rng.choice(hexd) for _ in range(40))
        if is_mixed_case(a):
            mixed += 1
            passed += is_checksum_address(a)
    return passed / max(mixed, 1)


def expected_random_checksum_pass_rate(addresses):
    """Analytic: a random casing of a real address passes EIP-55 w.p. 2^-(#hex letters a-f)."""
    v = [2.0 ** -sum(c.isalpha() for c in a[2:]) for a in addresses if isinstance(a, str)]
    return float(np.mean(v)) if v else np.nan


def random_case_flip_pass_rate(addresses, n_per=20, seed=0):
    """Baseline closer to model errors: take real checksummed addresses, flip the case of one letter."""
    rng = random.Random(seed)
    tot = ok = 0
    for a in addresses:
        cs = to_checksum_address(a)
        letters = [i for i, c in enumerate(cs) if i >= 2 and c.isalpha()]
        if not letters:
            continue
        for _ in range(n_per):
            i = rng.choice(letters)
            b = cs[:i] + cs[i].swapcase() + cs[i + 1:]
            tot += 1; ok += is_checksum_address(b)
    return ok / max(tot, 1)


def logistic_cv_scores(X, y, groups, n_splits=5, seed=0):
    """Out-of-fold predicted P(correct) from a logistic model, folds grouped by token."""
    X, y, groups = np.asarray(X, float), np.asarray(y, int), np.asarray(groups)
    out = np.full(len(y), np.nan)
    k = min(n_splits, len(set(groups)))
    if k < 2 or len(set(y)) < 2:
        return out
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        if len(set(y[tr])) < 2:
            out[te] = y[tr].mean(); continue
        m = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
        out[te] = m.predict_proba(X[te])[:, 1]
    return out


def bootstrap_ci(df, stat_fn, group_col="item_id", n_boot=1000, seed=0, alpha=0.05):
    """Cluster bootstrap over tokens. stat_fn(df)->float. Returns (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    point = stat_fn(df)
    ids = df[group_col].unique()
    idx = {g: np.where(df[group_col].values == g)[0] for g in ids}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(ids, len(ids), replace=True)
        rows = np.concatenate([idx[g] for g in pick])
        v = stat_fn(df.iloc[rows])
        if v is not None and not np.isnan(v):
            vals.append(v)
    if not vals:
        return point, np.nan, np.nan
    return point, float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))
