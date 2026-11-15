"""Build data/pool.csv (all cross-validated candidates) and data/items.csv (stratified sample).

Sources (raw copies in data/raw/, fetched 2026-11-14):
  coins_list.json  CoinGecko /coins/list?include_platform=true
  markets/*.json   CoinGecko /coins/markets (market-cap rank)
  uniswap.json     https://tokens.uniswap.org (Uniswap Labs default list)          -> Ethereum strict check
  cmc_uniswap.json https://api.coinmarketcap.com/data-api/v3/uniswap/all.json       -> Ethereum fallback check
  1inch.json       https://tokens.1inch.io/v1.2/1                                   -> Ethereum fallback check
  cg_base.json     https://tokens.coingecko.com/base/all.json                       -> Base check (required)
  superchain.json  https://static.optimism.io/optimism.tokenlist.json (+Uniswap 8453) -> Base independent check
"""
import glob, json, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, "src")
from hexeval import config

N_ETH = int(os.environ.get("HEX_N_PER_STRATUM", 100))
N_BASE = int(os.environ.get("HEX_N_BASE", 60))
R = "data/raw"


def ADDR_OK(a):
    return isinstance(a, str) and a.startswith("0x") and len(a) == 42


def low(a):
    return a.lower() if isinstance(a, str) else a


coins = json.load(open(f"{R}/coins_list.json"))
rank = {}
for f in sorted(glob.glob(f"{R}/markets/page_*.json")):
    for c in json.load(open(f)):
        if c.get("market_cap_rank"):
            rank.setdefault(c["id"], c["market_cap_rank"])
uni = json.load(open(f"{R}/uniswap.json"))["tokens"]
uni_eth = {low(t["address"]): t for t in uni if t["chainId"] == 1}
uni_base = {low(t["address"]): t for t in uni if t["chainId"] == 8453}
cmc_eth = {low(t["address"]): t for t in json.load(open(f"{R}/cmc_uniswap.json"))["tokens"] if t.get("chainId") == 1}
oneinch = {low(k): v for k, v in json.load(open(f"{R}/1inch.json")).items()}
cg_base = {low(t["address"]): t for t in json.load(open(f"{R}/cg_base.json"))["tokens"]}
sc_base = {low(t["address"]): t for t in json.load(open(f"{R}/superchain.json"))["tokens"] if t["chainId"] == 8453}

sym_count = pd.Series([c["symbol"].lower() for c in coins]).value_counts().to_dict()
addr_owners = {}
for c in coins:
    for plat, a in (c.get("platforms") or {}).items():
        if ADDR_OK(a):
            addr_owners.setdefault((plat, low(a)), set()).add(c["id"])


def sym_agree(t, sym):
    return t is None or str(t.get("symbol", "")).lower() == sym.lower()


rows = []
for c in coins:
    pl = c.get("platforms") or {}
    sym = c["symbol"]
    for chain in ("ethereum", "base"):
        a = low(pl.get(chain))
        if not ADDR_OK(a) or len(addr_owners.get((chain, a), ())) > 1:
            continue  # missing, malformed, or ambiguous (address shared by 2+ CoinGecko coins)
        if chain == "ethereum":
            s_uni, s_cmc, s_1i = a in uni_eth, a in cmc_eth, a in oneinch
            if not (s_uni or s_cmc or s_1i):
                continue
            if not all(sym_agree(d.get(a), sym) for d in (uni_eth, cmc_eth, oneinch) if a in d):
                continue
            tier = "strict" if s_uni else "extended"
            srcs = "+".join(n for n, f in (("coingecko", 1), ("uniswap", s_uni), ("cmc", s_cmc), ("1inch", s_1i)) if f)
        else:
            if a not in cg_base or not sym_agree(cg_base[a], sym):
                continue
            s_ind = (a in uni_base and sym_agree(uni_base[a], sym)) or (a in sc_base and sym_agree(sc_base[a], sym))
            tier = "strict" if s_ind else "extended"
            srcs = "coingecko+cg_tokenlist" + ("+uniswap/superchain" if s_ind else "")
        other = {p: low(v) for p, v in pl.items() if ADDR_OK(v) and p != chain}
        rows.append(dict(coin_id=c["id"], name=c["name"], symbol=sym.upper(), chain=chain, address=a,
                         rank=rank.get(c["id"]), xval_tier=tier, xval_sources=srcs,
                         symbol_collisions=sym_count.get(sym.lower(), 1),
                         shared_symbol=sym_count.get(sym.lower(), 1) >= 2,
                         other_chain_addresses=json.dumps(other)))
pool = pd.DataFrame(rows)
n_unranked = int(pool["rank"].isna().sum())
pool = pool[pool["rank"].notna()].copy()
pool["rank"] = pool["rank"].astype(int)
pool["stratum"] = ["base" if ch == "base" else ("top100" if r <= 100 else ("rank100_1000" if r <= 1000 else "rank1000plus"))
                   for r, ch in zip(pool["rank"], pool["chain"])]
rng = np.random.default_rng(config.SEED)
pool["shuffle_key"] = rng.random(len(pool))
pool["tier_order"] = (pool["xval_tier"] != "strict").astype(int)
# strict tier first, then a fixed random key -> samples of any size are nested and reproducible
pool = pool.sort_values(["stratum", "tier_order", "shuffle_key"]).drop(columns="tier_order")
os.makedirs("data", exist_ok=True)
pool.to_csv("data/pool.csv", index=False)

want = {"top100": N_ETH, "rank100_1000": N_ETH, "rank1000plus": N_ETH, "base": N_BASE}
items = pd.concat([pool[pool.stratum == s].head(n) for s, n in want.items()]).reset_index(drop=True)
items.insert(0, "item_id", [f"{ch[:3]}-{i:04d}" for i, ch in enumerate(items["chain"])])
items.to_csv("data/items.csv", index=False)

universe = {}
for (plat, a), ids in addr_owners.items():
    universe.setdefault(a, []).extend(f"cg:{plat}:{i}" for i in ids)
for nm, d in (("uniswap:1", uni_eth), ("uniswap:8453", uni_base), ("cmc:1", cmc_eth), ("1inch:1", oneinch),
              ("cg_base:8453", cg_base), ("superchain:8453", sc_base)):
    for a in d:
        universe.setdefault(a, []).append(nm)
json.dump(universe, open("data/address_universe.json", "w"))

g = lambda df, cols: {"/".join(map(str, k)) if isinstance(k, tuple) else str(k): int(v) for k, v in df.groupby(cols).size().items()}
summary = {"coingecko_coins": len(coins), "ranked_coins": len(rank), "candidates_dropped_unranked": n_unranked,
           "pool_by_stratum_tier": g(pool, ["stratum", "xval_tier"]), "items_by_stratum": g(items, ["stratum"]),
           "items_by_stratum_tier": g(items, ["stratum", "xval_tier"]),
           "items_shared_symbol": int(items.shared_symbol.sum()), "address_universe_size": len(universe),
           "targets": want}
json.dump(summary, open("data/dataset_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
