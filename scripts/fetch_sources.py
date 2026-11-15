"""Download the token-list sources into data/raw/ (CoinGecko markets pages: scripts/fetch_markets.py)."""
import os, requests
os.makedirs("data/raw", exist_ok=True)
SRC = {"coins_list.json": "https://api.coingecko.com/api/v3/coins/list?include_platform=true",
       "uniswap.json": "https://tokens.uniswap.org",
       "cg_base.json": "https://tokens.coingecko.com/base/all.json",
       "cmc_uniswap.json": "https://api.coinmarketcap.com/data-api/v3/uniswap/all.json",
       "1inch.json": "https://tokens.1inch.io/v1.2/1",
       "superchain.json": "https://static.optimism.io/optimism.tokenlist.json"}
for fn, url in SRC.items():
    p = f"data/raw/{fn}"
    if not os.path.exists(p):
        r = requests.get(url, timeout=120); r.raise_for_status(); open(p, "wb").write(r.content); print("fetched", fn)
