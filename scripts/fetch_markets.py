"""Fetch CoinGecko /coins/markets pages (market-cap rank) with polite rate limiting. Cached per page."""
import json, os, sys, time, requests
OUT = "data/raw/markets"; os.makedirs(OUT, exist_ok=True)
pages = int(sys.argv[1]) if len(sys.argv) > 1 else 20
for p in range(1, pages + 1):
    fn = f"{OUT}/page_{p:03d}.json"
    if os.path.exists(fn): continue
    for attempt in range(8):
        r = requests.get("https://api.coingecko.com/api/v3/coins/markets",
                         params=dict(vs_currency="usd", order="market_cap_desc", per_page=250, page=p), timeout=60)
        if r.status_code == 200:
            json.dump(r.json(), open(fn, "w")); print("page", p, len(r.json()), flush=True); break
        wait = 30 * (attempt + 1); print("page", p, "status", r.status_code, "sleep", wait, flush=True); time.sleep(wait)
    else:
        sys.exit(f"failed page {p}")
    time.sleep(13)
