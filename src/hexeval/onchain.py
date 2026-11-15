"""Classify where a wrong address would send funds. JSON-RPC results are cached on disk and rate-limited."""
import hashlib, json, os, threading, time
import requests
from . import config

CACHE = "cache/rpc"
_lock = threading.Lock()
_last = {}
MIN_INTERVAL = float(os.environ.get("HEX_RPC_INTERVAL", 0.25))
CATEGORIES = ["invalid_format", "same_token_other_chain", "other_known_token", "contract_code", "used_eoa", "never_used"]


def rpc(chain, method, params):
    key = hashlib.sha256(json.dumps([chain, method, params]).encode()).hexdigest()
    p = os.path.join(CACHE, key[:2], key + ".json")
    if os.path.exists(p):
        return json.load(open(p))["result"]
    url = config.RPC[chain]
    for attempt in range(6):
        with _lock:
            wait = MIN_INTERVAL - (time.time() - _last.get(chain, 0))
            if wait > 0:
                time.sleep(wait)
            _last[chain] = time.time()
        try:
            r = requests.post(url, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params}, timeout=30)
            j = r.json()
            if "result" in j:
                os.makedirs(os.path.dirname(p), exist_ok=True)
                json.dump({"chain": chain, "method": method, "params": params, "result": j["result"], "ts": time.time()}, open(p, "w"))
                return j["result"]
        except Exception:
            pass
        time.sleep(2 ** attempt)
    raise RuntimeError(f"rpc failed {chain} {method}")


def classify(pred, chain, other_chain_addresses, universe, gold=None):
    """Returns (category, details). `other_chain_addresses`: {platform: address} of the same coin."""
    from .parse import ADDR_RE
    if not isinstance(pred, str) or not ADDR_RE.fullmatch(pred):
        return "invalid_format", {}
    a = pred.lower()
    if a in set(other_chain_addresses.values()):
        return "same_token_other_chain", {"platforms": [k for k, v in other_chain_addresses.items() if v == a]}
    if a in universe and a != (gold or "").lower():
        return "other_known_token", {"sources": universe[a][:5]}
    code = rpc(chain, "eth_getCode", [a, "latest"])
    if code and code not in ("0x", "0x0"):
        return "contract_code", {"code_len": (len(code) - 2) // 2}
    nonce = int(rpc(chain, "eth_getTransactionCount", [a, "latest"]), 16)
    bal = int(rpc(chain, "eth_getBalance", [a, "latest"]), 16)
    if nonce > 0 or bal > 0:
        return "used_eoa", {"nonce": nonce, "balance_wei": str(bal)}
    return "never_used", {}
