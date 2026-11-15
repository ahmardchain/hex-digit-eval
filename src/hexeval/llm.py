"""Multi-provider free-tier chat client: disk cache, per-model RPM/RPD throttling,
resumable, and a hard $0 budget (non-free models are refused; any response that
reports a non-zero cost aborts the run).

Cache key = sha256(alias/model id, prompt, temperature, sample_idx, max_tokens, extra).
"""
import datetime as dt, hashlib, json, os, re, threading, time
from collections import deque
import requests
from . import config

_lock = threading.Lock()


class QuotaExhausted(RuntimeError):
    """Daily free-tier quota reached for a model; rerun later — cache makes it resumable."""


class PaidCallRefused(RuntimeError):
    pass


def cache_key(model_id, prompt, temperature, sample_idx, max_tokens, extra=None):
    blob = json.dumps([model_id, prompt, float(temperature), int(sample_idx), int(max_tokens), extra or {}], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _cache_path(key):
    return os.path.join(config.CACHE_DIR, key[:2], key + ".json")


def _today():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")


def _stop_epoch():
    """Run stop time (HEX_STOP_UTC, default Mon 16 Nov 2026 09:00 UTC)."""
    v = os.environ.get("HEX_STOP_UTC", "2026-11-16T09:00:00Z").replace("Z", "+00:00")
    return dt.datetime.fromisoformat(v).timestamp()


def _wait_enabled():
    return os.environ.get("HEX_WAIT_QUOTA", "0") == "1"


def _parse_wait(txt):
    """Seconds from 'try again in 1h2m3.5s' / '14.688s' / 'retry in 3s'; None if absent."""
    m = re.search(r"(?:try again|retry) in ([0-9.]+ms|(?:[0-9.]+h)?(?:[0-9.]+m(?!s))?(?:[0-9.]+s)?)", txt)
    if not m or not m.group(1):
        return None
    t = m.group(1)
    if t.endswith("ms") and "h" not in t and re.fullmatch(r"[0-9.]+ms", t):
        return float(t[:-2]) / 1000
    sec = 0.0
    for val, unit in re.findall(r"([0-9.]+)([hms])", t):
        sec += float(val) * {"h": 3600, "m": 60, "s": 1}[unit]
    return sec


def ledger_counts():
    """{(alias, utc_day): n_requests} and total cost from the ledger."""
    counts, cost = {}, 0.0
    if os.path.exists(config.LEDGER):
        for line in open(config.LEDGER):
            try:
                r = json.loads(line)
            except Exception:
                continue
            counts[(r["alias"], r["day"])] = counts.get((r["alias"], r["day"]), 0) + 1
            cost += float(r.get("cost", 0) or 0)
    return counts, cost


class Client:
    def __init__(self):
        self.counts, self.cost = ledger_counts()
        if self.cost > config.BUDGET_USD:
            raise PaidCallRefused(f"ledger shows ${self.cost} spent; budget is $0")
        self.windows = {}   # alias -> deque of request timestamps (last 60 s)

    def available(self, alias):
        s = config.spec(alias)
        return bool(config.load_key(config.PROVIDERS[s["provider"]]["key_env"]))

    def remaining_today(self, alias):
        return config.spec(alias)["rpd"] - self.counts.get((alias, _today()), 0)

    def _throttle(self, alias, rpm):
        with _lock:
            w = self.windows.setdefault(alias, deque())
        while True:
            with _lock:
                now = time.time()
                while w and now - w[0] > 60:
                    w.popleft()
                if len(w) < rpm:
                    w.append(now); return
                wait = 60 - (now - w[0]) + 0.05
            time.sleep(wait)

    def _key(self, alias, prompt, temperature, sample_idx, max_tokens=None):
        s = config.spec(alias)
        max_tokens = max_tokens or s.get("max_tokens") or (config.MAX_TOKENS_REASONING if "reasoning_effort" in s["extra"] and s["extra"]["reasoning_effort"] != "none" else config.MAX_TOKENS)
        return cache_key(s["id"], prompt, temperature, sample_idx, max_tokens, s["extra"])

    def cached_only(self, alias, prompt, temperature, sample_idx=0, max_tokens=None):
        """Return the cached record or None. Never makes a network call."""
        path = _cache_path(self._key(alias, prompt, temperature, sample_idx, max_tokens))
        if os.path.exists(path):
            rec = json.load(open(path)); rec["cached"] = True
            return rec
        return None

    def chat(self, alias, prompt, temperature, sample_idx=0, max_tokens=None):
        s = config.spec(alias)
        if not s.get("free"):
            raise PaidCallRefused(alias)
        if s["provider"] == "openrouter" and not s["id"].endswith(":free"):
            raise PaidCallRefused(s["id"])
        max_tokens = max_tokens or s.get("max_tokens") or (config.MAX_TOKENS_REASONING if "reasoning_effort" in s["extra"] and s["extra"]["reasoning_effort"] != "none" else config.MAX_TOKENS)
        key = cache_key(s["id"], prompt, temperature, sample_idx, max_tokens, s["extra"])
        path = _cache_path(key)
        if os.path.exists(path):
            rec = json.load(open(path)); rec["cached"] = True
            return rec
        prov = config.PROVIDERS[s["provider"]]
        api_key = config.load_key(prov["key_env"])
        if not api_key:
            raise RuntimeError(f"no {prov['key_env']} for {alias}")
        while True:
            with _lock:
                left = self.remaining_today(alias)
            if left > 0:
                break
            now = dt.datetime.now(dt.timezone.utc)
            nxt = (now + dt.timedelta(days=1)).replace(hour=0, minute=0, second=30, microsecond=0).timestamp()
            if not _wait_enabled() or nxt >= _stop_epoch():
                raise QuotaExhausted(f"{alias}: daily free quota {s['rpd']} used (UTC {_today()})")
            time.sleep(min(600, max(5, nxt - time.time())))
        self._throttle(alias, s["rpm"])
        body = {"model": s["id"], "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature, "max_tokens": max_tokens, **s["extra"]}
        if temperature == 0 and s["provider"] != "gemini":
            body["seed"] = config.SEED
        last, j = None, None
        attempt = 0
        while attempt < 7:
            attempt += 1
            try:
                r = requests.post(prov["base_url"], headers={"Authorization": f"Bearer {api_key}"}, json=body, timeout=120)
            except requests.RequestException as e:
                last = repr(e); time.sleep(2 ** attempt); continue
            if r.status_code == 200:
                j = r.json()
                if j.get("choices"):
                    break
                last = f"no choices: {str(j)[:200]}"
            elif r.status_code == 429:
                txt = r.text
                low = txt.lower()
                if "per day" in low or "perday" in low or "(rpd)" in low or "tokens per day" in low or "(tpd)" in low:
                    # Groq's daily limits refill continuously (token bucket): the error says how long to wait.
                    w = _parse_wait(txt)
                    if _wait_enabled() and w is not None and w <= 3 * 3600 and time.time() + w < _stop_epoch():
                        attempt -= 1   # a quota wait is not a failed attempt
                        time.sleep(w + 1 + 2 * (hash(prompt + str(sample_idx)) % 10)); continue
                    raise QuotaExhausted(f"{alias}: provider says daily quota hit: {txt[:400]}")
                ra = r.headers.get("retry-after")
                m = re.search(r"retry in ([0-9.]+)s", txt) or re.search(r"try again in ([0-9.]+)s", txt)
                wait = float(ra) if ra and ra.replace(".", "").isdigit() else (float(m.group(1)) + 1 if m else min(60, 5 * 2 ** attempt))
                last = f"429: {txt[:300]}"; time.sleep(min(wait, 120)); continue
            elif r.status_code in (400, 401, 402, 403, 404):
                raise RuntimeError(f"{alias} HTTP {r.status_code}: {r.text[:300]}")
            else:
                last = f"HTTP {r.status_code}: {r.text[:200]}"
            time.sleep(2 ** attempt)
        if j is None or not j.get("choices"):
            raise RuntimeError(f"{alias} failed: {last}")
        usage = j.get("usage") or {}
        cost = float(usage.get("cost") or 0.0)
        msg = j["choices"][0].get("message") or {}
        rec = {"alias": alias, "model": s["id"], "provider": s["provider"], "prompt": prompt,
               "temperature": temperature, "sample_idx": sample_idx, "max_tokens": max_tokens, "extra": s["extra"],
               "content": msg.get("content"), "finish_reason": j["choices"][0].get("finish_reason"),
               "usage": usage, "cost": cost, "ts": time.time()}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"; json.dump(rec, open(tmp, "w")); os.replace(tmp, path)
        with _lock:
            day = _today()
            self.counts[(alias, day)] = self.counts.get((alias, day), 0) + 1
            os.makedirs(os.path.dirname(config.LEDGER), exist_ok=True)
            with open(config.LEDGER, "a") as f:
                f.write(json.dumps({"alias": alias, "day": day, "cost": cost, "ts": rec["ts"], "key": key}) + "\n")
            self.cost += cost
        if cost > 0:
            raise PaidCallRefused(f"{alias} reported cost ${cost}; aborting (budget $0)")
        rec["cached"] = False
        return rec
