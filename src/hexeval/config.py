"""Central configuration — ZERO-BUDGET edition.

Only free-tier endpoints are allowed. Every model is declared with the provider, its
free-tier rate limits (requests/minute, requests/day) and a `free=True` flag; the client
refuses anything else. OpenRouter is restricted to ids ending in ':free'.

Free-tier limits below are the published defaults as of Nov 2026 and are conservative
on purpose; override with env HEX_RPM_<ALIAS>/HEX_RPD_<ALIAS>. The client also honours
429 Retry-After and Groq x-ratelimit-* headers.
"""
import os

PROVIDERS = {
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
               "key_env": "GEMINI_API_KEY"},
    "groq": {"base_url": "https://api.groq.com/openai/v1/chat/completions", "key_env": "GROQ_API_KEY"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1/chat/completions", "key_env": "OPENROUTER_API_KEY"},
}

# alias -> spec
MODELS = {
    "gemini-2.5-flash":      {"provider": "gemini", "id": "gemini-2.5-flash", "rpm": 5, "rpd": 20,
                              "extra": {"reasoning_effort": "none"}, "free": True},
    "gemini-3.1-flash-lite": {"provider": "gemini", "id": "gemini-3.1-flash-lite", "rpm": 14, "rpd": 1400,
                              "extra": {"reasoning_effort": "none"}, "free": True},
    "gemma-4-31b":           {"provider": "gemini", "id": "gemma-4-31b-it", "rpm": 25, "rpd": 14000,
                              "extra": {}, "max_tokens": 1500, "free": True},
    "gpt-oss-120b":          {"provider": "groq", "id": "openai/gpt-oss-120b", "rpm": 25, "rpd": 990,
                              "extra": {"reasoning_effort": "low"}, "free": True},
    "qwen3.8-27b":           {"provider": "groq", "id": "qwen/qwen3.8-27b", "rpm": 25, "rpd": 990,
                              "extra": {"reasoning_effort": "none"}, "free": True},
    "gpt-oss-20b":           {"provider": "groq", "id": "openai/gpt-oss-20b", "rpm": 25, "rpd": 990,
                              "extra": {"reasoning_effort": "low"}, "free": True},
    # OpenRouter free models: ~50 requests/day total across all :free models without credits.
    "or-free-deepseek":      {"provider": "openrouter", "id": "deepseek/deepseek-chat-v3.1:free", "rpm": 15, "rpd": 45,
                              "extra": {}, "free": True},
}
# gemini-2.5-flash free tier is only 20 requests/day (measured 429, 14 Nov 2026) -> excluded from defaults
# 15 Nov 2026: gemini-3.1-flash-lite DROPPED (free tier = 500 requests/day -> could not finish 1,878 calls before
# the deadline). Its cached responses are kept on disk but excluded from the run and from every reported metric.
DEFAULT_MODELS = ["gemma-4-31b", "gpt-oss-120b", "gpt-oss-20b", "qwen3.8-27b"]
DROPPED_MODELS = {"gemini-3.1-flash-lite": "dropped 15 Nov 2026: free-tier cap of 500 requests/day made the full "
                  "1,878-call design impossible before the deadline (363/1,878 done); cached data kept, not reported"}
ACTIVE_MODELS = [m.strip() for m in os.environ.get("HEX_MODELS", ",".join(DEFAULT_MODELS)).split(",") if m.strip()]

BUDGET_USD = 0.0          # hard: never call a paid model
MAX_TOKENS = 120          # answer JSON is ~30 tokens; reasoning models get more (below)
MAX_TOKENS_REASONING = 600
N_GREEDY_PARAPHRASES = 3
N_SAMPLES_T1 = 3
SEED = 20261114
CHAIN_NAMES = {"ethereum": "Ethereum", "base": "Base"}
RPC = {"ethereum": "https://ethereum-rpc.publicnode.com", "base": "https://mainnet.base.org"}

CACHE_DIR = "cache/llm"
LEDGER = "cache/request_ledger.jsonl"   # one line per real (non-cached) request: model, day, cost


def spec(alias):
    s = dict(MODELS[alias])
    s["rpm"] = int(os.environ.get(f"HEX_RPM_{alias.upper().replace('-', '_').replace('.', '_')}", s["rpm"]))
    s["rpd"] = int(os.environ.get(f"HEX_RPD_{alias.upper().replace('-', '_').replace('.', '_')}", s["rpd"]))
    return s


def load_key(env_name):
    k = os.environ.get(env_name)
    if k:
        return k.strip()
    for p in (f"/workspace/secrets/{env_name.split('_')[0].lower()}.env", "/workspace/secrets/llm.env",
              "/workspace/hex-digit-eval/.env"):
        if os.path.exists(p):
            for line in open(p):
                line = line.strip()
                if line.startswith(env_name + "="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None
