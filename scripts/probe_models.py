"""One cached request per model to verify the id, params and free-tier access."""
import sys; sys.path.insert(0, "src")
from hexeval import config
from hexeval.llm import Client
from hexeval.prompts import render
from hexeval.parse import parse_reply
c = Client()
for m in (sys.argv[1:] or config.ACTIVE_MODELS):
    try:
        r = c.chat(m, render(0, "Ethereum", "Tether", "USDT"), 0.0, 0)
        print(m, "OK", repr((r["content"] or "")[:160]), r.get("finish_reason"), r.get("usage", {}).get("completion_tokens"), parse_reply(r["content"]))
    except Exception as e:
        print(m, "FAIL", str(e)[:300])
