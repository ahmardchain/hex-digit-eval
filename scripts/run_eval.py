"""Query every (item, model): 3 paraphrases at T=0 + N_SAMPLES_T1 samples at T=1 (paraphrase 0).
Resumable: every call is cached on disk; rerun the same command after a daily quota resets.

  python scripts/run_eval.py --pilot 30        # 30 items (stratified), all active models with keys
  python scripts/run_eval.py                   # full items.csv
  HEX_MODELS=llama-3.3-70b python scripts/run_eval.py --pilot 30
"""
import argparse, json, os, sys, threading, time
import pandas as pd
sys.path.insert(0, "src")
from hexeval import config
from hexeval.llm import Client, QuotaExhausted, PaidCallRefused
from hexeval.parse import parse_reply
from hexeval.prompts import PARAPHRASES, render
from hexeval.subset import priority_order

ap = argparse.ArgumentParser()
ap.add_argument("--pilot", type=int, default=0, help="number of items for a pilot (stratified)")
ap.add_argument("--items", default="data/items.csv")
ap.add_argument("--out", default=None)
args = ap.parse_args()

items = pd.read_csv(args.items)
if args.pilot:
    per = max(1, args.pilot // items.stratum.nunique())
    items = items.groupby("stratum", group_keys=False).head(per)
    items = pd.concat([items, pd.read_csv(args.items).drop(items.index, errors="ignore").head(args.pilot - len(items))]) if len(items) < args.pilot else items.head(args.pilot)
if not args.pilot:
    # 15 Nov 2026: query items in the pre-registered, stratum-interleaved priority order (see hexeval/subset.py)
    # so every model covers the same balanced prefix first. Cache keys do not depend on order.
    items = priority_order(items).reset_index(drop=True)
out = args.out or ("results/pilot_responses.csv" if args.pilot else "results/responses.csv")
os.makedirs("results", exist_ok=True)

client = Client()
models = [m for m in config.ACTIVE_MODELS if client.available(m)]
skipped = [m for m in config.ACTIVE_MODELS if m not in models]
print(f"items={len(items)} models={models} skipped(no key)={skipped}", flush=True)

jobs = []
for _, it in items.iterrows():
    chain = config.CHAIN_NAMES[it.chain]
    for p in range(config.N_GREEDY_PARAPHRASES):
        jobs.append((it, p, 0.0, 0))
    for s in range(config.N_SAMPLES_T1):
        jobs.append((it, 0, 1.0, s))

rows, status, errs, lock = {}, {}, {}, threading.Lock()   # rows keyed by (model, item, paraphrase, temp, sample)
MAX_ERRS = int(os.environ.get("HEX_MAX_ERRORS", 20))
RETRY_ROUNDS = int(os.environ.get("HEX_RETRY_ROUNDS", 0))     # in-pass retries of failed calls (transient 5xx/timeouts)
RETRY_SLEEP = int(os.environ.get("HEX_RETRY_SLEEP", 300))


from concurrent.futures import ThreadPoolExecutor
CONC = int(os.environ.get("HEX_CONCURRENCY", 6))


def _jk(alias, job):
    it, p, temp, s = job
    return (alias, it.item_id, p, temp, s)


def worker(alias):
    stop = threading.Event()
    todo = jobs
    for rnd in range(RETRY_ROUNDS + 1):
        if rnd:
            print(f"{alias}: retry round {rnd}: {len(todo)} failed calls, sleeping {RETRY_SLEEP}s", flush=True)
            time.sleep(RETRY_SLEEP)
            with lock:
                errs[alias] = 0
        with ThreadPoolExecutor(max_workers=CONC) as ex:
            list(ex.map(lambda j: one(alias, j, stop), todo))
        with lock:
            todo = [j for j in jobs if rows.get(_jk(alias, j), {}).get("error")]
        if not todo or stop.is_set():
            break
    if todo and not stop.is_set():
        with lock:
            errs[alias] = len(todo)
    status.setdefault(alias, "done")
    print(alias, status[alias], flush=True)


def one(alias, job, stop):
    it, p, temp, s = job
    prompt = render(p, config.CHAIN_NAMES[it.chain], it["name"], it.symbol)
    if stop.is_set():
        # model stopped for this pass (quota/errors): still record anything already cached, no network call
        rec = client.cached_only(alias, prompt, temp, s)
        if rec is None:
            return
    else:
        try:
            rec = client.chat(alias, prompt, temp, s)
        except (QuotaExhausted, PaidCallRefused) as e:
            status[alias] = f"stopped: {e}"; stop.set(); print(status[alias], flush=True); return
        except Exception as e:
            with lock:
                rows[_jk(alias, job)] = dict(item_id=it.item_id, model=alias, paraphrase=p, temperature=temp, sample_idx=s,
                                             content=None, error=str(e)[:300])
                errs[alias] = errs.get(alias, 0) + 1
                if errs[alias] > MAX_ERRS:
                    status[alias] = f"too many errors, last: {e}"; stop.set()
            return
    if True:
        pr = parse_reply(rec["content"])
        with lock:
            rows[_jk(alias, job)] = (dict(item_id=it.item_id, model=alias, paraphrase=p, temperature=temp, sample_idx=s,
                             content=rec["content"], finish_reason=rec.get("finish_reason"),
                             prompt_tokens=(rec.get("usage") or {}).get("prompt_tokens"),
                             completion_tokens=(rec.get("usage") or {}).get("completion_tokens"),
                             cost=rec.get("cost", 0.0), cached=rec["cached"], error=None, **pr))


threads = [threading.Thread(target=worker, args=(m,)) for m in models]
[t.start() for t in threads]
def snapshot():
    with lock:
        df = pd.DataFrame(list(rows.values()))
    df.to_csv(out + ".tmp", index=False); os.replace(out + ".tmp", out)


tick = 0
while any(t.is_alive() for t in threads):
    time.sleep(15); tick += 1
    print(f"progress rows={len(rows)}/{len(jobs) * len(models)}", flush=True)
    if tick % 8 == 0:
        snapshot()   # partial results on disk every ~2 min
[t.join() for t in threads]
pd.DataFrame(list(rows.values())).to_csv(out, index=False)
for m in models:   # a model with failed calls is not "done": next pass retries them (uncached)
    if status.get(m) == "done" and errs.get(m):
        status[m] = f"incomplete: {errs[m]} failed calls (retry next pass)"
json.dump({"status": status, "skipped_no_key": skipped, "models": models, "dropped_models": getattr(config, "DROPPED_MODELS", {}), "n_items": len(items), "n_jobs_per_model": len(jobs)},
          open(out.replace(".csv", "_status.json"), "w"), indent=1)
print(json.dumps(status, indent=1))
