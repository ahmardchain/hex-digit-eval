#!/usr/bin/env bash
# One-command reproduce. Needs GEMINI_API_KEY and/or GROQ_API_KEY (free tiers) in the env or .env.
# Everything is cached: rerunning after a daily free-tier quota resets resumes where it stopped.
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate && pip install -q -r requirements.txt
[ -f .env ] && set -a && . ./.env && set +a
MODE="${1:-full}"            # full | pilot
[ -f data/raw/coins_list.json ] || python scripts/fetch_sources.py
[ -d data/raw/markets ] && [ "$(ls data/raw/markets | wc -l)" -ge 24 ] || python scripts/fetch_markets.py 24
python scripts/build_dataset.py
python -m pytest -q tests
if [ "$MODE" = pilot ]; then python scripts/run_eval.py --pilot 30; P=pilot; else python scripts/run_eval.py; P=full; fi
python scripts/score.py --prefix $P
python scripts/classify_wrong.py --prefix $P
python scripts/figures.py --prefix $P
echo "done -> results/${P}_metrics.json, figures/${P}_*.png"
