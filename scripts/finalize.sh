#!/usr/bin/env bash
# Score + on-chain classification + figures + RESULTS.md for a prefix (default full). Safe on partial data.
# 15 Nov 2026: full runs are scored on the shared pre-registered item subset of the active models
# (scripts/shared_subset.py); dropped models are excluded. Usage: finalize.sh [prefix] [out.md]
cd "$(dirname "$0")/.." && . .venv/bin/activate
P="${1:-full}"; OUT="${2:-$([ "$P" = pilot ] && echo RESULTS_pilot.md || echo RESULTS.md)}"
if [ "$P" = pilot ]; then
  python scripts/score.py --prefix "$P" && python scripts/classify_wrong.py --prefix "$P" && python scripts/figures.py --prefix "$P" && python scripts/make_results_md.py --prefix "$P" --out "$OUT"
else
  python scripts/shared_subset.py --prefix "$P" && \
  python scripts/score.py --prefix "$P" --responses "results/${P}_responses_shared.csv" && python scripts/classify_wrong.py --prefix "$P" && python scripts/figures.py --prefix "$P" && python scripts/make_results_md.py --prefix "$P" --out "$OUT"
fi
