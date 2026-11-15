#!/usr/bin/env bash
# Refresh RESULTS.md from partial data every hour while the full run loop is alive; final refresh when it exits.
cd "$(dirname "$0")/.."
while pgrep -f run_full_loop.sh >/dev/null; do
  [ -f results/responses.csv ] && ./scripts/finalize.sh full > results/finalize.log 2>&1
  sleep 3600
done
./scripts/finalize.sh full > results/finalize.log 2>&1; echo "final $(date -u +%FT%TZ)" >> results/finalize.log
