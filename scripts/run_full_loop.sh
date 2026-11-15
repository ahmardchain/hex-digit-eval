#!/usr/bin/env bash
# Resumable full run: reruns run_eval.py (cached) until every model reports "done" or the stop time passes
# (default Mon 16 Nov 2026 09:00 UTC). 15 Nov 2026: inside a pass, models now WAIT for Groq's continuously
# refilling daily token bucket / the next UTC day instead of stopping (HEX_WAIT_QUOTA=1, never past stop time),
# and transient failures (gemma timeouts / 5xx) are retried in-pass (HEX_RETRY_ROUNDS) and again next pass.
cd "$(dirname "$0")/.."
. .venv/bin/activate
[ -f .env ] && set -a && . ./.env && set +a
export HEX_STOP_UTC="${HEX_STOP_UTC:-2026-11-16T09:00:00Z}"
export HEX_WAIT_QUOTA="${HEX_WAIT_QUOTA:-1}" HEX_MAX_ERRORS="${HEX_MAX_ERRORS:-300}"
export HEX_RETRY_ROUNDS="${HEX_RETRY_ROUNDS:-6}" HEX_RETRY_SLEEP="${HEX_RETRY_SLEEP:-300}"
STOP_EPOCH=$(date -d "$HEX_STOP_UTC" +%s)
while :; do
  echo "=== pass $(date -u +%FT%TZ) models=${HEX_MODELS:-default}"
  python scripts/run_eval.py
  if python - <<'PY'
import json,sys; s=json.load(open("results/responses_status.json"))["status"]
sys.exit(0 if s and all(v=="done" for v in s.values()) else 1)
PY
  then echo "ALL DONE $(date -u +%FT%TZ)"; break; fi
  [ "$(date +%s)" -ge "$STOP_EPOCH" ] && { echo "stop time reached"; break; }
  sleep "${HEX_SLEEP:-900}"
done
