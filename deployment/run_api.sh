#!/usr/bin/env bash
# Laptop 1: start the PathSnap API.  Usage: deployment/run_api.sh [/path/to/evaluator_trips]
set -e
cd "$(dirname "$0")/.."
[ -n "$1" ] && export PATHSNAP_EVAL_DIR="$1"
echo "Eval trips: ${PATHSNAP_EVAL_DIR:-data/student_test}"
echo "LAN IP(s) for laptop 2:"; (hostname -I 2>/dev/null || ipconfig getifaddr en0) | tr ' ' '\n' | grep -v '^$'
uv sync -q
exec uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
