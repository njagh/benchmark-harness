#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

python -m bench_harness run \
  --suite smoke \
  --models agent-code,qwen-dense,max-brain \
  --runs 1 \
  --out "runs/$(date +%Y-%m-%d)-smoke"
