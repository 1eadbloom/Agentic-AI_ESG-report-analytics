#!/usr/bin/env bash
# Verifies the HW4 required file structure.
# Exit 0 = everything present and no secrets committed; non-zero = at least one problem.
# Used by the GitHub Classroom autograder as a graded test (file-structure only; no pytest).
set -uo pipefail

status=0

required_files=(
  memory/__init__.py
  memory/bm25.py
  memory/store.py
  memory/core.py
  memory/cli.py
  tests/test_memory.py
  benchmark/run_benchmark.py
  README.md
  REPORT.md
  requirements.txt
)

required_dirs=(
  demo
)

for f in "${required_files[@]}"; do
  if [ ! -f "$f" ]; then
    echo "::error::missing required file: $f"
    status=1
  fi
done

for d in "${required_dirs[@]}"; do
  if [ ! -d "$d" ]; then
    echo "::error::missing required directory: $d"
    status=1
  fi
done

if [ -f .env ]; then
  echo "::error::.env must not be committed"
  status=1
fi

if [ "$status" -eq 0 ]; then
  echo "OK: all required files and directories present; no .env committed."
fi

exit "$status"
