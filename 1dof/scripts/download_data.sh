#!/usr/bin/env bash
# Fetch LIBERO demonstration datasets. Single-digit GB per 10-task suite.
set -euo pipefail

SUITES="${*:-libero_object libero_spatial}"
DEST="${DEST:-data/libero}"

if ! python -c "import libero" 2>/dev/null; then
  echo "LIBERO is not installed — run scripts/setup_libero.sh first." >&2
  exit 1
fi

mkdir -p "$DEST"
for suite in $SUITES; do
  echo "==> $suite"
  python -m libero.libero.benchmark.download_datasets --datasets "$suite" --download_dir "$DEST"
done

echo
echo "==> datasets under $DEST"
find "$DEST" -name '*.hdf5' | head -20
echo
echo "Next: python scripts/cache_language.py --data-root $DEST --suites $SUITES"
