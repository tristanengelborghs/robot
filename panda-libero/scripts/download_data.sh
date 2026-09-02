#!/usr/bin/env bash
# Fetch LIBERO demonstration datasets into data/libero/<suite>/*.hdf5.
# Single-digit GB per 10-task suite; ten files each when complete.
#
#     bash scripts/download_data.sh libero_goal
#     bash scripts/download_data.sh libero_goal libero_spatial
#
# Goes through LIBERO's own download helpers so the link table and the integrity
# check stay pinned with the checkout. Its command-line script is not used: it
# asks a y/n question on stdin, which hangs under nohup, make, or ssh -- and the
# module this script once called does not exist at the pinned commit.
set -euo pipefail

SUITES="${*:-libero_goal}"
DEST="${DEST:-data/libero}"
SOURCE="${SOURCE:-box}"     # box: the authors' UT Box links | hf: the Hugging Face mirror

if ! python -c "import libero" 2>/dev/null; then
  echo "LIBERO is not installed — run scripts/setup_libero.sh first." >&2
  exit 1
fi

mkdir -p "$DEST"
for suite in $SUITES; do
  echo "==> $suite -> $DEST/$suite"
  python - "$suite" "$DEST" "$SOURCE" <<'PY'
import sys
from pathlib import Path

from libero.libero.utils import download_utils as du

suite, dest, source = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
have = sorted((dest / suite).glob("*.hdf5"))
if len(have) == 10:
    print(f"    already complete: {len(have)} files, nothing to do")
    sys.exit(0)
if have:
    print(f"    {len(have)} of 10 files present — fetching the suite again")
if source == "hf":
    du.download_from_huggingface(suite, str(dest), check_overwrite=False)
else:
    # The zip unpacks to <dest>/<suite>/*.hdf5 and is deleted afterwards.
    du.download_url(du.DATASET_LINKS[suite], download_dir=str(dest), check_overwrite=False)
have = sorted((dest / suite).glob("*.hdf5"))
if len(have) != 10:
    raise SystemExit(f"    expected 10 .hdf5 files under {dest / suite}, found {len(have)}")
print(f"    complete: {len(have)} files")
PY
done

echo
echo "==> datasets under $DEST"
find "$DEST" -name '*.hdf5' | sort | head -40
echo
echo "Next: python scripts/cache_language.py --data-root $DEST --suites $SUITES"
