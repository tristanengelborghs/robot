#!/usr/bin/env bash
# Install the simulator stack. Only needed for evaluation — training never imports it.
# LIBERO is pinned to a commit: task definitions and init states have changed over
# time, and an unpinned install makes your numbers uncomparable to anyone else's.
set -euo pipefail

LIBERO_COMMIT="${LIBERO_COMMIT:-master}"
THIRD_PARTY="${THIRD_PARTY:-third_party}"

# --- interpreter check ------------------------------------------------------
# mujoco requires Python >= 3.10. On 3.9 pip finds no wheel, falls back to
# building from source, and dies on a missing MUJOCO_PATH — an error that reads
# like a MuJoCo problem but is really an interpreter-version problem.
PYV=$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')
if [ "$(printf '%s\n3.10\n' "$PYV" | sort -V | head -1)" != "3.10" ]; then
  cat >&2 <<MSG
Python $PYV is too old for the simulator stack — mujoco needs >= 3.10.

Rebuild the environment on a newer interpreter, then re-run this script:

    deactivate 2>/dev/null || true
    rm -rf .venv
    python3.11 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt && pip install -e .
MSG
  exit 1
fi

echo "==> simulator dependencies (Python $PYV)"
# only-if-needed keeps pip from upgrading numpy/torch out from under the
# training stack to satisfy robosuite's unbounded lower-bound requirements.
pip install --upgrade-strategy only-if-needed -r requirements-sim.txt

mkdir -p "$THIRD_PARTY"
if [ ! -d "$THIRD_PARTY/LIBERO" ]; then
  echo "==> cloning LIBERO"
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$THIRD_PARTY/LIBERO"
fi

cd "$THIRD_PARTY/LIBERO"
git fetch --all --tags
git checkout "$LIBERO_COMMIT"
RESOLVED=$(git rev-parse HEAD)
# --no-deps: LIBERO's requirements.txt is unpinned and pulls its own torch build.
pip install --no-deps -e .

# LIBERO's top-level `libero/` directory has no __init__.py, so upstream's
# find_packages() matches nothing and the editable install registers an EMPTY
# package mapping. `import libero` then only works from inside the LIBERO
# checkout, where cwd happens to be on sys.path — which is why the breakage is
# easy to miss. Put the checkout on the path explicitly so it imports from
# anywhere, as the PEP 420 namespace package it actually is.
LIBERO_ROOT=$(pwd)
cd - >/dev/null
SITE=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
echo "$LIBERO_ROOT" > "$SITE/libero_checkout.pth"
echo "==> registered $LIBERO_ROOT on sys.path (libero_checkout.pth)"

echo "$RESOLVED" > .libero_commit

# On first import LIBERO calls input() to ask where datasets should live, which
# hangs forever under any non-interactive shell — CI, nohup, a setup script.
# Write the config up front, pointing datasets at this repo's data/libero.
# An existing config is kept if its paths still resolve. It is rewritten if they
# do not: the file is absolute paths, so moving or renaming the checkout leaves
# it pointing at directories that no longer exist, and the evaluator then fails
# on the first BDDL file it opens.
LIBERO_PKG="$LIBERO_ROOT/libero/libero"
CONFIG_DIR="${LIBERO_CONFIG_PATH:-$HOME/.libero}"
CONFIG="$CONFIG_DIR/config.yaml"
mkdir -p "$CONFIG_DIR"
WRITE_CONFIG=1
if [ -f "$CONFIG" ]; then
  EXISTING=$(sed -n 's/^bddl_files: *//p' "$CONFIG")
  if [ -n "$EXISTING" ] && [ -d "$EXISTING" ]; then
    echo "==> $CONFIG exists and its paths resolve — left untouched"
    WRITE_CONFIG=0
  else
    echo "==> $CONFIG points at '$EXISTING', which does not exist — rewriting"
  fi
fi
if [ "$WRITE_CONFIG" = 1 ]; then
  cat > "$CONFIG" <<YAML
benchmark_root: $LIBERO_PKG
bddl_files: $LIBERO_PKG/bddl_files
init_states: $LIBERO_PKG/init_files
datasets: $(pwd)/data/libero
assets: $LIBERO_PKG/assets
YAML
  echo "==> wrote $CONFIG (datasets -> $(pwd)/data/libero)"
fi

echo
echo "==> verifying the stack imports and numpy was not clobbered"
python - <<'PY'
import numpy, sys
assert numpy.__version__.startswith("1."), f"numpy {numpy.__version__} — training stack needs 1.x"
import mujoco, robosuite
from libero.libero import benchmark, get_libero_path
suite = benchmark.get_benchmark_dict()["libero_object"]()
n = suite.n_tasks
assert n > 0, "libero_object exposes no tasks"
print(f"    numpy {numpy.__version__} | mujoco {mujoco.__version__} | robosuite {robosuite.__version__}")
print(f"    libero_object exposes {n} tasks, e.g. {suite.get_task(0).language!r}")
print(f"    bddl_files -> {get_libero_path('bddl_files')}")
PY

echo
echo "==> LIBERO installed at commit $RESOLVED (recorded in .libero_commit)"
echo "    Put this hash in your results table."
echo
case "$(uname -s)" in
  Darwin)
    echo "macOS offscreen rendering — EGL is Linux-only, use GLFW here:"
    echo "    export MUJOCO_GL=glfw"
    echo "    (rollouts run on CPU physics and are slow; fine for verifying the"
    echo "     eval path, do the real runs on a Linux GPU box)" ;;
  *)
    echo "Headless Linux needs an EGL context:"
    echo "    export MUJOCO_GL=egl"
    echo "    export PYOPENGL_PLATFORM=egl" ;;
esac
