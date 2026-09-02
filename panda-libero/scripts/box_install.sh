#!/usr/bin/env bash
# Runs ON the GPU host (not inside the Isaac container): builds the training and
# simulator stack for panda-libero in ./.venv. Idempotent — rerun after a reboot
# or a sync and it only does what is missing.
#
#     make box-install          # from the laptop; this file is piped over ssh
#
# The host is an Ubuntu 24.04 AWS g6.xlarge (one L4, 24 GB) provisioned by Brev
# for Isaac Lab. It ships Docker and an NVIDIA driver; what Python it carries was
# not known when this was written, so the script reports what it finds and falls
# back to `uv` for a 3.11 interpreter, matching the laptop.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> host"
echo "    $(uname -a)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
  || echo "    nvidia-smi not found — is this the GPU box?"
echo "    disk: $(df -h . | awk 'NR==2{print $4" free of "$2}')"
echo "    cpus: $(nproc)   ram: $(free -g | awk '/Mem/{print $2}') GB"

# --- interpreter ------------------------------------------------------------
PY=""
for cand in python3.11 python3.12 python3.10; do
  if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
done
if [ -z "$PY" ]; then
  echo "==> no python3.10+ on PATH; installing uv and a 3.11 interpreter"
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  uv python install 3.11
  PY="$(uv python find 3.11)"
fi
echo "==> interpreter: $PY ($($PY --version))"

if [ ! -x .venv/bin/python ]; then
  echo "==> creating .venv"
  "$PY" -m venv .venv || { command -v uv >/dev/null && uv venv --python "$PY" .venv; }
fi
export PATH="$PWD/.venv/bin:$PATH"
python -m pip install -q -U pip

# --- headless rendering -------------------------------------------------------
# MuJoCo's EGL backend needs the EGL loader and GL libraries the driver plugs
# into. Harmless if already present; skipped when apt or sudo is unavailable.
if command -v apt-get >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
  echo "==> EGL/GL libraries"
  sudo apt-get install -y -qq libegl1 libgl1 libglib2.0-0 >/dev/null 2>&1 || echo "    apt-get install failed; continuing"
fi

# --- python stacks --------------------------------------------------------------
echo "==> training stack"
make install >/dev/null
echo "==> simulator stack (LIBERO pinned to $(cat .libero_commit 2>/dev/null || echo master))"
LIBERO_COMMIT="$(cat .libero_commit 2>/dev/null || echo master)" bash scripts/setup_libero.sh

echo
echo "==> verifying: CUDA, bf16 autocast, headless render"
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python - <<'PY'
import torch
print(f"    torch {torch.__version__}  cuda={torch.cuda.is_available()}", end="")
if torch.cuda.is_available():
    print(f"  {torch.cuda.get_device_name(0)}  {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        (torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda")).sum().item()
    print("    bf16 autocast ok")
else:
    print()
import mujoco
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos="0 0 1"/><geom size="0.1"/></worldbody></mujoco>')
r = mujoco.Renderer(m, 64, 64); r.update_scene(mujoco.MjData(m)); img = r.render()
print(f"    MUJOCO_GL=egl render ok: {img.shape}, mean {img.mean():.1f}")
PY
echo
echo "==> done. Next: make box-data, then make box-train"
