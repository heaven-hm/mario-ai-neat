#!/bin/sh
# Install the SMB1 PPO baseline into a local virtual environment.
#
#   scripts/smb1_ppo_setup.sh
#
# The installer picks the dependency set that matches your Python version, because
# gym-super-mario-bros has two incompatible layouts:
#
#   Python 3.13+  gym-super-mario-bros 9.x + nes-py 9.x. Prebuilt wheels on Linux,
#                 macOS, and Windows; Gymnasium-native; supports NumPy 2.
#   Python 3.12   gym-super-mario-bros 7.4.0 + nes-py 8.2.1. nes-py builds from
#                 source, so it needs a C++ toolchain (MSVC Build Tools 2022 on
#                 Windows).
#
# Override the interpreter with PYTHON=/path/to/python and the venv location with
# VENV=/path/to/venv. On Linux you can also pin the smaller CPU-only PyTorch wheel:
#
#   SMB1_PPO_TORCH_INDEX=https://download.pytorch.org/whl/cpu scripts/smb1_ppo_setup.sh
set -eu
cd "$(dirname "$0")/.."

VENV="${VENV:-.venv}"

if [ -z "${PYTHON:-}" ]; then
    for candidate in python3.14 python3.13 python3.12; do
        if command -v "$candidate" >/dev/null 2>&1; then
            PYTHON="$candidate"
            break
        fi
    done
fi
if [ -z "${PYTHON:-}" ] || ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "error: no Python 3.12+ interpreter found; set PYTHON=/path/to/python" >&2
    exit 1
fi

VERSION="$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
case "$VERSION" in
    3.13|3.14) REQUIREMENTS=requirements-smb1-ppo.txt ;;
    3.12) REQUIREMENTS=requirements-smb1-ppo-py312.txt ;;
    *)
        echo "error: Python $VERSION is not supported; install Python 3.12 or 3.13+" >&2
        exit 1
        ;;
esac

echo "using $PYTHON (Python $VERSION) with $REQUIREMENTS"
if [ ! -d "$VENV" ]; then
    echo "creating $VENV"
    "$PYTHON" -m venv "$VENV"
fi

# venv created by other tools (uv, for instance) may have no pip at all.
if ! "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
    echo "pip is missing from $VENV; bootstrapping it"
    "$VENV/bin/python" -m ensurepip --upgrade
fi

"$VENV/bin/python" -m pip install --quiet --upgrade pip setuptools wheel

if [ -n "${SMB1_PPO_TORCH_INDEX:-}" ]; then
    echo "installing torch from $SMB1_PPO_TORCH_INDEX"
    "$VENV/bin/python" -m pip install --index-url "$SMB1_PPO_TORCH_INDEX" "torch>=2.2"
fi

echo "installing $REQUIREMENTS"
"$VENV/bin/python" -m pip install -r "$REQUIREMENTS"

"$VENV/bin/python" - <<'PY'
import cv2
import gymnasium
import numpy
import stable_baselines3
import torch

import gym_super_mario_bros
import nes_py

print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  mps={torch.backends.mps.is_available()}")
print(f"numpy {numpy.__version__}  gymnasium {gymnasium.__version__}  opencv {cv2.__version__}")
print(f"stable-baselines3 {stable_baselines3.__version__}")
print(f"gym-super-mario-bros {getattr(gym_super_mario_bros, '__version__', 'unknown')}")
print(f"nes-py {getattr(nes_py, '__version__', 'unknown')}")

import sys
sys.path.insert(0, ".")
from smb1_ppo.rom import rom_seam_name

seam = rom_seam_name()
print(f"ROM lookup seam: {seam} ({'Gymnasium-native' if seam == 'smb1_rom_path' else 'legacy API'})")
PY

cat <<'MESSAGE'

setup complete.

next:
  python -m smb1_ppo.rom import --source "/path/to/Super Mario Bros. (World).nes"
  python -m smb1_ppo.train --rom roms/super-mario-bros.nes --world 1 --stage 1 --workers 8
  tests/smb1_ppo/run.sh          # the full suite, no ROM needed
MESSAGE
