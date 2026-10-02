#!/bin/sh
# Install the SMB1 PPO baseline into a local virtual environment.
#
#   scripts/smb1_ppo_setup.sh
#
# On Linux you can optionally build/install the smaller CPU-only PyTorch wheel:
#
#   SMB1_PPO_TORCH_INDEX=https://download.pytorch.org/whl/cpu scripts/smb1_ppo_setup.sh
#
# Apple Silicon needs no special handling: the default PyTorch wheel already
# carries MPS support.
set -eu
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
VENV="${VENV:-.venv}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "error: $PYTHON not found; install Python 3.10 or newer" >&2
    exit 1
fi

if [ ! -d "$VENV" ]; then
    echo "creating $VENV"
    "$PYTHON" -m venv "$VENV"
fi

"$VENV/bin/python" -m pip install --quiet --upgrade pip setuptools wheel

if [ -n "${SMB1_PPO_TORCH_INDEX:-}" ]; then
    echo "installing torch from $SMB1_PPO_TORCH_INDEX"
    "$VENV/bin/python" -m pip install --index-url "$SMB1_PPO_TORCH_INDEX" "torch>=2.2"
fi

echo "installing SMB1 PPO dependencies"
"$VENV/bin/python" -m pip install -r requirements-smb1-ppo.txt

"$VENV/bin/python" - <<'PY'
import cv2
import gymnasium
import numpy
import stable_baselines3
import torch

print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  mps={torch.backends.mps.is_available()}")
print(f"numpy {numpy.__version__}  gymnasium {gymnasium.__version__}  opencv {cv2.__version__}")
print(f"stable-baselines3 {stable_baselines3.__version__}")
try:
    import gym_super_mario_bros  # noqa: F401

    print("gym-super-mario-bros: importable")
except Exception as error:  # pragma: no cover - setup diagnostics
    print(f"gym-super-mario-bros: FAILED ({error})")
PY

cat <<'MESSAGE'

setup complete.

next:
  python -m smb1_ppo.rom import --source "/path/to/Super Mario Bros. (World).nes"
  python -m smb1_ppo.train --rom roms/super-mario-bros.nes --world 1 --stage 1 --workers 8
  tests/smb1_ppo/run.sh          # the full suite, no ROM needed
MESSAGE
