#!/bin/sh
# Run the SMB1 PPO test suite plus static checks.
#
# The suite needs no Super Mario Bros. ROM: everything that touches the real
# emulator is exercised through the ROM-free stand-in in smb1_ppo.synthetic.
#
# It runs on either environment layout. To check the other one, point PYTHON at
# that virtual environment:
#
#   PYTHON=.venv313/bin/python tests/smb1_ppo/run.sh
set -eu
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

if [ -z "${PYTHON:-}" ]; then
    PYTHON=".venv/bin/python"
    if [ ! -x "$PYTHON" ]; then
        PYTHON="python3"
    fi
fi

"$PYTHON" -m unittest discover -s tests/smb1_ppo -p 'test_*.py' -v
"$PYTHON" -m py_compile smb1_ppo/*.py

RUFF="$(dirname "$PYTHON")/ruff"
if [ ! -x "$RUFF" ]; then
    RUFF="$(command -v ruff || true)"
fi
if [ -n "$RUFF" ]; then
    "$RUFF" check smb1_ppo tests/smb1_ppo
    "$RUFF" format --check smb1_ppo tests/smb1_ppo
else
    echo "ruff not found; skipping static checks"
fi
echo "smb1-ppo checks passed"
