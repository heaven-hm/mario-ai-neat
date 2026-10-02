#!/bin/sh
# Run the SMB1 PPO test suite plus static checks.
#
# The suite needs no Super Mario Bros. ROM: everything that touches the real
# emulator is exercised through the ROM-free stand-in in smb1_ppo.synthetic.
set -eu
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python3}"
if [ -x .venv/bin/python ]; then
    PYTHON=".venv/bin/python"
fi
"$PYTHON" -m unittest discover -s tests/smb1_ppo -p 'test_*.py' -v
"$PYTHON" -m py_compile smb1_ppo/*.py
if [ -x .venv/bin/ruff ]; then
    .venv/bin/ruff check smb1_ppo tests/smb1_ppo
    .venv/bin/ruff format --check smb1_ppo tests/smb1_ppo
fi
echo "smb1-ppo checks passed"
