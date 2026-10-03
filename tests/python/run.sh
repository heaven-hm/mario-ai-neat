#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
PYTHONPATH=python python3 -m unittest discover -s tests/python -p 'test_*.py'
python3 -m py_compile python/mario_ai_fceux/*.py
LUA_BIN="${LUA_BIN:-lua5.1}"
command -v "$LUA_BIN" >/dev/null 2>&1 || LUA_BIN=luajit
"$LUA_BIN" -e "assert(loadfile('python/fceux_bridge/mario_ai_fceux_bridge.lua'))"
