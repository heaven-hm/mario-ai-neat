#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
LUA_BIN="${LUA_BIN:-lua5.1}"
command -v "$LUA_BIN" >/dev/null 2>&1 || LUA_BIN=luajit
"$LUA_BIN" tests/run.lua
"$LUA_BIN" tests/integration.lua
"$LUA_BIN" -e "assert(loadfile('mario_ai_neat.lua'))"
