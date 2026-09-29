#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
LUA_BIN="${LUA_BIN:-lua5.1}"
command -v "$LUA_BIN" >/dev/null 2>&1 || LUA_BIN=luajit
"$LUA_BIN" tests/run.lua
"$LUA_BIN" -e "assert(loadfile('mario_ai_neat.lua'))"

# Keep emulator loop tests away from the live training database in this folder.
test_directory=$(mktemp -d "${TMPDIR:-/tmp}/mario-ai-tests.XXXXXX")
trap 'rm -rf "$test_directory"' EXIT
cp mario_ai_neat.lua tests/integration.lua tests/recovery.lua "$test_directory/"
cd "$test_directory"
"$LUA_BIN" integration.lua
"$LUA_BIN" recovery.lua
