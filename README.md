# Mario AI Heaven

Mario AI Heaven is an autonomous controller for **Super Mario Bros. on the NES**, intended to run as a Lua script inside FCEUX. The project reads game memory and sends ordinary controller buttons. It does not modify the game state to escape hazards or get unstuck.

## Technology stack

| Component | Use |
| --- | --- |
| Nintendo Entertainment System (NES) | Target game platform; the project controls SMB1, it does not emulate or compile the console |
| FCEUX 2.x | NES emulator and Lua host |
| Lua 5.1 | Script language expected by FCEUX's Lua API |
| LuaJIT 2.1 | Optional local Lua 5.1-compatible runtime for fast, ROM-free tests |
| GitHub Actions | Runs syntax checks and deterministic tests without a ROM |
| C/6502 assembler | Not used to build this project; no game ROM or emulator is included |

The code has no Lua package dependencies. The target is the original SMB1 NES ROM revision described in the project's RAM notes. Modified ROMs and The Lost Levels are not yet validated.

## Run in FCEUX

1. Obtain and open your own compatible SMB1 NES ROM in FCEUX.
2. Start the game at the title screen or enter a level.
3. In FCEUX, open the Lua script dialog and load `bot.lua` from this directory.
4. The on-screen status display shows the phase, target, current action, and recovery count.
5. Stop the Lua script from FCEUX's Lua window to return to manual control.

The script requires FCEUX's `memory.readbyte`, `joypad.set`, and `emu.frameadvance`. HUD drawing is optional. The bot advances exactly one frame after each observation and action.

## Local checks

From this directory, run:

```sh
luajit tests/run.lua
```

LuaJIT is optional for FCEUX itself. Lua 5.1 can run the ROM-free tests too. CI installs Lua 5.1 and never needs a ROM.

## Behavior

The bot evaluates survival, reachable routes, gaps, obstacles, enemy danger, power-up reward, and backward movement. When an action stalls, it records the failed state/action and tries a different bounded approach, such as changing takeoff timing, jump hold, speed, or route. Ties among safe actions can be broken with a seeded choice so runs remain reproducible. Recovery uses controller inputs only; it never raises Mario, teleports him, edits collision state, or clears enemy slots.

See [behavior inventory and weakness audit](docs/requirements-and-weaknesses.md), [architecture](docs/architecture.md), [RAM map status](docs/ram-map.md), [evaluation protocol](docs/evaluation.md), and [known limitations](docs/limitations.md).

## Attribution and licensing

The original [LuaRio_Bot.lua](legacy/LuaRio_Bot_v1.lua) was written by Haseeb Mir. Its header credits SethBling and doppelganger and identifies the SMB disassembly gist used for game data. The original file is preserved verbatim. The repository does not add a license until rights for the original and derived code are confirmed.
