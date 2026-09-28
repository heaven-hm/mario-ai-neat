# Mario AI Heaven

An autonomous Lua bot for **Super Mario Bros. on the NES**. The bot observes Mario, level tiles, enemies, and powerups, then chooses an action each frame. Its main file is self-contained and ready to load in FCEUX.

## Stack

- **Game:** Super Mario Bros. for NES. The ROM is not included.
- **Runtime:** FCEUX with its Lua scripting interface.
- **Language:** Lua 5.1 syntax. LuaJIT 2.1 is used for local tests; FCEUX runs the bot.
- **Build:** No compiler, C library, Lua packages, ML runtime, RL, or RLHF is required for this version. RLHF is deferred.
- **Repository checks:** GitHub Actions runs synthetic behavior tests without a ROM.

The project is the game-playing bot. FCEUX supplies each frame's memory and controller interface; the bot's action selection and recovery are implemented in [`mario_ai_heaven.lua`](mario_ai_heaven.lua).

## Run it

1. Open your own compatible NES Super Mario Bros. ROM in FCEUX.
2. Load `mario_ai_heaven.lua` from FCEUX's Lua script menu.
3. Let the bot start from the title screen. Stop the script in FCEUX to return to manual control.

The target RAM layout is the SMB1 revision described by the original bot and the linked [SMB disassembly](https://gist.github.com/1wErt3r/4048722). Modified ROMs and The Lost Levels are not validated.

## How the bot chooses

Every frame it compares running, short and long jumps, braking, safe retreat, fire attacks, and reachable powerup pursuit. It estimates short-term landing and collision risk, prefers survival over optional rewards, and replans after the next observation. Powerups can justify a bounded backward detour. Mario only retreats to attack when powered, the threat is close, and the ground behind him is present.

When Mario makes no progress for 42 frames, the bot records the action that failed in that local situation. On the next decision it penalizes that same action and evaluates alternatives. This is structured exploration, not random button mashing. It never raises or teleports Mario, writes game RAM, deletes enemy slots, or overrides collision state.

## Tests

```sh
sh tests/run.sh
```

The suite checks gap handling, unsafe landings, stuck recovery, powerup backtracking, powered attacks, one input per frame, and that the production loop never writes RAM. These synthetic checks do not prove full-game completion. See [evaluation](docs/evaluation.md) and [limitations](docs/limitations.md).

## Project files

- `mario_ai_heaven.lua`: self-contained FCEUX bot and decision logic.
- `legacy/LuaRio_Bot_v1.lua`: original script preserved byte-for-byte.
- `docs/requirements-and-weaknesses.md`: behavior inventory and recovery requirements.
- `docs/ram-map.md`: inherited RAM addresses and verification status.

The legacy header credits Haseeb Mir, SethBling, and doppelganger. The original file did not declare a license, so this repository does not add one without rights confirmation.
