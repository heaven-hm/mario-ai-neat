# Mario AI Heaven

A learning bot for **Super Mario Bros. 1 on NES, running in FCEUX**. It reads the SMB1 RAM layout, observes nearby tiles and enemies, and evolves a neural-network controller across repeated play attempts. Other games and emulators are outside the target.

## Stack

- **Game:** NES Super Mario Bros. 1. The ROM is not included.
- **Runtime:** FCEUX 2.x Lua scripting.
- **Language:** Lua 5.1 compatible; no compiler, external packages, or separate ML runtime.
- **Learning:** NEAT-style neuroevolution, local to the Lua bot. It evolves neural-network topology and weights using episode fitness.
- **Learning database:** `mario_ai_heaven_neat.db`, generated beside the script. It stores the population, genome weights, mutation settings, and generation so training continues after restarting FCEUX.

## Run and train

1. Open a compatible SMB1 NES ROM in FCEUX, preferably at the start of World 1-1.
2. Load `mario_ai_heaven.lua` from FCEUX's Lua script menu.
3. Leave the script running. On the first active SMB1 frame it saves a fixed training start in FCEUX savestate slot 9. It then tests each genome from that same state, scores the attempt, breeds a new generation, and repeats.
4. Stop the script when you want. The population database is saved periodically, after every completed attempt, and when FCEUX stops the script. Leave the database beside the Lua script to continue learning later.
5. Read `mario_ai_heaven.log` beside the script for startup, episode, and database-save events.

The bot uses FCEUX's predefined slot 9 for fair training episodes. This overwrites that slot, so reserve it for Mario AI Heaven. It uses `savestate.object()` when available and the older `savestate.create()` compatibility API otherwise. It never calls `savestate.persist()`, the native FCEUX function that crashed on the Homebrew Apple Silicon build. If a FCEUX build has no compatible savestate API, the bot logs the condition and continues with less-controlled input-only episodes.

New databases contain 300 genomes. Existing databases retain their current population size so that previous learning is not discarded. Delete `mario_ai_heaven_neat.db` to begin a new 300-genome run.

## Champion play

After training, set `local PLAY_CHAMPION_ONLY = true` near the top of `mario_ai_heaven.lua`. The bot loads the genome with the highest saved fitness and repeatedly plays it from slot 9 without mutation, crossover, or generation changes. Set it back to `false` to resume training.

## Testing aids

The current testing build refreshes the SMB1 timer to `999` while gameplay is active and refreshes the lives byte at `0x075A` to `9`. This prevents a training run from reaching Game Over, while SMB1 still performs every normal death and respawn. The bot never presses Start automatically; begin a game manually in FCEUX. Before a real evaluation, change `TESTING_FREEZE_TIMER` and `TESTING_INFINITE_LIVES` near the top of `mario_ai_heaven.lua` to `false`.

## What it senses and learns

The bot uses the SMB1 positions and tile data from the legacy script and MarI/O's SMB1 sensor layout: a nearby tile/enemy grid plus Mario movement and power state. The neural network scores controller actions. A safety filter removes forward-only actions when an unpowered Mario is close to an enemy, while preserving learned choices such as jumping, braking, and retreating. The original bot's jump-over-ground-enemies and fire-as-Fire-Mario behaviors inform that filter.

Each attempt earns fitness for furthest forward progress and survival, with a large bonus for reaching the flag. Completed generations retain a champion, group related genomes into species, select fitter parents, cross over matching genes, and mutate connections and weights. This is evolutionary reinforcement learning: the learned population persists in the database and is evaluated during real SMB1 play sessions.

The FCEUX overlay shows the active generation, genome, species, current lesson, chosen action, observed threat or gap, progress, and database status. It does not show testing-aid settings.

The implementation is inspired by the MarI/O approach, but does not redistribute its code. The supplied MarI/O gist says its code may be used but should not be redistributed. This project implements its own NEAT-style trainer and adapts the sensor/runtime to FCEUX SMB1.

## Tests and limitations

```sh
sh tests/run.sh
```

Tests cover neural-network evaluation, enemy sensors, enemy safety filtering, population save/load, generation breeding, and the FCEUX control loop without RAM writes. They do not establish that a learned genome beats the game. No compatible ROM is present in the workspace, so real training and playthrough validation remain necessary. See [evaluation](docs/evaluation.md), [learning](docs/learning.md), [limitations](docs/limitations.md), and [RAM map](docs/ram-map.md).

The target RAM layout is the SMB1 revision described by the original bot and the linked [SMB disassembly](https://gist.github.com/1wErt3r/4048722). Other revisions and ROM hacks are not validated.

## Files

- `mario_ai_heaven.lua`: self-contained FCEUX SMB1 bot, sensors, NEAT trainer, episode loop, and database persistence.
- `legacy/LuaRio_Bot_v1.lua`: original bot preserved byte-for-byte.
- `docs/learning.md`: training loop, fitness, genome database, and restart behavior.
- `docs/requirements-and-weaknesses.md`: behavior inventory and remaining risks.
- `docs/ram-map.md`: inherited RAM addresses and verification status.

The legacy header credits Haseeb Mir, SethBling, and doppelganger. The original file did not declare a license, so this repository does not add one without rights confirmation.
