# Local FCEUX acceleration trial

This branch changes NEAT reproduction and checkpointing. The Lua checks confirm those behaviors, but only SMB1 play can show whether it reaches a target in fewer generations.

## Prepare two isolated runs

1. Stop the Lua script in FCEUX and make a backup of the current `mario_ai_neat.db`.
2. Make two separate folders, `baseline` and `accelerated`. Copy the current project script into `baseline` and this branch's `mario_ai_neat.lua` into `accelerated`.
3. Copy the **same** database backup into both folders as `mario_ai_neat.db`. Keep the live Downloads database out of these runs.
4. Set `PLAY_CHAMPION_ONLY = false` in both scripts. Keep the timer setting, population size, ROM, FCEUX settings, and starting location identical. Both scripts in this workspace initialize the timer to 999 once per attempt.

## Run and record

1. Open SMB1 in FCEUX, enter World 1-1 manually, and put Mario at the same start location for each run.
2. Load one script in FCEUX and allow a fixed number of *new genome evaluations* (for example 1,000). Stop it and keep its `.db` and `.log` together.
3. Repeat with the other script from the same start. Do not swap databases between folders after training begins.
4. Repeat at least three times with fresh copies of the original backup. Starting both scripts in different seconds gives different random seeds; note the run dates and times.
5. From each log, record the best `max_x`, the generation when it first reaches the same target X, the number of `reason=victory` episodes, deaths, and real elapsed time.

The warm-start trial above measures improvement **from the saved population**. To test the claim of reaching in 40–50 generations what the old trainer reaches around generation 100, do a separate cold-start comparison with identical population sizes and the same level start. This takes many FCEUX hours. Compare results at equal evaluated genomes too: population size changes make generation counts misleading.

## Decision rule

Keep this branch only if it reaches a chosen SMB1 progress milestone earlier across repeated runs and does not reduce the completion rate. The target of 40–50 generations is provisional until measured. If the result is worse, test each change separately: singleton mutation, species weighting, parent tournament, sensor sampling, and early stopping.

The branch uses optional `P`, `H`, and `S` records in the V1 text database for resume progress and structural history. The current reader still accepts old V1 checkpoints. Preserve a backup before running an older script on an experimental checkpoint.
