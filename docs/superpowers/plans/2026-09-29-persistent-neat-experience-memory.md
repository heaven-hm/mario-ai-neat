# Persistent NEAT Experience Memory Implementation Plan

> **For agentic workers:** Execute this plan inline in the existing `experiment/accelerated-neat` branch and workspace.

**Goal:** Preserve useful state-to-action experience in the checkpoint database and reuse it to guide later NEAT decisions in similar SMB1 situations.

**Architecture:** Keep NEAT genomes, selection, and the safety shield. Add a bounded contextual action memory keyed by local obstacle/enemy context, movement speed, grounded state, and power state. Label a context/action pair only when the tracked encounter clears or fails; apply its evidence as a small action-score bias. Store memory in optional `X` rows so older checkpoints remain loadable.

**Tech Stack:** Lua 5.1/FCEUX, existing text checkpoint format, existing Lua test harness.

## Global Constraints

- Do not write RAM except existing timer/lives test aids.
- Do not let memory bypass `calculateAllowedActions` safety filtering.
- Preserve existing checkpoint records and load compatibility; old databases start with empty experience memory.
- Do not add or commit the live `mario_ai_neat.db`, backup, or log.
- Push only `experiment/accelerated-neat`; leave `main` and `develop` untouched.

---

### Task 1: Context classification and memory scoring

**Files:** `mario_ai_neat.lua`, `tests/run.lua`

- [x] Add stable, level-position-independent context keys for clear ground, enemy approach, gaps, and solid overhead/forward obstacles.
- [x] Add bounded per-context/per-action success and failure counts, running reward, and a conservative confidence-gated action-score bias.
- [x] Test similar state keys, distinct hazard classes, confidence gating, and action scoring without bypassing the allowed-action mask.

### Task 2: Experience outcome capture and persistence

**Files:** `mario_ai_neat.lua`, `tests/run.lua`

- [x] Track unique context/action choices during each hazard encounter and label them on grounded clearance, death, or stuck termination.
- [x] Add optional `X,key,action,attempts,successes,failures,rewardMean` checkpoint rows and load them safely.
- [x] Test success/failure updates, bounded memory, save/load round trips, and loading a legacy checkpoint with no `X` rows.

### Task 3: Training integration, documentation, and verification

**Files:** `mario_ai_neat.lua`, `README.md`, `tests/run.lua`

- [x] Use memory-biased scores in normal training while keeping Champion Mode and the safety shield behavior intact.
- [x] Show a compact experience-memory count in existing status/log output without creating another GUI panel.
- [x] Document what memory stores, how old checkpoints migrate, and why it can improve reuse but cannot guarantee a fixed generation count.
- [x] Run `LUA_BIN=luajit sh tests/run.sh`, inspect the diff, commit selected project files, and push the current experimental branch. The live database checkpoint is included at its current snapshot because the project tracks it.
