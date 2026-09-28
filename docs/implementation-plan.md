# LuarioBot 2 Implementation Plan

> **For agentic workers:** Execute this plan task by task. Check boxes only after the stated evidence exists. Keep the original `LuaRio_Bot.lua` unchanged until the new bot passes the acceptance gates.

**Goal:** Turn the current beta into a GitHub-ready, autonomous NES Super Mario Bros. 1 bot that completes levels through controller input. Lost Levels is out of scope for this implementation and needs a separate validation pass.

**Architecture:** A FCEUX Lua entry point takes one immutable RAM observation per emulated frame. A world model interprets the scrolling tile buffer, sprites, player motion, and game phase; a hierarchical planner chooses a short action; a controller sends buttons and records the outcome. The planner combines safe route selection, a calibrated jump model, short-horizon action simulation, and recovery when observed progress differs from prediction.

**Tech stack:** FCEUX Lua API, Lua 5.1 compatible source, local LuaJIT or Lua 5.1 for unit tests, GitHub Actions for ROM-free checks. No external Lua packages or bundled ROMs.

## Global constraints

- Use `LuaRio_Bot.lua` as the behavioral audit input and preserve it verbatim in the repository's `legacy/` folder.
- Use the user-provided [SMB disassembly](https://gist.github.com/1wErt3r/4048722) as the authority for game internals. If it is unavailable to the executing agent, flag each unverified address and do not silently substitute a different source.
- Read game RAM and press controller buttons during production play. Never write game RAM, suppress enemy slots, alter collision flags, freeze timers, or use savestates in production play.
- Savestates are allowed only in deterministic tests and calibration runs; mark them as test mode in code and logs.
- Target the original NES SMB1 PRG0 ROM. Do not claim Lost Levels compatibility in this implementation.
- Do not claim “plays every level” until reproducible clean-start runs demonstrate it. Publish measured completion rates and failure traces.
- Do not commit ROMs, savestates derived from commercial ROMs, emulator binaries, or gameplay footage without explicit rights review.

## Current-state audit

The root contains only `LuaRio_Bot.lua` (853 lines); it is not yet a Git repository. The script contains useful RAM address and enemy-ID knowledge, but several behaviors prevent reliable autonomous play:

| Existing behavior | Consequence | Replacement |
| --- | --- | --- |
| `delayFrameInterval()` advances frames according to `os.clock()` (lines 71–83) | Decisions depend on host speed; hazards can change during a blocking routine | Exactly one observation, decision, input, and `emu.frameadvance()` per loop |
| Nested tile-distance/enemy-slot loops in `RunGameEngineSubRoutines()` (roughly lines 549–727) | One sprite can trigger multiple contradictory inputs and frame advances | One snapshot, one scored action per frame |
| `setVelocity()` and writes to enemy/collision bytes (lines 66, 488, 656–657, 816) | Game state is modified rather than played | Controller input only in production |
| Only enemy proximity and collision trigger jumps | Gaps, landing zones, low ceilings, and moving platforms are missed | Tile map, collision geometry, motion prediction |
| Global booleans and state-name routines mix observation, display, and control | Stale state and unclear behavior | Small modules with explicit data contracts |
| Power-up chase has an unbounded `while` loop | Bot can hang or run into hazards | Bounded goal with re-evaluation every frame |
| Death loop waits on `0x0772` without a bounded recovery policy | May stop progress indefinitely | Explicit game-phase state machine and retry budget |

## Planned repository layout

```text
LuarioBot/
  README.md                    Setup, compatibility, measured results, controls
  .gitignore                   ROMs, savestates, traces, logs, OS artifacts
  .github/workflows/lua.yml    Lua syntax, unit, and ROM-free integration checks
  docs/ram-map.md              Verified addresses, meanings, source anchors
  docs/architecture.md         Data contracts and one-frame flow
  docs/evaluation.md           Reproducible emulator playthrough protocol
  docs/limitations.md          Known failures and evidence, updated per run
  legacy/LuaRio_Bot_v1.lua     Exact copy of original file
  bot.lua                      FCEUX entry point
  src/adapter/fceux.lua        API normalization and ROM/game identification
  src/observe/ram.lua          One-frame raw RAM snapshot
  src/observe/world.lua        Terrain, enemies, items, phase, derived geometry
  src/model/physics.lua        Observed motion and calibrated jump trajectories
  src/model/map.lua            Scrolling map cache with confidence/expiry
  src/plan/routes.lua          Reachable landing platforms and route goals
  src/plan/actions.lua         Short-horizon candidate actions and risk scoring
  src/plan/goals.lua           Goal state machine: progress, item, exit, retry
  src/control.lua              Jump hold/release, swim, run, fire, action lock
  src/telemetry.lua            Optional frame trace and failure summary
  tests/fixtures/*.lua         Synthetic RAM snapshots and scenario sequences
  tests/unit/*.lua             Pure module tests
  tests/integration/*.lua      Mock FCEUX and trace replay tests
  tools/replay.lua             ROM-free deterministic trace replay
```

### Shared contracts

The implementing model should define these contracts before adding behavior. Use named fields; avoid shared globals.

```lua
Observation = {
  frame=0, game="smb1", phase="title|loading|playing|death|victory",
  world=1, stage=1,
  player={x=0, y=0, vx=0, vy=0, size="small|big", power="none|fire",
          swimming=false, grounded=false, invulnerable=false},
  enemies={{slot=0, id=0, state=0, x=0, y=0, vx=0, vy=0}},
  items={{id=0, x=0, y=0}},
  tiles={ get=function(self, world_x, screen_y) return nil end }
}
Action = {left=false, right=false, up=false, down=false,
          A=false, B=false, start=false, reason="run"}
```

`tiles:get()` returns a tile ID or `nil` when the cell is unobserved; `nil` must never be treated as confirmed empty floor. `Observation` is recreated each frame. Planner and control state persist in owned module instances. Every module has a constructor where state is required, and pure functions otherwise.

## Task 1: Source audit and repository baseline

**Files:** Create `LuarioBot/legacy/LuaRio_Bot_v1.lua`, `.gitignore`, `README.md`, `docs/ram-map.md`, `docs/architecture.md`.

**Deliverable:** A reviewable inventory of the current script and a source-anchored RAM table before implementing behavior.

- [ ] Copy the original byte-for-byte. Verify with `cmp LuaRio_Bot.lua LuarioBot/legacy/LuaRio_Bot_v1.lua`.
- [ ] Inspect the linked disassembly. For every planned RAM address, record symbol, numeric address, value semantics, ROM revision, and evidence location in `docs/ram-map.md`. In particular verify `0x000E`, `0x0057`, `0x006D`, `0x0086`, `0x009F`, `0x000F..0x0013`, `0x0016..0x001A`, `0x001E..0x0022`, `0x0500..0x069F`, `0x0754`, `0x0756`, `0x075C`, `0x075F`, `0x0770`, and power-up bytes from the original.
- [ ] Mark addresses as **verified**, **observed in emulator**, or **unverified**. Do not use an unverified address for an irreversible branch such as death detection or route selection.
- [ ] Document the exact FCEUX Lua functions used by the original (`memory.readbyte`, `joypad.set`, `emu.frameadvance`, `gui.text`) and run a five-frame API smoke test against the installed FCEUX build before depending on optional functions.
- [ ] Write the data contracts above into `docs/architecture.md`; include the single-frame invariant and how title/death/victory phases are handled.
- [ ] Commit this baseline as one reviewable change. Do not create a `LICENSE` for copied legacy code until authorship and rights are confirmed.

**Gate:** Original matches byte-for-byte; RAM table has evidence for every address used in Task 2; no production RAM write is planned.

## Task 2: Observation and emulator adapter

**Files:** Create `bot.lua`, `src/adapter/fceux.lua`, `src/observe/ram.lua`, `src/observe/world.lua`, `tests/fixtures/level_1_1.lua`, `tests/unit/observe.lua`, `tests/integration/frame_loop.lua`.

**Interfaces:** `Fceux.new(api):read_byte(address)`, `:send(action)`, `:advance()`; `Ram.read(adapter, frame) -> raw`; `World.build(raw, map) -> Observation`.

- [ ] Build fixture sequences for title, level entry, grounded Mario, jump, active enemy, death, and flagpole. Store only synthetic byte maps in Lua tables.
- [ ] Write failing tests asserting world X across page boundaries, signed speeds, five enemy-slot pairing, player power/state, and `nil` for unavailable tiles. Example: page `1`, X byte `0x04` must give `260`, and speed byte `0xFC` must be interpreted as negative.
- [ ] Implement one byte-read pass per frame. No `memory.readbyte` inside planning loops; the observation owns a local tile snapshot for the visible two-page buffer.
- [ ] Implement `bot.lua` as a loop whose only frame advance occurs at the bottom. Count calls in a mock integration test: 100 iterations must produce 100 decisions, 100 input writes, and 100 frame advances.
- [ ] Verify start, death, and victory transitions on synthetic traces. In inactive phases, release all movement buttons; press Start only according to an explicit title/retry policy.
- [ ] Run unit and integration tests, then commit.

**Gate:** No production use of `memory.writebyte` or savestates; all observation tests pass; FCEUX smoke test runs without Lua errors.

## Task 3: Terrain memory and geometry

**Files:** Create `src/model/map.lua`, expand `src/observe/world.lua`, `tests/unit/map.lua`, `tests/fixtures/gap.lua`, `tests/fixtures/pipe.lua`.

**Interfaces:** `Map.new():update(raw_tiles, camera_x, frame)`, `:tile(x,y) -> id|nil`, `:surface(x) -> y|nil`, `:platforms(window) -> {Platform}`. A `Platform` has `{left,right,top,confidence}`.

- [ ] Derive the two-page buffer indexing and tile classes from the linked disassembly. Test a 255→256 world-X crossing and a 511→512 ring-buffer reuse.
- [ ] Cache only cells actually observed in the active camera window. Invalidate a cached page on reuse and clear the cache on area transition, death/reload, or a backward warp.
- [ ] Distinguish empty, solid, pass-through, damaging, collectible, and unknown cells. Where the disassembly cannot establish collision behavior, use an explicit unknown class and verify in FCEUX.
- [ ] From solid cells, compute the next gap, obstacle height, headroom, platform edges, and reachable landing surfaces. Tests must include a 2-tile gap, 4-tile gap, pipe, low ceiling, elevated platform, and unloaded map edge.
- [ ] Run tests and commit.

**Gate:** Terrain model never infers safe floor from an unseen tile; page reuse does not expose stale geometry.

## Task 4: Motion model and jump calibration

**Files:** Create `src/model/physics.lua`, `tests/unit/physics.lua`, `tests/fixtures/jumps.lua`, extend `docs/evaluation.md`.

**Interfaces:** `Physics.new(profile):observe(previous, current, action)`, `:predict(player, action_sequence) -> states`, `:reachable(from, platform, options) -> bool, margin`.

- [ ] Capture frame traces for standing jump, walking jump, running jump, short tap, long hold, fall, skid, and swimming. Record X/Y, RAM speed, button input, and contact state every frame. Keep capture tooling in test mode only.
- [ ] Fit conservative empirical horizontal acceleration, deceleration, jump impulse, variable-height hold effect, gravity, and landing tolerance. Store the SMB1 profile with version and trace provenance.
- [ ] Write tests comparing predicted trajectories with held-out traces. Set acceptance thresholds before tuning: median position error ≤4 pixels at 20 frames and no false “reachable” result for a gap that the trace falls into.
- [ ] Implement interval bounds for uncertain motion; choose actions using the pessimistic bound near pits and hazards.
- [ ] Run tests and commit.

**Gate:** Profiles have measured evidence; unsafe overestimates of jump range fail the task even if average error is low.

## Task 5: Route planning and action selection

**Files:** Create `src/plan/routes.lua`, `src/plan/actions.lua`, `src/plan/goals.lua`, `src/control.lua`, `tests/unit/routes.lua`, `tests/unit/actions.lua`, `tests/fixtures/enemy_approach.lua`.

**Interfaces:** `Routes.plan(observation, map, physics) -> {goal, landings}`; `Actions.choose(observation, route, physics, memory) -> Action`; `Control.new():apply(action, observation) -> Action`.

- [ ] Build a graph of visible platform surfaces. Add a directed edge only when the motion model finds a feasible jump with a landing margin. Favor progress to the right while allowing short backward movement for a reachable item or recovery.
- [ ] Evaluate candidate button sequences over a 20–35 frame horizon: run, brake, jump with several hold durations, release, swim pulse, and fire. Replan every frame; execute only the first frame of the chosen sequence.
- [ ] Score survival before speed: confirmed pit fall or collision is disqualifying; then score landing margin, enemy separation, forward progress, and optional power-up value. Unknown map geometry adds risk rather than a safe assumption.
- [ ] Predict moving enemy positions from observed X/Y differences. Treat a stomp as safe only if Mario approaches from above with downward velocity and sufficient horizontal overlap; otherwise avoid or fire when feasible. Distinguish shells, Piranha Plants, projectiles, lifts, flagpole objects, and harmless sprites from the disassembly's IDs/states.
- [ ] Make controller timing explicit: a fresh `A` press starts a jump; hold length controls height; release before the next jump; `B` enables running and triggers fire only when appropriate. Respect player lock, damage animation, auto-walk, and swimming phases.
- [ ] Test gap takeoff, landing, low ceiling, enemy from left/right, two enemies together, pipe with insufficient speed, and power-up behind a hazard. Assert exact first action and reason for each fixture.
- [ ] Run tests and commit.

**Gate:** The planner never selects an action predicted to die when a tested survivable candidate exists. One input table is sent per frame.

## Task 6: Progress recovery and game flow

**Files:** Expand `src/plan/goals.lua`, `src/control.lua`, `src/telemetry.lua`; create `tests/unit/recovery.lua`, `tests/integration/replay.lua`, `tools/replay.lua`.

**Interfaces:** `Goals.new():update(observation, outcome) -> goal`; `Telemetry.new(config):record(observation, action, prediction)`; `Replay.run(trace, policy) -> actions, summary`.

- [ ] Detect lack of forward progress over a bounded window, collision with a wall, unexpected fall, repeated death at the same X range, area transition, and victory. Reset the plan when observations contradict it.
- [ ] Add bounded recovery strategies: brake and retry with a different takeoff point, choose an alternative platform, retreat only when safe, and stop/back off when no safe plan is visible. Avoid endless loops by recording repeated state/action signatures.
- [ ] On death, release controls, wait for the game's transition, and resume according to configured retry policy. Do not load a savestate in production. On victory/flagpole, allow the game's auto-walk and transition to the next stage.
- [ ] Record optional compact traces: ROM/profile label, frame, world/stage, player state, local tiles, enemies, chosen action, reason, predicted risk, and observed outcome. Never log ROM bytes.
- [ ] Replay a trace deterministically through the planner and assert the same action stream. Add a regression fixture for each emulator failure fixed in later tasks.
- [ ] Run tests and commit.

**Gate:** A no-progress sequence terminates or changes strategy within a configured bound; death and level transitions cannot trap the main loop.

## Task 7: Real emulator evaluation and tuning

**Files:** Create `docs/evaluation.md`, `docs/limitations.md`; add regression fixtures and measured profiles. Do not alter the original file.

- [ ] Define a clean-start protocol: record emulator version, ROM identity/hash locally without publishing ROM data, character, game speed, bot revision, retry limit, and whether any test savestate was used. A production validation run begins from the title screen without savestate loads or RAM writes.
- [ ] Establish a baseline on SMB1 1-1: run at least 10 clean-start attempts. Record completion count, deaths, farthest X, cause of each failure, and median elapsed game frames.
- [ ] For each failed attempt, preserve a compact trace, classify the cause (perception, physics, route, timing, game phase, or unsupported feature), add one focused regression fixture, fix the responsible module, and rerun that fixture plus the clean-start scenario.
- [ ] Expand gates in order: all four stages of World 1; then Worlds 2–4; then Worlds 5–8. Test underwater, moving lifts, springs, castle hazards, maze routes, Bowser bridge/axe, and transitions explicitly. A world passes only after 10 clean-start attempts from its first stage without manual input; report actual pass rate, not only a best run.
- [ ] Do not test or advertise Lost Levels compatibility in this implementation. Track it as a separate future plan.
- [ ] Update `docs/limitations.md` after every gate. Keep any partially working feature labeled accurately.

**Gate:** Publish evidence for each claimed stage/world. Full completion claim requires at least one clean-start completion of all SMB1 stages and 10 repeat runs with the completion rate stated. Lost Levels receives its own claim and evidence.

## Task 8: GitHub packaging and final review

**Files:** Finalize `README.md`, `.gitignore`, `.github/workflows/lua.yml`, `docs/architecture.md`, `docs/evaluation.md`, `docs/limitations.md`.

- [ ] Document exact installation, FCEUX Lua launch path, supported ROM revision, supported game/character matrix, controls to stop the bot, test commands, and measured evaluation table.
- [ ] Configure CI to syntax-check every Lua file and run ROM-free unit, integration, and replay tests on push/PR. Fail CI on `memory.writebyte` usage outside explicitly marked test utilities.
- [ ] Review attribution to Haseeb Mir, SethBling, and doppelganger; preserve the original header. Confirm rights before choosing a repository license for the new and legacy code.
- [ ] Check repository contents for ROMs, savestates, large trace data, secrets, and generated artifacts. Verify the legacy copy still matches the root original.
- [ ] Run the complete test suite and one final clean-start emulator evaluation, record the results in README, then commit the documentation and packaging.

**Gate:** A new contributor can clone the folder, run ROM-free tests, supply their own ROM, launch the bot, and reproduce the stated results. No unsupported completion claim appears in the README.

## Decision log for the executor

- **Planning method:** Use a deterministic hierarchical planner with measured physics and short-horizon simulation. It provides inspectable failure traces and runs inside Lua without a training service. A learned policy may be added later only if benchmark data shows a concrete gap that the planner cannot address.
- **Success target:** Autonomous controller input and reproducible completion, not altered RAM or a scripted prerecorded input movie.
- **Scope:** Original NES SMB1 PRG0 only. A future Lost Levels plan may reuse interfaces after independently validating its RAM map and physics.
- **Evidence rule:** Unit tests prove local logic; only clean-start FCEUX runs prove real level completion.

## Handoff status

This document is the plan only. No V2 implementation or GitHub repository has been created. The original `LuaRio_Bot.lua` remains the sole program in the workspace root. The linked gist did not load in this planning environment, so Task 1 explicitly requires the executing model to inspect it and verify addresses before implementation.
