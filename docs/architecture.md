# Architecture

## Frame loop

Each iteration follows one order:

```text
RAM snapshot -> world model -> goal/route -> action candidates -> risk choice
             -> controller normalization -> HUD/trace -> one frame advance
```

No planner routine advances the emulator. No production code writes NES RAM. The only persistent state lives in planner, goal, control, map, and telemetry objects. The observation is a fresh snapshot.

## Modules

- `src/adapter/fceux.lua`: wraps FCEUX APIs and normalizes controller tables.
- `src/observe/ram.lua`: snapshots bytes, player, enemy slots, phase inputs, and live tile buffer.
- `src/model/map.lua`: turns observed tile cells into terrain queries; unknown remains unknown.
- `src/model/physics.lua`: learns motion from action/outcome pairs and predicts conservative short trajectories.
- `src/plan/routes.lua`: builds reachable platform destinations from local geometry.
- `src/plan/actions.lua`: compares run, brake, jump holds, evade, fire, retreat, and objective pursuit.
- `src/plan/goals.lua`: prioritizes survival, route progress, safe optional powerups, and recovery.
- `src/control.lua`: translates the chosen intent into exact per-frame button states and jump press/release timing.
- `src/telemetry.lua`: emits optional local traces for deterministic replay and failure analysis.
- `bot.lua`: FCEUX entry point only.

## Decision and recovery

Use lexicographic priorities: immediate survival, safe landing, no repeated failed state/action, current escape objective, level progress, power-up value, then optional score. The planner uses model-predictive evaluation: generate short sequences, estimate player/enemy motion and landing, reject unsafe outcomes, score remaining candidates, execute one frame, observe, and replan.

Progress is measured against the active goal rather than just world X. When progress stalls, classify the cause, record a quantized state/action signature, and advance through a bounded alternative set: shift takeoff timing, change jump hold, adjust run/brake, select a different route, or retreat to a verified safe platform. Repeated options are penalized. A seeded tie-break may vary equally safe candidates, but randomness may not bypass safety checks. If no candidate is safe, release controls and return `no_safe_action`; never modify RAM to simulate elevation or teleportation.

## Data contract

`World.build(raw, map)` returns `Observation` with `phase`, `world_x`, player position/velocity/form, active enemy list, and a tile query returning an ID or `nil`. `Actions.choose(observation, route, physics, memory)` returns one `Action` table containing button booleans, intent, target, estimated risk, and reason. An unknown tile or object is uncertain, never automatically safe.

## Test modes

Unit and trace tests use synthetic memory. Any savestate-based calibration tool must require an explicit test-mode flag and cannot be imported into the production bot. The production adapter exposes no write-memory method.
