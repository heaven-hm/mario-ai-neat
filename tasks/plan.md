# Accelerated SMB1 NEAT experiment

## Goal

Reduce the generations needed to reach a fixed SMB1 progress milestone. The target is comparable progress by generation 40–50 where the current trainer needs about 100. This is a hypothesis, not a promised result. This branch stays separate from `main`, `develop`, the running Downloads script, and its database.

## Baseline and measurement

- Use the same SMB1 ROM, FCEUX version, starting savestate, and population size for baseline and experiment.
- Compare at equal numbers of evaluated genomes as well as equal generations; a generation count alone hides population changes.
- Record best world X, median world X, flagpole rate, deaths, evaluations, and wall time every generation.
- Run at least three independent seeds per variant. The experiment succeeds if median milestone generation falls by roughly half without lowering the held-out completion rate.
- Source checkpoint (`mario_ai_neat.db`) is a snapshot of learned networks, not a fair baseline unless both variants start from the same copy. Never train directly against the committed or Downloads database.

## Architecture decisions

1. Keep the existing FCEUX Lua runtime and V1 database format. Add an optional progress record that older readers ignore. No external model, Python bridge, or emulator modification.
2. Fix correctness issues before tuning hyperparameters. A single-member species currently creates identical, unmutated children, and independent splits can reuse a hidden-node number.
3. Use better parent selection inside a species: favor the strongest surviving genomes while retaining diversity through species and a protected champion.
4. Keep each training episode's fixed start, because equal starts make fitness comparable. A separate curriculum can be evaluated later using explicit checkpoints.
5. Preserve old behavior behind small constants where useful so the same code can run an ablation comparison.

## Work items

### Task 1 — Reproduction correctness

Mutate children of one-member species and sample parents from the fitter half of each surviving species. Preserve the champion exactly once and keep population size stable.

**Acceptance:** a one-member species produces descendants with changed genes over multiple seeded trials; champion and population size survive; V1 checkpoints still load.

### Task 2 — Useful early exploration

Reduce wasted attempts from no-progress genomes using a short, configurable initial progress window, while keeping enough time for a deliberate jump. Bias new connections toward compact movement, enemy, item, and gap inputs, while still exploring the full grid.

**Acceptance:** a stationary genome ends earlier; a genome making progress is unaffected; new links sample global inputs more often without excluding grid inputs; death and flagpole scoring are unchanged.

### Task 3 — Population-wide structural history

Assign one hidden-node number per distinct split innovation across the population. Evaluate nodes by their enabled dependencies because a newer split node can feed an older hidden node. Save split history in optional V1 records and infer the next safe number from older databases.

**Acceptance:** splitting the same connection in two genomes reuses the same hidden node; splitting different connections gives distinct hidden nodes; a newer hidden node can drive an older hidden target; old checkpoints load without node collisions on future mutations.

### Task 4 — Fair experiment

Persist the next genome index so a restart does not replay already scored genomes. Add deterministic offline reproduction/episode checks and a concise local FCEUX comparison procedure. Record the results of actual play separately from structural tests.

**Acceptance:** existing tests and new focused checks pass; V1 checkpoints load, restart resumes the next genome, and no unmeasured speedup is claimed.

## Checkpoints

- After Task 1: run focused mutation and persistence checks.
- After Task 2: run the Lua suite and inspect the diff for checkpoint compatibility.
- After Task 3: run a local FCEUX trial if an SMB1 ROM and suitable start state are available; otherwise hand off exact steps for the user's local run.

## Risks

| Risk | Response |
| --- | --- |
| Faster episodes discard a genome that is setting up a jump | Use a generous early window and compare survival/progress logs. |
| Stronger parent selection reduces diversity | Retain species, crossover, mutation, and the global champion. |
| Existing checkpoint hides improvements | Compare fresh starts and identical checkpoint copies separately. |
| Offline checks look good but gameplay does not | Treat FCEUX play results as the deciding evidence. |

The reproduction changes are motivated by the original [NEAT paper](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf), which describes incremental mutation and protection of new structures through speciation. The exact SMB1 settings here are experimental.
