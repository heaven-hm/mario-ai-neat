# Behavior Inventory and Weakness Audit

This document is the acceptance checklist for the NES SMB1 bot. It is based on the supplied `LuaRio_Bot.lua`; RAM semantics beyond that file must be checked against the user's SMB disassembly gist before they are enabled.

## What the original bot currently does

| Area | Existing behavior | Weakness |
| --- | --- | --- |
| Main loop | Prints a HUD, checks death/collision/flagpole, calls a routine, advances frames | Decisions are spread across nested routines that also advance frames; no consistent one-decision-per-frame contract |
| Default movement | Repeatedly sends right+B | It follows a single forward pattern and cannot choose when stopping or retreating is safer |
| Jumping | Jumps after enemy proximity, selected object proximity, or a crude collision test | No grounded confirmation, gap scan, landing selection, headroom check, variable jump timing, or jump outcome learning |
| Enemies | Reads five slots and compares positions and a small set of types/states | No predicted enemy motion, projectile trajectory, collision geometry, threat ranking, shell state strategy, or joint handling of multiple threats |
| Powerups | Mushroom collection can temporarily stop forward movement and chase a direction | It does not compare reward with danger; checks only the first two enemy slots during pursuit; has unbounded loops and no recovery budget |
| Fire Mario | Attempts to fire at enemies using B and delay calls | Does not estimate shot timing/range or confirm whether a shot defeated the threat; B is also run input |
| Terrain | Detects one generic object collision bit | No map model for pits, platforms, pipes, ceiling clearance, water, castle routes, or landing zones |
| Stuck handling | No systematic progress detector; collision branch writes velocity/collision RAM | No action alternatives, failure memory, or bounded recovery; RAM writes make this a cheat and can mask defects |
| Death/transition | Clears input while waiting for an operation task to change | No bounded timeout, explicit phase confidence, life-aware retry policy, or repeated-death strategy |
| Game progress | Stops input at one flagpole byte condition | No explicit level-exit/transition confirmation or campaign-level goal state |
| Timing | Uses `os.clock()` to decide frame delays | Host speed affects behavior; waits advance many game frames without replanning |
| State/data | Shared globals, mutable flags, nested `if` blocks | Stale values and side effects can cause contradictory decisions; hard to test individual behavior |
| Inputs | Several reused button tables with incomplete button fields | Button release/hold semantics are not centralized or tested |
| Robustness | Assumes an SMB-like memory layout and known object IDs | No ROM/revision guard or unknown-value fallback; altered ROMs can be misread silently |
| Evidence | Header says beta and credits sources | No unit tests, deterministic trace replay, measured success rate, or declared compatibility matrix |

## Required decisions the new bot must make

1. **Stay alive:** Can a candidate action intersect a pit, hazard, enemy, projectile, or unsafe landing? Survival dominates all optional rewards.
2. **Move toward a useful destination:** Which visible platform or level route can be reached with margin? Forward progress is the default goal, not a hard-coded action.
3. **Jump deliberately:** Is Mario grounded? When is takeoff required? How long should jump be held? Is the arc clear? Where will Mario land? Did the actual arc match prediction?
4. **React to danger:** Predict near-future relative motion for every visible enemy and projectile. Choose stomp, evade, fire, brake, or retreat based on Mario's form, geometry, and outcome uncertainty.
5. **Evaluate powerups:** Estimate whether a mushroom/flower is reachable, whether pursuing it increases death risk, the value of the upgrade, and a return route to the primary goal. Abandon the chase if its safety or time budget is exceeded.
6. **Backtrack intentionally:** Retreat only for a specific objective (safe landing, powerup, enemy setup, or recovery), with a destination and a timeout. Do not reverse just because progress is slow.
7. **Recover from a miss or stall:** Compare recent observations with the predicted result. Mark the action/state region that failed. Change one meaningful parameter or choose a different route, then measure the result.
8. **Handle game phases:** Distinguish title, active level, damage/transform, death, auto-walk, flagpole, area transition, and victory. In uncertain phases release movement and gather evidence instead of issuing a risky pattern.
9. **Know when it does not know:** Unobserved tiles, unknown enemy IDs, unsupported level mechanics, and uncertain RAM values increase risk. The bot may pause or try a safe information-gathering action; it must not call unknown space safe.
10. **Explain choices:** Every frame's selected action has a concise reason, target, risk estimate, and recovery attempt number in the optional HUD/trace.

## Professional recovery design

Recovery is a bounded, evidence-driven search over controller actions, not random key mashing and never RAM manipulation.

### Progress and failure signals

- Track world X, vertical position, grounded/airborne phase, health/form, target platform, action, and predicted landing.
- Declare a stall only when world X and goal distance fail to improve over a configurable interval and the player is not in a known transition or intentional wait.
- Classify the result: blocked by solid geometry, missed jump, low ceiling, enemy collision, fell, repeatedly changed direction, or observation uncertain.
- Store a short rolling history of quantized state plus action. A repeated state/action pair with no improvement is a loop and receives a penalty.

### Alternative generation order

For each failed attempt, generate structured alternatives in order, only if the map and risk checks permit them:

1. Keep the same route and change takeoff point by a few frames/pixels.
2. Keep takeoff point and change jump hold duration.
3. Drop run acceleration or add a short braking interval before jumping.
4. Choose another reachable landing platform/route edge.
5. Retreat to a confirmed safe platform, re-observe the enemy or item, then approach with a different tactic.
6. If powered, compare firing and jumping; if small, compare evasion and stomp only when the trajectory supports it.
7. If no tested candidate has a safe predicted outcome, release buttons briefly, collect another observation, and report `no_safe_action` rather than looping forever.

Each attempt has a frame limit and an improvement threshold. The planner records the attempted parameter tuple so it does not repeat it at the same local state. A fixed seed may break ties between equally scored safe alternatives; randomness must never bypass risk checks. The same trace and seed must replay identically.

## Goal and risk priority

Use lexicographic ordering, not one opaque weighted sum:

1. Avoid immediate death and known collision.
2. Prefer a safe landing over remaining airborne or falling.
3. Avoid repeating a failed state/action loop.
4. Complete a current survival objective (escape a pit edge, dodge a threat, exit damage/transition lock).
5. Progress toward the level exit.
6. Pursue a powerup only when its estimated survival-adjusted value exceeds the current route value and a safe return path exists.
7. Seek coins/score only as a low-priority tie-breaker.

Enemy elimination is not inherently a goal. The planner should kill an enemy only when the action reduces risk or opens a route; never backtrack into a hazard for score alone.

## Completion criteria

- Production action path uses button input only; no writes to NES RAM, no teleport/elevation behavior, no enemy-slot clearing, no collision-bit overrides, and no savestate loads.
- One action is chosen and sent per emulated frame. No routine blocks while the game advances.
- Every stuck recovery is bounded, records the failed attempt, and selects a materially different safe candidate or returns `no_safe_action`.
- Deterministic tests cover failures, alternatives, no-repeat memory, retreat decisions, powerup value, threat ranking, and phase transitions.
- Emulator evaluation reports real completion rate and failure causes. “Smart” or “completes a level” claims require real clean-start NES SMB1 runs.
