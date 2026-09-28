# Behavior and weakness audit

## Original LuaRio bot

| Area | Original behavior | Main weakness |
| --- | --- | --- |
| Default movement | Sends right+B repeatedly | Marches into danger and repeats one pattern |
| Enemy response | Uses sprite positions; reacts around close contact | Response starts late; enemy type/state is inconsistently paired with sprite; no learned outcome |
| Jumping | Uses right+B+A around selected threats | Does not reason about takeoff distance, current motion, or the outcome of an earlier jump |
| Fire Mario | Sends B/right to fire | Does not confirm that the enemy was hit or defeated |
| Powerups | Chases mushrooms/flowers with nested loops | Can block progress indefinitely and does not compare risk with reward |
| Recovery | Several loops and direct RAM writes | No clean failure feedback; can force Mario's state instead of learning a controller action |
| Timing | Uses wall-clock delays and nested frame advancement | Emulator speed changes behavior and decision timing |
| Learning | None | No state/action experience, persistent policy, fitness, or training history |

## Current implementation goals

- Train only for NES Super Mario Bros. 1 in FCEUX.
- Represent nearby tiles, active enemies, Mario's movement/form, and visible powerups as neural-network inputs.
- Let evolved genomes choose among run, jump-run, retreat, brake, jump-in-place, and walk controller actions.
- Prevent forward-only actions when an unpowered Mario is close to an enemy; expose jumping, braking, and retreat as choices.
- Score repeated attempts by furthest progress, survival, powerup form, and level completion.
- Persist the evolving population to a local database so later FCEUX sessions continue training.
- Never write NES RAM, clear enemy slots, elevate Mario, or change collision state.

## Remaining risks

- RAM addresses and enemy semantics still need to be checked in a live SMB1 ROM against the supplied disassembly.
- Evolution can take many complete level attempts; a fresh population is expected to play poorly.
- Fitness shaping can learn a bad shortcut or stall strategy. Training results need repeated clean-start evaluations.
- Safety filtering is approximate and may not prevent a death from an enemy already too close or from unsupported hazards.
- The current code has not been evaluated on an actual ROM in this workspace.

See [learning design](learning.md), [evaluation protocol](evaluation.md), and [RAM map](ram-map.md).
