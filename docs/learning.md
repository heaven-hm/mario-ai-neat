# Learning design

## Training loop

The Lua bot runs one genome at a time. When SMB1 first enters active gameplay, it stores that game state in a persistent, in-memory FCEUX savestate. After each genome dies, reaches the flag, or stops making progress, the bot scores it and reloads that start state for the next genome. Once all genomes have played, it forms the next generation and repeats.

All genomes therefore see the same starting state within a generation. Episodes end after death, victory, 600 frames without new forward progress, or a 12,000-frame cap. A player can leave the bot running for repeated generations; training is automatic after the script starts.

## Observation and action

The input vector contains a 13-by-13 nearby grid, with solid tiles encoded as `1`, active enemies as `-1`, and empty cells as `0`. Fifteen additional values describe horizontal/vertical movement, grounded state, size/power, the closest enemy's relative position/speed/type, any visible powerup's presence/relative position/type, a forward-gap signal, and immediate enemy contact. A constant bias input completes the network input. Collected upgrades also contribute to episode fitness.

Six network outputs score valid controller actions: run, jump while running, retreat, brake, jump in place, and walk. A safety filter narrows the available choices when an active enemy is close ahead. It blocks forward-only movement for unpowered Mario, leaves a jump response available before contact with a stompable ground enemy, and allows Fire Mario to use forward fire at a safer distance.

## Fitness and evolution

Fitness is based on furthest rightward progress from the episode start and frames survived. Death loses 120 fitness points; reaching the flag earns 10,000; collecting a stronger form earns a bonus; repeated no-progress and timeout episodes are penalized. A population's fitter genomes become parents. Related genomes are grouped into species so a new topology can survive while it competes. Crossover combines matching genes, while mutations change weights, add connections, split connections with new nodes, and enable or disable genes. The top genome carries into the next generation.

This is neuroevolution in the style of NEAT, used as an evolutionary reinforcement-learning method. The FCEUX play result supplies the reward signal; no labeled human action traces or external training service are used.

## Saved database

`mario_ai_heaven_neat.db` is a line-based database beside the Lua script. It contains the generation counter, population size, genome fitness, mutation rates, network connections, weights, and innovation IDs. The bot writes a temporary file and replaces the database, saves after every episode, periodically during long attempts, and when FCEUX stops the script. On the next launch it resumes with that population. Deleting the database starts a fresh population.

The database does not contain the ROM or a savestate. Keep it with the version of the Lua script that created it; input/action layout changes can require a fresh database.

## References

- [MarI/O Lua gist supplied for this project](https://gist.github.com/d12frosted/7471e2123f10485d96bb). Its header requests that the code not be redistributed; this project uses the general NEAT approach and implements its own code.
- [FCEUX Lua help: savestates](https://fceux.com/web/help/LuaFunctionsList.html). FCEUX documents creating, saving, loading, and preserving an in-memory savestate.
- [FCEUX Lua help: controller input and frame advance](https://fceux.com/web/help/Commands.html).
