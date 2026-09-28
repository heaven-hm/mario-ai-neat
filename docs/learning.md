# Learning design

## Training loop

The Lua bot runs one genome at a time. Start SMB1 manually, then the bot saves the first active frame to FCEUX predefined slot 9. After a genome dies, reaches the flag, or stops making progress, the bot scores it, restores slot 9, and tests the next genome from the same point. Once all genomes have played, it forms the next generation and repeats. During death, title, and transition screens it releases every controller button; it never presses Start.

The bot uses a predefined slot because FCEUX keeps predefined states across loads. It selects the documented `savestate.object(slot)` API and falls back to the older `savestate.create(slot)` API when needed. It never calls `savestate.persist()`, which crashed in the Homebrew Apple Silicon FCEUX build. If neither API is available, training continues without resets and the log records the fallback. Episodes end after death, victory, 600 frames without new forward progress, or a 12,000-frame cap.

## Observation and action

The input vector contains a 13-by-13 nearby grid, with solid tiles encoded as `1`, active enemies as `-1`, and empty cells as `0`. Fifteen additional values describe horizontal/vertical movement, grounded state, size/power, the closest enemy's relative position/speed/type, any visible powerup's presence/relative position/type, a forward-gap signal, and immediate enemy contact. A constant bias input completes the network input. Collected upgrades also contribute to episode fitness.

Six network outputs score valid controller actions: run, jump while running, retreat, brake, jump in place, and walk. A safety filter narrows the available choices when an active enemy is close ahead. It blocks forward-only movement for unpowered Mario, leaves a jump response available before contact with a stompable ground enemy, and allows Fire Mario to use forward fire at a safer distance.

## Fitness and evolution

Fitness is based on furthest rightward progress from the episode start and frames survived. Death loses 120 fitness points; reaching the flag earns 10,000; collecting a stronger form earns a bonus; repeated no-progress and timeout episodes are penalized. A population's fitter genomes become parents. Related genomes are grouped into species so a new topology can survive while it competes. Crossover combines matching genes, while mutations change weights, add connections, split connections with new nodes, and enable or disable genes. The top genome carries into the next generation.

This is neuroevolution in the style of NEAT, used as an evolutionary reinforcement-learning method. The FCEUX play result supplies the reward signal; no labeled human action traces or external training service are used.

## NEAT implementation details

A genome is a directed neural graph. Every gene contains `into`, `out`, `weight`, `enabled`, and `innovation` fields. The evaluator loads the 184 observed features plus one bias node, follows enabled weighted connections, applies the NEAT sigmoid activation, and reads six action scores.

The population starts from a sparse seeded controller rather than a fully connected network. Structural mutation can add links or split an enabled link into two links with a new hidden node. Weight mutation perturbs most existing weights and occasionally replaces a weight. Enable and disable mutations can restore or suppress individual connections. Each structural connection receives a population-wide innovation ID so crossover can match homologous genes.

Compatibility distance combines unmatched genes and the average weight difference of matching genes. Genomes below the distance threshold share a species. Global fitness rank is divided by species size for selection pressure, while the champion remains protected. Species stop reproducing after 15 stale generations unless they contain the global champion. In a species with multiple genomes, crossover is chosen with a 75% probability and the resulting child is mutated; the alternative copies one selected parent and mutates it. A species containing only one genome currently reproduces by cloning that genome.

Mutation rates are stored per genome and multiplied by either `0.95` or `1.05263` as evolution proceeds. This allows different lineages to explore different rates of weight, link, node, bias, enable, and disable mutation. The database persists these rates along with every connection and innovation ID.

The implementation is called NEAT-style because it adapts the algorithm to this game. Six outputs represent complete SMB1 controller actions, the initial population includes one seeded movement prior, and a deterministic safety filter can remove an immediately dangerous option before the highest remaining output is used. The evolving genome still supplies the learned policy; the filter does not move Mario or manufacture a jump through RAM.

## Saved database

`mario_ai_heaven_neat.db` is a line-based database beside the Lua script. It contains the generation counter, population size, genome fitness, mutation rates, network connections, weights, and innovation IDs. Fresh databases start with 300 genomes. The bot writes a temporary file and replaces the database, saves after every episode, periodically during long attempts, and when FCEUX stops the script. On the next launch it resumes with that population. Deleting the database starts a fresh population.

The database does not contain the ROM or a savestate. `mario_ai_heaven.log` records startup, episode starts and ends, and database-save failures. Keep the database with the version of the Lua script that created it; input/action layout changes can require a fresh database.

## Champion play

Set `PLAY_CHAMPION_ONLY` to `true` after a trained database exists. The bot selects the highest-fitness genome, restores the fixed slot after each result, and does not modify the population. This mode is for observing a mature controller; set the option back to `false` to train again.

## References

- [MarI/O Lua gist supplied for this project](https://gist.github.com/d12frosted/7471e2123f10485d96bb). Its header requests that the code not be redistributed; this project uses the general NEAT approach and implements its own code.
- [FCEUX Lua help: controller input and frame advance](https://fceux.com/web/help/Commands.html).
- [FCEUX Lua help: predefined savestate slots](https://fceux.com/web/help/LuaFunctionsList.html).
