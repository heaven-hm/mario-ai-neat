# Mario AI NEAT

A learning AI for **Super Mario Bros. 1 on NES, running only in FCEUX**. It reads the SMB1 RAM layout, observes nearby tiles and enemies, and evolves a neural-network controller across repeated play attempts. It is based on the MarI/O project and NEAT approach described below, adapted specifically for SMB1 and FCEUX. Super Mario World, other games, and other emulators are outside this project's scope.

![Mario AI NEAT training in FCEUX](docs/images/mario-ai-neat-training.png)

*Live FCEUX training: the HUD explains the active NEAT genome, chosen action, sensed threat, progress, and saved-learning state.*

## What kind of AI is this?

Mario AI NEAT is a **NEAT-style neuroevolution system**, which is a form of machine learning. It uses an evolutionary reinforcement signal: neural-network controllers play SMB1, receive fitness from their results, and reproduce according to that fitness. It is not a language model, generative AI, Q-learning, PPO, or a network trained with backpropagation.

The controller is a hybrid system. NEAT learns which action to prefer, while a small deterministic safety layer removes immediately unsafe choices such as running directly into a close enemy. The learned neural network still decides whether to run, jump, brake, retreat, or walk among the allowed actions.

## Technology stack

| Layer | Technology | Role |
| --- | --- | --- |
| Game | NES Super Mario Bros. 1 | The only supported game; the ROM is not included |
| Emulator | FCEUX 2.x | Runs SMB1, exposes RAM, controller, frame, GUI, and savestate APIs |
| Runtime | Embedded Lua 5.1 | Executes the complete AI inside FCEUX |
| Machine learning | NEAT-style neuroevolution | Evolves neural-network connections, weights, nodes, and mutation rates |
| Reinforcement signal | Episode fitness | Rewards progress, survival, stronger forms, and victory; penalizes death and stalls |
| Model storage | `mario_ai_neat.db` | Persists generations, genomes, genes, innovation IDs, mutation rates, and fitness |
| Diagnostics | `mario_ai_neat.log` and FCEUX HUD | Records episodes and explains the live decision context |
| Verification | Lua behavior and mocked-FCEUX tests | Checks sensors, networks, evolution, persistence, compatibility, and controller behavior |

No Python process, ML framework, compiler, GPU runtime, cloud service, or network connection is required during training.

## AI architecture

```mermaid
flowchart LR
    Game["SMB1 running in FCEUX"]
    RAM["RAM observer<br/>tiles, Mario, enemies, items"]
    Encoder["Observation encoder<br/>185 neural inputs"]
    Genome["NEAT genome<br/>evolving nodes and weighted genes"]
    Scores["6 action scores"]
    Shield["Safety layer<br/>removes immediately unsafe actions"]
    Pad["NES controller input<br/>A, B, Left, Right"]
    Result["Episode result<br/>progress, survival, power, death, victory"]
    Fitness["Fitness function"]
    Evolution["Speciation, selection,<br/>crossover and mutation"]
    Database[("mario_ai_neat.db")]

    Game --> RAM --> Encoder --> Genome --> Scores --> Shield --> Pad --> Game
    Game --> Result --> Fitness --> Evolution --> Genome
    Evolution <--> Database
```

### Observation and action model

| Neural interface | Size | Contents |
| --- | ---: | --- |
| Local grid | 169 | A 13×13 area around Mario: solid tile `1`, active enemy `-1`, empty space `0` |
| Global features | 15 | Velocity, grounded state, size/power, closest enemy, visible power-up, forward gap, contact danger |
| Bias | 1 | Constant input that lets actions activate without a particular sensor |
| **Total inputs** | **185** | Values evaluated by each genome every decision frame |
| Outputs | 6 | Run, running jump, retreat, brake, jump in place, controlled walk |

## How NEAT learns

Each genome is one candidate neural-network brain. A gene records a source node, destination node, weight, enabled state, and historical innovation ID. Innovation IDs let crossover align equivalent connections even after different genomes evolve different structures.

```mermaid
flowchart TD
    Start["Create or load population<br/>300 genomes for a new database"]
    State["Save one fixed SMB1 start<br/>in FCEUX slot 9"]
    Run["Run one genome from the fixed state"]
    Score["Calculate episode fitness"]
    More{"Every genome evaluated?"}
    Group["Group compatible genomes into species"]
    Cull["Cull weak and stale genomes<br/>preserve the champion"]
    Breed["Crossover fitter parents"]
    Mutate["Mutate weights, links, nodes,<br/>enabled genes and mutation rates"]
    Next["Next generation"]
    Save["Save population database"]

    Start --> State --> Run --> Score --> More
    More -- No --> State
    More -- Yes --> Group --> Cull --> Breed --> Mutate --> Next --> Save --> State
```

The implementation includes the main NEAT mechanisms:

- **Topology evolution:** mutations can add a connection or split an existing connection to create a hidden node.
- **Weight evolution:** connection weights are perturbed or replaced.
- **Historical markings:** innovation IDs align matching genes during crossover.
- **Speciation:** structural and weight distance separates different network families so new structures have time to improve.
- **Fitness sharing:** global rank is adjusted by species size before parent selection.
- **Crossover:** matching genes can come from either parent; unmatched structure follows the fitter parent.
- **Adaptive mutation:** mutation probabilities themselves drift slightly between generations.
- **Staleness control:** species that stop improving are removed after 15 generations unless they contain the global champion.
- **Elitism:** the strongest genome is copied unchanged into the next generation.
- **Seeded prior:** generation one begins with a small useful bias toward running and jumping for enemies or gaps; evolution may replace all of it.

### What a genome represents

A genome is one candidate controller. Its **nodes** are neural-network inputs, optional hidden neurons, and action outputs. Its **genes** are the weighted connections between those nodes. The 185 observations and six action scores stay fixed; evolution changes the connections, their weights, and sometimes the number of hidden nodes.

```mermaid
flowchart LR
    Obs["SMB1 observations<br/>185 input values"] --> Inputs["Input nodes<br/>tile grid + Mario state"]
    Inputs -->|"connection gene"| Hidden["Hidden nodes<br/>zero or more; topology evolves"]
    Hidden -->|"connection gene"| Actions["6 action outputs<br/>run · jump · retreat<br/>brake · jump in place · walk"]
    Inputs -->|"connection gene"| Actions
```

Each connection gene stores its source node, target node, weight, enabled state, and innovation ID. The innovation ID is the historical label used to recognize corresponding connections during species comparison and crossover.

### How topology grows

An add-node mutation splits an existing connection. The old connection is disabled, a hidden node is inserted, and two new connection genes are created. This lets evolution add complexity gradually while preserving a path for the signal through the network.

```mermaid
flowchart LR
    subgraph Before["Before: one connection"]
        SourceA["Input"] -->|"weight 0.7 · innovation 42"| ActionA["Action output"]
    end
    subgraph After["After: add-node mutation"]
        SourceB["Input"] -->|"new gene · weight 1.0"| HiddenB["New hidden node"]
        HiddenB -->|"new gene · weight 0.7"| ActionB["Action output"]
    end
    Before -. "split connection" .-> After
```

### How parents produce a new genome

After every genome has played an episode, the trainer ranks results and groups similar genomes into species. Parents are selected within species. Matching innovation IDs identify corresponding genes; a child can inherit either parent's version of a matching gene. Unmatched genes come from the fitter parent, then mutation can change weights or topology.

```mermaid
flowchart TD
    Fitter["Fitter parent<br/>genes 4, 7, 9"] --> Align["Align connections<br/>by innovation ID"]
    Other["Other parent<br/>genes 4, 8, 9"] --> Align
    Align --> Match["Matching genes 4 and 9<br/>inherit from either parent"]
    Align --> Unique["Unmatched gene 7<br/>keep from fitter parent"]
    Match --> Child["Child genome"]
    Unique --> Child
    Child --> Mutate["Mutate weights and topology"]
    Mutate --> Evaluate["Evaluate in SMB1"]
```

In this illustration, gene 8 is unique to the less-fit parent, so it is not copied. Species protect different network structures while they are being evaluated; fitness sharing and champion preservation help balance exploration with retaining the strongest result.

### Why the project says “NEAT-style”

It implements NEAT's defining ideas—historical innovation numbers, topology growth, speciation, crossover, mutation, fitness sharing, and champion preservation—but it is specialized for SMB1 rather than a byte-for-byte copy of the original NEAT paper. Its six outputs select complete controller actions, the first genome has an SMB1 movement prior, fitness uses game progress and survival, and a deterministic safety layer rejects immediately dangerous actions. These choices make the learner practical inside FCEUX while keeping the neural policy and its topology trainable through evolution.

## Run and train

1. Open a compatible SMB1 NES ROM in FCEUX, preferably at the start of World 1-1.
2. Load `mario_ai_neat.lua` from FCEUX's Lua script menu.
3. Leave the script running. On the first active SMB1 frame it saves a fixed training start in FCEUX savestate slot 9. It then tests each genome from that same state, scores the attempt, breeds a new generation, and repeats.
4. Stop the script when you want. The population database is saved periodically, after every completed attempt, and when FCEUX stops the script. Leave the database beside the Lua script to continue learning later.
5. Read `mario_ai_neat.log` beside the script for startup, episode, and database-save events.

### Resume or restart training

`mario_ai_neat.db` is the included learning checkpoint. It contains the saved NEAT population, including its generation, genomes, mutation rates, connection weights, innovation IDs, and fitness values. The database is a plain-text file tracked in this repository; `.gitignore` explicitly allows this file while ignoring other local databases.

### Load the included database in FCEUX

1. Keep `mario_ai_neat.lua` and `mario_ai_neat.db` together in the same folder. The repository already places them together; if you copy the Lua script elsewhere, copy the database beside it too.
2. Open the compatible Super Mario Bros. 1 ROM in FCEUX.
3. Use FCEUX's Lua script menu to load `mario_ai_neat.lua`. Do not load the `.db` file as a Lua script or through a separate database-import menu.
4. The script automatically looks for `mario_ai_neat.db` beside its own Lua file, loads the saved population, and logs the loaded generation/population. Keep `PLAY_CHAMPION_ONLY = false` to continue training from that population.

The script loads and saves the exact filename `mario_ai_neat.db`; it does not automatically discover `mario_ai_heaven_neat.db` or other names. If your trained checkpoint has a different name, stop the Lua script in FCEUX first, make a backup, then copy or rename that checkpoint to `mario_ai_neat.db` beside the script. Keep the backup outside the active filename so FCEUX cannot overwrite it. Do not replace the included database while training is running.

The learner periodically saves after attempts and at checkpoints, so loading the script again resumes from the last saved population. To start over, stop the script, move `mario_ai_neat.db` to a backup location, and load the Lua script; a fresh 300-genome population is created automatically.

To play the saved champion without changing the database, set `PLAY_CHAMPION_ONLY = true`, load the script, and begin SMB1 manually. Set it back to `false` and reload the script to resume evolution from the same database.

The AI uses FCEUX's predefined slot 9 for fair training episodes. This overwrites that slot, so reserve it for Mario AI NEAT. It uses `savestate.object()` when available and the older `savestate.create()` compatibility API otherwise. It never calls `savestate.persist()`, the native FCEUX function that crashed on the Homebrew Apple Silicon build. If a FCEUX build has no compatible savestate API, the AI logs the condition and continues with less-controlled input-only episodes.

New databases contain 300 genomes. Existing databases retain their current population size so that previous learning is not discarded. Delete `mario_ai_neat.db` to begin a new 300-genome run.

## Champion play

After training, set `local PLAY_CHAMPION_ONLY = true` near the top of `mario_ai_neat.lua`. The AI loads the genome with the highest saved fitness and repeatedly plays it from slot 9 without mutation, crossover, or generation changes. Set it back to `false` to resume training.

## Testing aids

The current testing build refreshes the SMB1 timer to `999` while gameplay is active and refreshes the lives byte at `0x075A` to `9`. This prevents a training run from reaching Game Over, while SMB1 still performs every normal death and respawn. The AI never presses Start automatically; begin a game manually in FCEUX. Before a real evaluation, change `TESTING_FREEZE_TIMER` and `TESTING_INFINITE_LIVES` near the top of `mario_ai_neat.lua` to `false`.

## What it senses and learns

The AI uses the SMB1 positions and tile data from the legacy script and MarI/O's SMB1 sensor layout: a nearby tile/enemy grid plus Mario movement and power state. The neural network scores controller actions. A safety filter removes forward-only actions when an unpowered Mario is close to an enemy, while preserving learned choices such as jumping, braking, and retreating. The original controller's jump-over-ground-enemies and fire-as-Fire-Mario behaviors inform that filter.

Each attempt earns fitness for furthest forward progress and survival, with a large bonus for reaching the flag. Completed generations retain a champion, group related genomes into species, select fitter parents, cross over matching genes, and mutate connections and weights. This is evolutionary reinforcement learning: the learned population persists in the database and is evaluated during real SMB1 play sessions.

The FCEUX overlay shows the active generation, genome, species, current lesson, chosen action, observed threat or gap, progress, and database status. It does not show testing-aid settings.

## Background and attribution

This project is based on [MarI/O by SethBling](https://gist.github.com/d12frosted/7471e2123f10485d96bb), which demonstrated NEAT neuroevolution playing **Super Mario World**. Watch SethBling's [MarI/O: Machine Learning for Video Games](https://www.youtube.com/watch?v=qv6UVOQ0F44). Mario AI NEAT adapts that project's learning approach to **Super Mario Bros. 1 for the NES, running exclusively in the FCEUX emulator**. It is not a Super Mario World project and does not target other games or emulators.

The neuroevolution method is based on the original paper by Kenneth O. Stanley and Risto Miikkulainen, [“Evolving Neural Networks through Augmenting Topologies”](https://direct.mit.edu/evco/article/10/2/99/1123/Evolving-Neural-Networks-through-Augmenting), *Evolutionary Computation*, 10(2), 99–127 (2002). This repository contains its own SMB1/FCEUX-oriented implementation and does not redistribute MarI/O's source code. The referenced MarI/O gist requests that its code not be redistributed.

## Tests and limitations

```sh
sh tests/run.sh
```

Tests cover neural-network evaluation, enemy sensors, enemy safety filtering, population save/load, generation breeding, the FCEUX compatibility layer, and the controller loop. RAM writes are restricted to the documented testing timer and lives aids. Tests do not establish that a learned genome beats the game. See [evaluation](docs/evaluation.md), [learning](docs/learning.md), [limitations](docs/limitations.md), and [RAM map](docs/ram-map.md).

The target RAM layout is the SMB1 revision described by the original bot and the linked [SMB disassembly](https://gist.github.com/1wErt3r/4048722). Other revisions and ROM hacks are not validated.

## Files

- `mario_ai_neat.lua`: self-contained FCEUX SMB1 AI, sensors, NEAT trainer, episode loop, and database persistence; automatically loads the adjacent database.
- `mario_ai_neat.db`: included NEAT population checkpoint used to resume training or play its saved champion.
- `legacy/LuaRio_Bot_v1.lua`: original bot preserved byte-for-byte.
- `docs/learning.md`: training loop, fitness, genome database, and restart behavior.
- `docs/requirements-and-weaknesses.md`: behavior inventory and remaining risks.
- `docs/ram-map.md`: inherited RAM addresses and verification status.

The legacy header credits Haseeb Mir, SethBling, and doppelganger. The original file did not declare a license, so this repository does not add one without rights confirmation.
