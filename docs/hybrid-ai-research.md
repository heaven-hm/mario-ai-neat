# Which AI hybrid fits SMB1 in FCEUX?

## Recommendation

Keep the in-emulator player small and predictable: **NEAT for evolving the
neural controller, plus contextual online Q-learning for fast action feedback,
plus the existing safety shield**. This branch now implements that hybrid.
The Q memory updates between decisions during an episode and transfers evidence
to nearby contexts such as the same enemy class at a slightly different
distance. It is not limited to waiting for an entire genome run to finish.

Do not add a chat or text-generating model to the frame loop. SMB1 control is a
small, timed action problem with structured RAM observations. A language model
does not provide a useful interface for predicting exact button presses every
few frames, and FCEUX embeds Lua 5.1 rather than Python/PyTorch. Transformers
become a credible experiment only after we collect full, high-quality state,
action, reward, and next-state trajectories for offline training.

## Comparison for this project

| Method | Fit for SMB1 + FCEUX | Main benefit | Main cost or risk | Decision |
|---|---|---|---|---|
| **NEAT / MarI/O style** | Excellent fit for the existing Lua controller and small action set | Evolves both weights and topology; works directly on observations | One full game attempt per genome makes feedback slow; generation count alone hides evaluation cost | Keep as the population-level optimizer |
| **NEAT + contextual Q memory** | Best incremental fit; now implemented in Lua | Reuses action value updates within a run and carries them across genomes and restarts; lightweight and persistent | Context bins can alias different geometry; reward shaping and transfer need real-ROM evaluation | Use as the active experimental hybrid |
| **Mixture of Experts (MoE)** | Possible later, currently premature | Separate policies could specialize in gaps, enemies, pipes, and power-ups | Needs enough examples per expert and a trained router; a few specialists can starve each other of data | Do not add until logs show distinct, recurring task regimes |
| **Decision Transformer / sequence model** | Python offline experiment, not a drop-in FCEUX script | Learns action sequences conditioned on state/history and target return | Requires trajectory rows, training compute, a runtime bridge, and validation against distribution shift | Consider after collecting transition data; current episode summaries are insufficient |
| **ChatGPT-style GenAI / Transformers LLM** | Poor live-control fit | Can explain logs or help author offline tools | Text generation is costly and nondeterministic for frame timing; does not create missing SMB1 experience | Keep out of the controller; optional analysis tool only |
| **Ape-X Rainbow DQN in PyTorch** | Current experimental Python path in this branch | C51, learner-side NoisyNet, Double/Dueling, per-actor n-step returns, global proportional PER, eight asynchronous FCEUX actors with epsilon-greedy policies, and a bounded 10,000-batch queue | Still needs controlled evaluation against NEAT, basic DDQN, and PPO on identical starts | Current Python benchmark candidate; not yet shown to outperform Lua NEAT |

## Why this is the practical hybrid

The original NEAT paper identifies speciation, innovation tracking, and
incremental topology growth as core strengths. The MarI/O family demonstrates
that a compact evolved network can control a Mario game. This project already
has those pieces, SMB1 RAM sensors, repeated fixed-start attempts, and an
embedded Lua runtime.

The previous experience memory only counted successful and failed context/action
pairs. It now also updates a bounded Q value from progress and game events after
each selected action finishes. Similar contexts contribute with lower weight;
different hazard families do not share values. NEAT still supplies the neural
scores, and the safety filter still removes unsafe choices. This combines
evolutionary search with a small online reinforcement-learning signal without
adding a large external runtime.

Decision Transformer is a real reinforcement-learning sequence-modeling method,
not a general-purpose chat model. Its paper evaluates offline datasets and
conditions generated actions on desired return and trajectory history. It would
need much richer data than this project's episode summaries currently store.
Likewise, MoE results in deep RL concern trained expert modules at larger model
scales; NEAT species are evolutionary compatibility groups, not MoE policies
selected by a learned router.

If we later move learning into Python, first add a robust FCEUX transition
collector/bridge and benchmark a compact **Rainbow or Double-DQN with prioritized
replay and n-step returns** against the current hybrid. PyTorch plus a tested RL
library is more relevant than `transformers` for that value-learning baseline.
Only after that should a Decision Transformer be tested on saved trajectories.
Do not combine MoE, an LLM, a Transformer, and NEAT at once: ablation would be
impossible and extra components do not guarantee faster learning.

## What “10 generations” can and cannot mean

This change makes useful feedback available during a genome's attempt; it does
not establish that a full 1-1 completion will happen in 10 generations. A
generation is tied to population size, so comparisons must report **episodes,
environment frames, wall-clock time, best progress, and clean-start completion
rate**. The current FCEUX run must be measured on the actual ROM. The 10-gen
target remains an experiment, not a promise.

For a meaningful trial, compare the old NEAT branch and this branch from copies
of the same checkpoint and fixed training start, with identical ROM, FCEUX,
population, and random seeds where available. Use multiple runs and evaluate the
champion separately from a clean level start. Keep this hybrid only if it reaches
milestones earlier without lowering completion rate.

## References

- Stanley and Miikkulainen, [Evolving Neural Networks through Augmenting Topologies](https://direct.mit.edu/evco/article/10/2/99/1123/Evolving-Neural-Networks-through-Augmenting), 2002.
- Chen et al., [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345), 2021.
- Schaul et al., [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952), 2016.
- Hessel et al., [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298), 2017.
- K. Li et al., [Mixtures of Experts Unlock Parameter Scaling for Deep RL](https://arxiv.org/abs/2402.08609), 2024.
- [FCEUX Lua Scripting reference](https://fceux.com/web/help/LuaScripting.html).
- [Vivek's SMB1 NEAT project](https://github.com/vivek3141/super-mario-neat), a Python/FCEUX comparison point.
- [SethBling's MarI/O source](https://gist.github.com/SethBling/598639f8d5e8afb5453a0b9519be51ff), the NEAT Mario reference.
