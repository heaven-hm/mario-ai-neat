# Known limitations

- The target is only NES Super Mario Bros. 1 running in FCEUX. Other games, ROM revisions, and emulators are not supported or validated.
- RAM meanings and tile-buffer indexing are inherited from `LuaRio_Bot.lua` and the supplied SMB1 disassembly. The current container has no compatible ROM for live address verification.
- No actual gameplay training was possible in this workspace. Synthetic tests verify the trainer and its safety filter, not winning behavior.
- NEAT needs many complete episodes to improve. New databases use 300 genomes; one generation can take substantial emulator time. Evolution may initially perform worse than the original hand-coded bot.
- Fitness rewards forward progress and survival, so it can favor fast progress over optional coins or score items. Powerups are visible to the policy and upgrades earn a fitness bonus, but path safety and item value are still approximated.
- The safety filter reduces direct enemy collisions but cannot guarantee survival. Enemy speed, invulnerability, projectile timing, and Mario's precise collision box are only approximated.
- The bot reserves FCEUX savestate slot 9 and overwrites it at the beginning of training. It avoids `savestate.persist()` because the Homebrew Apple Silicon FCEUX 2.6.6 build crashed in that native call. A build without `savestate.object()` or `savestate.create()` falls back to continuous training, where episodes are less directly comparable.
- The learned population is local to `mario_ai_heaven_neat.db`. Back up that file to preserve training. Changing the input/action layout makes older databases incompatible.
