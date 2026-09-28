# Known limitations

- Some RAM semantics and tile-buffer indexing are inherited from `LuaRio_Bot.lua` and still need live verification against the supplied disassembly and target ROM.
- No compatible ROM is present in this workspace, so clean-start gameplay validation is pending.
- The Lost Levels and modified ROMs are outside the current target.
- Jump physics and collision predictions are approximate. Swimming, moving platforms, castle mazes, and several special enemies need dedicated behavior.
- The current policy is deterministic search with structured recovery, not a trained ML policy. RLHF and training from human play traces are future work.
- Passing synthetic tests does not establish human-like behavior or full-campaign completion; those claims require clean-start gameplay runs.
