# Evaluation protocol

No full-game result is claimed until runs are performed with an actual compatible NES SMB1 ROM in FCEUX.

For each run, record local ROM label/hash, FCEUX version, bot commit, player/form, frame count, world/stage reached, death count, recovery attempts, completion result, and trace path. Do not publish ROM bytes or upload a ROM-derived savestate.

Start cleanly from title for campaign runs. Do not use manual input after launching the bot, memory writes, or savestate loads. Run at least ten attempts for each claimed gate. First gate: SMB1 world 1-1. Then test the remaining stages and mechanics in order: gap and pipe jumps, power-up pursuit, moving lifts/springs, underwater movement, castle route, Bowser/axe, death/retry, and campaign transitions. Record failure cause and add a deterministic trace regression for each corrected bug.

Report attempts/completions and failure causes. A best run is not a completion rate. Do not claim full campaign completion until one clean-start campaign completion exists and repeated runs support the stated rate.
