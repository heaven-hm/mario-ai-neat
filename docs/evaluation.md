# Evaluation protocol

No full-game result is claimed until training and validation run against a real compatible NES SMB1 ROM in FCEUX.

## Training

Start the bot at the beginning of World 1-1. Let it evaluate full generations. Keep the generated `mario_ai_heaven_neat.db` so later sessions continue evolving the population. Record the FCEUX version, ROM label/hash locally, bot commit, generation, best fitness, best world position, completion count, and database backup. Do not publish ROM bytes or ROM-derived savestates.

The training loop uses only controller input and does not reload FCEUX savestates. Record game position and episode context with each result because candidates can begin in different game situations. Do not treat training episodes as independent clean-start completion trials.

## Validation

After the database has evolved, evaluate the best saved population from a clean title-screen start without manual input. Run at least ten attempts for the claimed gate and record completions, deaths, furthest stage, and failure causes. A best run is not a completion rate. Do not claim full-campaign completion until the bot completes a clean-start campaign and repeated attempts support the stated rate.

Test SMB1 1-1 first, then gaps, pipes, powerups, lifts/springs, underwater movement, castle route, Bowser/axe, death/retry, and level transitions. The Lost Levels, hacks, and other emulators are out of scope.
