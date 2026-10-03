"""SMB1-only PPO training system.

This package is a self-contained baseline for original NES Super Mario Bros.
It is deliberately independent of the FCEUX Ape-X Rainbow trainer in
``python/mario_ai_fceux``: it uses the proven ``gym-super-mario-bros`` /
``nes-py`` environment instead of the FCEUX Lua RAM bridge, so Rainbow
checkpoints (``model.pt``) are not loadable here by design. See
``docs/smb1-ppo.md`` for the full rationale.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
