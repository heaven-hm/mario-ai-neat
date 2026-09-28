# NES SMB1 RAM map status

The user-provided [SMB disassembly gist](https://gist.github.com/1wErt3r/4048722) is the required source of truth. The legacy bot also includes addresses in its own code/header. This table deliberately distinguishes inherited values from values that still need source confirmation. Do not add new behavior based only on a guessed address.

| Address | Meaning in legacy script | Status |
| --- | --- | --- |
| `0x000E` | Player state; `0x0B` is dying | Inherited from legacy code |
| `0x000F..0x0013` | Five enemy-present slots | Inherited from legacy code |
| `0x0016..0x001A` | Five enemy type slots | Inherited from legacy code |
| `0x001E..0x0022` | Five enemy state slots | Inherited from legacy code |
| `0x006D`, `0x0086` | Player world page and screen X | Inherited from legacy code |
| `0x006E..0x0072`, `0x0087..0x008B` | Enemy page and screen X slots | Inherited from legacy code |
| `0x0057` | Player speed byte | Inherited from legacy code; signed fixed-point interpretation needs confirmation |
| `0x0058..0x005C` | Enemy speed bytes | Inherited address pattern; per-slot meaning needs confirmation |
| `0x009F` | Player vertical speed; legacy `setVelocity()` writes here | Inherited use; write is prohibited in production |
| `0x00CF..0x00D3` | Enemy Y slots | Inherited from legacy code |
| `0x03AD`, `0x03B8` | Player screen-relative X/Y | Inherited from legacy code |
| `0x0490` | Collision bits checked against `0xFE` | Inherited; do not use as a substitute for terrain understanding |
| `0x0500..0x069F` | Candidate two-page tile buffer | Needs direct confirmation from supplied disassembly before production map use |
| `0x0712` | Death music flag | Inherited from legacy code |
| `0x070F`, `0x010E` | Flagpole collision values | Inherited from legacy code |
| `0x0754`, `0x0756` | Player size and power state | Inherited from legacy code |
| `0x0770`, `0x0772` | Candidate game/operation mode bytes | `0x0772` is used in legacy death clear; exact phase meanings need disassembly confirmation |
| Power-up bytes in `checkPowerUp()` | Drawn/type/direction/screen position | Inherited, but world coordinates and semantics need confirmation |

Before enabling a value in `src/config.lua`, update this table with the disassembly symbol, definition/use location, and observed emulator value. Unverified values must not cause blocking, unsafe movement, or a false completion result. The gist page exceeds the current web reader's content limit; the executing developer must inspect the specific source lines locally or through the gist's raw file view.
