"""Import and validate a locally supplied Super Mario Bros. (NES) ROM.

No ROM is shipped, downloaded, or redistributed by this repository. The
training and evaluation entry points require an explicit local file, validate
that it is an SMB1-shaped cartridge image, and record its fingerprints so a run
can be traced back to the exact image it used.

``gym_super_mario_bros`` resolves its ROM through a fixed internal lookup that
offers no path argument (``gym_super_mario_bros._roms.rom_path``), so
``use_local_rom`` substitutes that lookup for the supplied file. That seam is
the only supported way to train against a user-supplied image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROM_PATH = REPOSITORY_ROOT / "roms" / "super-mario-bros.nes"
ROM_ENVIRONMENT_VARIABLE = "SMB1_PPO_ROM"

INES_HEADER_SIZE = 16
INES_MAGIC = b"NES\x1a"
SMB1_PRG_KILOBYTES = 32
SMB1_CHR_KILOBYTES = 8
SMB1_MAPPER = 0


class RomError(RuntimeError):
    """Raised when a ROM is missing, malformed, or unusable for SMB1."""


@dataclass(frozen=True)
class Smb1Dump:
    """A publicly documented Super Mario Bros. cartridge dump."""

    label: str
    sha256: str
    md5: str
    sha1: str
    crc32: str


# Fingerprints of the widely distributed "Super Mario Bros. (World)" dump, the
# image the gym-super-mario-bros RAM map was written against. Several
# independent public ROM databases list the same four checksums.
KNOWN_DUMPS: tuple[Smb1Dump, ...] = (
    Smb1Dump(
        label="Super Mario Bros. (World)",
        sha256="f61548fdf1670cffefcc4f0b7bdcdd9eaba0c226e3b74f8666071496988248de",
        md5="811b027eaf99c2def7b933c5208636de",
        sha1="ea343f4e445a9050d4b4fbac2c77d0693b1d0922",
        crc32="3337ec46",
    ),
)


@dataclass(frozen=True)
class RomInspection:
    """Everything measured about a candidate ROM file."""

    path: str
    size_bytes: int
    sha256: str
    md5: str
    sha1: str
    crc32: str
    mapper: int
    prg_kilobytes: int
    chr_kilobytes: int
    vertical_mirroring: bool
    has_trainer: bool
    is_pal: bool
    verdict: str
    dump_label: str | None

    @property
    def is_known_dump(self) -> bool:
        return self.verdict == "known-good"

    @property
    def is_usable(self) -> bool:
        return self.verdict in {"known-good", "unrecognized"}

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def fingerprint(data: bytes) -> dict[str, str]:
    """Return the four checksums public ROM databases list."""
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "md5": hashlib.md5(data).hexdigest(),
        "sha1": hashlib.sha1(data).hexdigest(),
        "crc32": format(zlib.crc32(data) & 0xFFFFFFFF, "08x"),
    }


def inspect(path: Path) -> RomInspection:
    """Measure and structurally validate a candidate SMB1 ROM."""
    path = Path(path)
    if not path.is_file():
        raise RomError(f"ROM file not found: {path}")
    data = path.read_bytes()
    if len(data) <= INES_HEADER_SIZE:
        raise RomError(f"{path} is too small to be an iNES image ({len(data)} bytes)")
    if data[:4] != INES_MAGIC:
        raise RomError(f"{path} does not start with the iNES magic 'NES\\x1a'; it is not a NES ROM")
    prg_kilobytes = data[4] * 16
    chr_kilobytes = data[5] * 8
    flags_six = data[6]
    flags_seven = data[7]
    mapper = (flags_six >> 4) | (flags_seven & 0xF0)
    has_trainer = bool(flags_six & 0b0000_0100)
    is_pal = bool(data[9] & 0b0000_0001)
    if has_trainer:
        raise RomError(f"{path} contains a 512-byte trainer, which nes-py rejects")
    if mapper != SMB1_MAPPER:
        raise RomError(f"{path} uses mapper {mapper}; Super Mario Bros. is NROM (mapper 0)")
    if (prg_kilobytes, chr_kilobytes) != (SMB1_PRG_KILOBYTES, SMB1_CHR_KILOBYTES):
        raise RomError(
            f"{path} carries {prg_kilobytes} KiB PRG and {chr_kilobytes} KiB CHR; "
            f"Super Mario Bros. needs {SMB1_PRG_KILOBYTES} KiB PRG and "
            f"{SMB1_CHR_KILOBYTES} KiB CHR"
        )
    if is_pal:
        raise RomError(f"{path} is flagged PAL; nes-py only supports NTSC images")
    expected_size = INES_HEADER_SIZE + (prg_kilobytes + chr_kilobytes) * 1024
    if len(data) != expected_size:
        raise RomError(f"{path} is {len(data)} bytes; its header declares {expected_size}")

    sums = fingerprint(data)
    matched = next((dump for dump in KNOWN_DUMPS if dump.sha256 == sums["sha256"]), None)
    return RomInspection(
        path=str(path),
        size_bytes=len(data),
        mapper=mapper,
        prg_kilobytes=prg_kilobytes,
        chr_kilobytes=chr_kilobytes,
        vertical_mirroring=bool(flags_six & 0b0000_0001),
        has_trainer=has_trainer,
        is_pal=is_pal,
        verdict="known-good" if matched else "unrecognized",
        dump_label=matched.label if matched else None,
        **sums,
    )


def default_rom_path() -> Path:
    """Return the ROM path to use when none is given on the command line."""
    override = os.environ.get(ROM_ENVIRONMENT_VARIABLE)
    return Path(override).expanduser() if override else DEFAULT_ROM_PATH


def resolve_rom(path: Path | str | None) -> Path:
    """Normalize a user-supplied ROM path, falling back to the default."""
    return Path(path).expanduser() if path else default_rom_path()


def require_rom(path: Path | str | None = None) -> RomInspection:
    """Return the inspection of a usable ROM or explain exactly how to supply one."""
    resolved = resolve_rom(path)
    if not resolved.is_file():
        raise RomError(
            f"no Super Mario Bros. ROM at {resolved}\n"
            "This project never ships or downloads ROMs. Supply your own legally "
            "obtained cartridge dump and import it:\n"
            f"  python -m smb1_ppo.rom import --source /path/to/Super Mario Bros. (World).nes\n"
            f"or point {ROM_ENVIRONMENT_VARIABLE} at it."
        )
    return inspect(resolved)


def import_rom(
    source: Path | str,
    destination: Path | str | None = None,
    allow_unverified: bool = False,
    force: bool = False,
) -> RomInspection:
    """Copy a validated ROM into the local run tree and write a metadata sidecar."""
    source = Path(source).expanduser()
    destination = resolve_rom(destination)
    if source.resolve() == destination.resolve():
        raise RomError(f"source and destination are the same file: {source}")
    inspection = inspect(source)
    if not inspection.is_known_dump and not allow_unverified:
        raise RomError(
            f"{source} is an SMB1-shaped image but its SHA-256 "
            f"({inspection.sha256}) is not a known Super Mario Bros. (World) dump.\n"
            "Training would run against an image this project cannot identify, so "
            "every measurement would be unverifiable. Re-run with --allow-unverified "
            "only if you are certain this is Super Mario Bros. for NES."
        )
    if destination.exists() and not force:
        raise RomError(f"{destination} already exists; pass --force to replace it")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    copied = inspect(destination)
    if copied.sha256 != inspection.sha256:
        raise RomError("the copied ROM does not match the source checksum")
    (destination.parent / "rom-import.json").write_text(
        json.dumps(
            {
                "imported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source": str(source),
                "destination": str(destination),
                "verdict": copied.verdict,
                "dump_label": copied.dump_label,
                "checksums": {
                    "sha256": copied.sha256,
                    "md5": copied.md5,
                    "sha1": copied.sha1,
                    "crc32": copied.crc32,
                },
                "header": {
                    "mapper": copied.mapper,
                    "prg_kilobytes": copied.prg_kilobytes,
                    "chr_kilobytes": copied.chr_kilobytes,
                    "vertical_mirroring": copied.vertical_mirroring,
                    "is_pal": copied.is_pal,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return copied


def use_local_rom(path: Path | str) -> None:
    """Make gym-super-mario-bros load the supplied ROM instead of its own.

    The upstream environment hard-codes ``rom_path(lost_levels, rom_mode)``
    inside ``SuperMarioBrosEnv.__init__``, so the module attribute it calls is
    the only injection point. Only the plain SMB1 vanilla image is accepted:
    the ROM-hack modes change the pixels the CNN sees and would silently
    invalidate a benchmark against the Rainbow system.
    """
    from gym_super_mario_bros import smb_env

    requested = str(Path(path).expanduser().resolve())

    def rom_path(lost_levels: bool, rom_mode: str) -> str:
        if lost_levels:
            raise RomError("this project trains Super Mario Bros. 1 only, not Lost Levels")
        if rom_mode != "vanilla":
            raise RomError(f"only the unmodified SMB1 image is supported, not {rom_mode!r}")
        return requested

    smb_env.rom_path = rom_path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate or import a local Super Mario Bros. ROM.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    importer = subparsers.add_parser("import", help="copy a validated ROM into roms/")
    importer.add_argument("--source", type=Path, required=True)
    importer.add_argument("--destination", type=Path, default=None)
    importer.add_argument("--allow-unverified", action="store_true")
    importer.add_argument("--force", action="store_true")
    checker = subparsers.add_parser("check", help="inspect a ROM without copying it")
    checker.add_argument("--path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    options = arguments()
    try:
        if options.command == "import":
            inspection = import_rom(
                options.source,
                options.destination,
                allow_unverified=options.allow_unverified,
                force=options.force,
            )
        else:
            inspection = require_rom(options.path)
    except RomError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps(inspection.as_dict(), indent=2))


if __name__ == "__main__":
    main()
