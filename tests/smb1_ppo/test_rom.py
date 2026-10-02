"""ROM validation, import, and the injection seam that uses a local image."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from smb1_ppo import rom


def ines_image(
    prg_kib: int = 32,
    chr_kib: int = 8,
    mapper: int = 0,
    trainer: bool = False,
    pal: bool = False,
    magic: bytes = b"NES\x1a",
    declared_extra: int = 0,
    fill: int = 0,
) -> bytes:
    """Build a structurally valid SMB1-shaped iNES image."""
    header = bytearray(16)
    header[0:4] = magic
    header[4] = prg_kib // 16
    header[5] = chr_kib // 8
    flags_six = (mapper & 0x0F) << 4
    if trainer:
        flags_six |= 0b0000_0100
    header[6] = flags_six
    header[7] = mapper & 0xF0
    header[9] = 1 if pal else 0
    body = bytes(16 + (prg_kib + chr_kib) * 1024 + (512 if trainer else 0) - 16 + declared_extra)
    return bytes(header) + bytes([fill]) * len(body)


class InspectionTests(unittest.TestCase):
    def write(self, directory: str, data: bytes, name: str = "rom.nes") -> Path:
        path = Path(directory) / name
        path.write_bytes(data)
        return path

    def test_a_valid_image_is_smb1_shaped_but_unrecognized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, ines_image())
            inspection = rom.inspect(path)
            self.assertEqual(inspection.verdict, "unrecognized")
            self.assertIsNone(inspection.dump_label)
            self.assertTrue(inspection.is_usable)
            self.assertFalse(inspection.is_known_dump)
            self.assertEqual(
                (inspection.mapper, inspection.prg_kilobytes, inspection.chr_kilobytes), (0, 32, 8)
            )
            self.assertEqual(inspection.size_bytes, 16 + 40 * 1024)
            self.assertEqual(len(inspection.sha256), 64)
            self.assertEqual(len(inspection.md5), 32)

    def test_a_matching_known_dump_is_reported_as_known_good(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = ines_image(fill=7)
            path = self.write(directory, data)
            known = rom.Smb1Dump(
                label="Super Mario Bros. (World)",
                sha256=rom.fingerprint(data)["sha256"],
                md5="0" * 32,
                sha1="0" * 40,
                crc32="00000000",
            )
            with mock.patch.object(rom, "KNOWN_DUMPS", (known,)):
                inspection = rom.inspect(path)
            self.assertEqual(inspection.verdict, "known-good")
            self.assertEqual(inspection.dump_label, "Super Mario Bros. (World)")
            self.assertTrue(inspection.is_known_dump)

    def test_the_canonical_world_dump_fingerprints_are_recorded(self) -> None:
        canonical = rom.KNOWN_DUMPS[0]
        self.assertEqual(canonical.sha256, "f61548fdf1670cffefcc4f0b7bdcdd9eaba0c226e3b74f8666071496988248de")
        self.assertEqual(canonical.md5, "811b027eaf99c2def7b933c5208636de")
        self.assertEqual(canonical.crc32, "3337ec46")

    def test_the_recognized_dumps_are_distinct_and_identify_real_releases(self) -> None:
        checksums = [dump.sha256 for dump in rom.KNOWN_DUMPS]
        self.assertEqual(len(checksums), len(set(checksums)))
        self.assertGreaterEqual(len(rom.KNOWN_DUMPS), 2)
        # The same ROM data behind a different iNES header; both are documented as
        # Super Mario Bros. (World) with headerless CRC32 D445F698.
        alternate = rom.KNOWN_DUMPS[1]
        self.assertEqual(alternate.sha256, "0b3d9e1f01ed1668205bab34d6c82b0e281456e137352e4f36a9b2cfa3b66dea")
        self.assertEqual(alternate.md5, "f94bb9bb55f325d9af8a0fff80b9376d")
        self.assertEqual(alternate.crc32, "393a432f")
        for dump in rom.KNOWN_DUMPS:
            for value in (dump.sha256, dump.md5, dump.sha1, dump.crc32):
                self.assertRegex(value, r"^[0-9a-f]+$")

    def test_malformed_images_are_rejected_with_a_reason(self) -> None:
        cases = {
            "not a NES file": ines_image(magic=b"ZIP\x1a"),
            "mapper": ines_image(mapper=1),
            "prg size": ines_image(prg_kib=16),
            "chr size": ines_image(chr_kib=0),
            "trainer": ines_image(trainer=True),
            "pal": ines_image(pal=True),
            "length": ines_image(declared_extra=32),
        }
        with tempfile.TemporaryDirectory() as directory:
            for expected, data in cases.items():
                path = self.write(directory, data, name=f"{expected.replace(' ', '-')}.nes")
                with self.assertRaises(rom.RomError) as context:
                    rom.inspect(path)
                self.assertTrue(str(context.exception), f"{expected} should be rejected")

    def test_a_tiny_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, b"NES\x1a")
            with self.assertRaises(rom.RomError):
                rom.inspect(path)

    def test_a_missing_file_is_reported_as_missing(self) -> None:
        with self.assertRaises(rom.RomError):
            rom.inspect(Path("/nonexistent/super-mario-bros.nes"))


class ImportTests(unittest.TestCase):
    def test_an_unverifiable_image_needs_an_explicit_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.nes"
            source.write_bytes(ines_image())
            destination = Path(directory) / "roms" / "super-mario-bros.nes"
            with self.assertRaises(rom.RomError) as context:
                rom.import_rom(source, destination)
            self.assertIn("--allow-unverified", str(context.exception))
            self.assertFalse(destination.exists())

    def test_an_unverifiable_image_is_copied_with_a_metadata_sidecar_when_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.nes"
            data = ines_image(fill=3)
            source.write_bytes(data)
            destination = Path(directory) / "roms" / "super-mario-bros.nes"
            inspection = rom.import_rom(source, destination, allow_unverified=True)
            self.assertTrue(destination.is_file())
            self.assertEqual(destination.read_bytes(), data)
            self.assertEqual(inspection.sha256, rom.fingerprint(data)["sha256"])
            sidecar = destination.parent / "rom-import.json"
            self.assertTrue(sidecar.is_file())
            payload = sidecar.read_text(encoding="utf-8")
            for key in ('"verdict"', '"sha256"', '"header"', '"imported_at"'):
                self.assertIn(key, payload)

    def test_an_existing_destination_is_not_overwritten_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.nes"
            source.write_bytes(ines_image(fill=1))
            destination = Path(directory) / "roms" / "super-mario-bros.nes"
            rom.import_rom(source, destination, allow_unverified=True)
            with self.assertRaises(rom.RomError) as context:
                rom.import_rom(source, destination, allow_unverified=True)
            self.assertIn("--force", str(context.exception))
            rom.import_rom(source, destination, allow_unverified=True, force=True)
            self.assertTrue(destination.is_file())

    def test_importing_onto_itself_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "rom.nes"
            source.write_bytes(ines_image())
            with self.assertRaises(rom.RomError):
                rom.import_rom(source, source, allow_unverified=True)


class ResolutionTests(unittest.TestCase):
    def test_the_environment_variable_overrides_the_default_path(self) -> None:
        with mock.patch.dict("os.environ", {rom.ROM_ENVIRONMENT_VARIABLE: "/tmp/custom.nes"}):
            self.assertEqual(rom.default_rom_path(), Path("/tmp/custom.nes"))
            self.assertEqual(rom.resolve_rom(None), Path("/tmp/custom.nes"))

    def test_an_explicit_path_wins_over_the_default(self) -> None:
        with mock.patch.dict("os.environ", {rom.ROM_ENVIRONMENT_VARIABLE: "/tmp/custom.nes"}):
            self.assertEqual(rom.resolve_rom("/tmp/other.nes"), Path("/tmp/other.nes"))

    def test_the_default_path_is_not_inside_the_package(self) -> None:
        default = rom.resolve_rom(None)
        self.assertIn("roms", default.parts)
        self.assertNotIn("site-packages", str(default))

    def test_a_missing_rom_explains_exactly_how_to_supply_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(rom.RomError) as context:
                rom.require_rom(Path(directory) / "absent.nes")
            message = str(context.exception)
            self.assertIn("python -m smb1_ppo.rom import", message)
            self.assertIn(rom.ROM_ENVIRONMENT_VARIABLE, message)


class LocalRomSeamTests(unittest.TestCase):
    """gym-super-mario-bros hard-codes its ROM lookup; the seam must redirect it."""

    def test_both_layouts_are_recognized(self) -> None:
        self.assertEqual(rom.rom_seam_name(["rom_path", "SuperMarioBrosEnv"]), "rom_path")
        self.assertEqual(rom.rom_seam_name(["smb1_rom_path", "rom_path"]), "smb1_rom_path")
        self.assertIsNone(rom.rom_seam_name(["SuperMarioBrosEnv"]))

    def test_the_installed_layout_is_recognized(self) -> None:
        self.assertIn(rom.rom_seam_name(), ("smb1_rom_path", "rom_path"))

    def test_the_seam_redirects_the_upstream_lookup(self) -> None:
        from gym_super_mario_bros import smb_env

        seam = rom.rom_seam_name()
        original = getattr(smb_env, seam)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mine.nes"
            path.write_bytes(ines_image())
            try:
                rom.use_local_rom(path)
                seam_function = getattr(smb_env, seam)
                resolved = seam_function(False, "vanilla") if seam == "rom_path" else seam_function()
                self.assertEqual(resolved, str(path.resolve()))
            finally:
                setattr(smb_env, seam, original)

    def test_unsupported_images_are_refused_at_the_seam(self) -> None:
        from gym_super_mario_bros import smb_env

        seam = rom.rom_seam_name()
        restored = {
            name: getattr(smb_env, name)
            for name in ("rom_path", "smb1_rom_path", "smb2jp_rom_path")
            if hasattr(smb_env, name)
        }
        try:
            rom.use_local_rom("/tmp/any.nes")
            if seam == "smb1_rom_path":
                with self.assertRaises(rom.RomError):
                    smb_env.smb2jp_rom_path()
            else:
                with self.assertRaises(rom.RomError):
                    smb_env.rom_path(True, "vanilla")
                for mode in ("pixel", "rectangle", "downsample"):
                    with self.assertRaises(rom.RomError):
                        smb_env.rom_path(False, mode)
        finally:
            for name, function in restored.items():
                setattr(smb_env, name, function)


if __name__ == "__main__":
    unittest.main()
