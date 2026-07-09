from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from input_output import (
    found_status_text,
    holo_input_status,
    read_holo_path_list,
    stem_input_status,
)


class HoloInputStatusTests(unittest.TestCase):
    def test_holo_status_checks_direct_ef_h5_files(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            holo_path = root / "sample.holo"
            holo_path.write_text("", encoding="utf-8")
            ef_dir = root / "sample" / "sample_EF"
            ef_dir.mkdir(parents=True)

            self.assertFalse(
                holo_input_status(holo_path, require_holo_file=True).ef
            )

            (ef_dir / "sample.h5").write_text("", encoding="utf-8")

            self.assertTrue(
                holo_input_status(holo_path, require_holo_file=True).ef
            )
            self.assertTrue(stem_input_status("sample", root).ef)

    def test_holo_status_checks_nested_ef_h5_files(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            holo_path = root / "sample.holo"
            holo_path.write_text("", encoding="utf-8")
            ef_h5_dir = root / "sample" / "sample_EF" / "h5"
            ef_h5_dir.mkdir(parents=True)
            (ef_h5_dir / "sample_EF.h5").write_text("", encoding="utf-8")

            self.assertTrue(
                holo_input_status(holo_path, require_holo_file=True).ef
            )
            self.assertTrue(stem_input_status("sample", root).ef)

    def test_found_status_text_matches_eyeflow_shape(self) -> None:
        self.assertEqual(found_status_text("EF", 1, 1, []), "EF found")
        self.assertEqual(found_status_text("EF", 0, 1, ["a"]), "EF not found")
        self.assertEqual(
            found_status_text("EF", 1, 2, ["b"]),
            "EF 1/2 found: missing b",
        )

    def test_holo_path_list_returns_root_dir_and_stems(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "20260906_GOA.holo"
            second = root / "20260907_GOB.holo"
            first.write_text("", encoding="utf-8")
            second.write_text("", encoding="utf-8")
            list_path = root / "list.txt"
            list_path.write_text(f"{first}\n\n{second}\n", encoding="utf-8")

            parsed = read_holo_path_list(list_path)

            self.assertEqual(root, parsed.root_dir)
            self.assertEqual(("20260906_GOA", "20260907_GOB"), parsed.stems)
            self.assertEqual((first, second), parsed.holo_paths)

    def test_holo_path_list_accepts_paths_from_different_parent_dirs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            other = root / "other"
            other.mkdir()
            first = root / "20260906_GOA.holo"
            second = other / "20260907_GOB.holo"
            first.write_text("", encoding="utf-8")
            second.write_text("", encoding="utf-8")
            list_path = root / "list.txt"
            list_path.write_text(
                f"{first}\n{second}\n",
                encoding="utf-8",
            )

            parsed = read_holo_path_list(list_path)

            self.assertEqual((first, second), parsed.holo_paths)
            self.assertEqual((), parsed.warnings)

    def test_holo_path_list_warns_and_skips_invalid_entries(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid = root / "valid.holo"
            valid.write_text("", encoding="utf-8")
            list_path = root / "list.txt"
            list_path.write_text(
                "not-a-holo.txt\nmissing.holo\nvalid.holo\n",
                encoding="utf-8",
            )

            parsed = read_holo_path_list(list_path)

            self.assertEqual((valid,), parsed.holo_paths)
            self.assertEqual(2, len(parsed.warnings))
            self.assertIn("not a .holo file", parsed.warnings[0])
            self.assertIn("does not exist", parsed.warnings[1])

    @unittest.skipUnless(Path("C:/").anchor, "Windows path semantics required")
    def test_holo_path_list_reports_unavailable_mapped_drive(self) -> None:
        with TemporaryDirectory() as tmp:
            list_path = Path(tmp) / "list.txt"
            list_path.write_text("X:/share/sample.holo\n", encoding="utf-8")
            original_is_file = Path.is_file
            original_exists = Path.exists

            def is_file(path: Path) -> bool:
                return (
                    original_is_file(path)
                    if path == list_path
                    else False
                )

            def exists(path: Path) -> bool:
                return False if path == Path("X:/") else original_exists(path)

            with (
                mock.patch.object(Path, "is_file", is_file),
                mock.patch.object(Path, "exists", exists),
            ):
                parsed = read_holo_path_list(list_path)

            self.assertEqual((), parsed.holo_paths)
            self.assertIn("path root is unavailable", parsed.warnings[0])
            self.assertIn("UNC path", parsed.warnings[0])

    def test_holo_path_list_warns_and_continues_when_path_cannot_expand(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid = root / "valid.holo"
            valid.write_text("", encoding="utf-8")
            list_path = root / "list.txt"
            list_path.write_text("~/broken.holo\nvalid.holo\n", encoding="utf-8")
            path_type = type(list_path)
            original_expanduser = path_type.expanduser

            def expanduser(path: Path) -> Path:
                if path.parts and path.parts[0] == "~":
                    raise RuntimeError("home directory cannot be resolved")
                return original_expanduser(path)

            with mock.patch.object(path_type, "expanduser", expanduser):
                parsed = read_holo_path_list(list_path)

            self.assertEqual((valid,), parsed.holo_paths)
            self.assertEqual(1, len(parsed.warnings))
            self.assertIn("could not expand listed path", parsed.warnings[0])


if __name__ == "__main__":
    unittest.main()
