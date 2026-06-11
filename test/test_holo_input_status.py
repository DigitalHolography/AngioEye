from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

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
            list_path = root / "list.txt"
            list_path.write_text(f"{first}\n\n{second}\n", encoding="utf-8")

            parsed = read_holo_path_list(list_path)

            self.assertEqual(root, parsed.root_dir)
            self.assertEqual(("20260906_GOA", "20260907_GOB"), parsed.stems)
            self.assertEqual((first, second), parsed.holo_paths)

    def test_holo_path_list_rejects_mixed_root_dirs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            other = root / "other"
            other.mkdir()
            list_path = root / "list.txt"
            list_path.write_text(
                f"{root / '20260906_GOA.holo'}\n"
                f"{other / '20260907_GOB.holo'}\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "same parent folder"):
                read_holo_path_list(list_path)


if __name__ == "__main__":
    unittest.main()
