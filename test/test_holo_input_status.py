from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from input_output import found_status_text, holo_input_status, stem_input_status


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

    def test_found_status_text_matches_eyeflow_shape(self) -> None:
        self.assertEqual(found_status_text("EF", 1, 1, []), "EF found")
        self.assertEqual(found_status_text("EF", 0, 1, ["a"]), "EF not found")
        self.assertEqual(
            found_status_text("EF", 1, 2, ["b"]),
            "EF 1/2 found: missing b",
        )


if __name__ == "__main__":
    unittest.main()
