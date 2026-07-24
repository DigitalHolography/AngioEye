import sys
import tempfile
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from input_output import (  # noqa: E402
    default_h5_output_dir,
    default_h5_output_filename,
    default_output_filename_for_run,
)
from ui.controllers.run import RunTabController  # noqa: E402


class OutputPathTests(unittest.TestCase):
    def test_modern_ef_input_resolves_to_sibling_ae_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            parent = Path(tmp_dir)
            holo_path = parent / "sample.holo"
            h5_path = parent / "sample" / "sample_EF" / "h5" / "sample.h5"
            holo_path.write_text("holo", encoding="utf-8")
            h5_path.parent.mkdir(parents=True)
            h5_path.write_text("h5", encoding="utf-8")

            self.assertEqual(
                parent / "sample" / "sample_AE",
                default_h5_output_dir(h5_path),
            )
            self.assertEqual("sample_AE.h5", default_h5_output_filename(h5_path))
            self.assertEqual(
                "sample_AE.h5",
                default_output_filename_for_run(h5_path, (h5_path,)),
            )

    def test_h5_outside_app_tree_uses_legacy_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            parent = Path(tmp_dir)
            h5_path = parent / "inputs" / "sample.h5"
            h5_path.parent.mkdir(parents=True)
            h5_path.write_text("h5", encoding="utf-8")

            self.assertEqual(h5_path.parent, default_h5_output_dir(h5_path))

    def test_gui_and_cli_single_file_names_use_the_same_ae_rule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            h5_path = Path(tmp_dir) / "sample_EF.h5"
            h5_path.write_text("h5", encoding="utf-8")

            gui_name = RunTabController.default_output_artifact_name(
                object(),
                h5_path,
            )
            cli_name = default_output_filename_for_run(h5_path, (h5_path,))

            self.assertEqual("sample_AE.h5", gui_name)
            self.assertEqual(gui_name, cli_name)


if __name__ == "__main__":
    unittest.main()
