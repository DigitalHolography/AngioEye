import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from input_output import (  # noqa: E402
    acquisition_output_dir,
    ae_result_filename,
    h5_output_parent,
)
from input_output.output_paths import companion_output_dir  # noqa: E402


class OutputPathTests(unittest.TestCase):
    def test_acquisition_output_defaults_to_holo_parent(self) -> None:
        source = Path("data") / "sample.holo"
        self.assertEqual(
            Path("data") / "sample" / "sample_AE",
            acquisition_output_dir(source),
        )

    def test_acquisition_output_preserves_batch_group_parent(self) -> None:
        self.assertEqual(
            Path("results") / "control" / "sample" / "sample_AE",
            acquisition_output_dir("data/sample.holo", "results", "control"),
        )

    def test_acquisition_output_keeps_holo_stems_ending_in_app_suffix(self) -> None:
        for stem in ("sample_AE", "sample_EF"):
            with self.subTest(stem=stem):
                self.assertEqual(
                    Path("results") / stem / f"{stem}_AE",
                    acquisition_output_dir(f"{stem}.holo", "results"),
                )

    def test_ef_names_use_ae_artifact_suffix(self) -> None:
        self.assertEqual("sample_AE.h5", ae_result_filename(Path("sample_EF.h5")))

    def test_plain_h5_names_keep_the_acquisition_identifier(self) -> None:
        self.assertEqual("sample_AE.h5", ae_result_filename(Path("sample.h5")))

    def test_batch_output_keeps_group_parent_below_h5(self) -> None:
        output = Path("results")
        self.assertEqual(output / "h5" / "control", h5_output_parent(output, "control"))

    def test_source_companion_folder_resolves_within_acquisition_tree(self) -> None:
        source = Path("data") / "sample" / "sample_EF" / "h5" / "sample_EF.h5"
        self.assertEqual(
            Path("data") / "sample" / "sample_DV" / "png",
            companion_output_dir(source, app_suffix="DV", query_type="png"),
        )

    def test_generated_companion_folder_resolves_within_acquisition_tree(self) -> None:
        source = (
            Path("results") / "sample_EF" / "sample_EF_AE" / "h5" / "sample_EF_AE.h5"
        )
        self.assertEqual(
            Path("results") / "sample_EF" / "sample_EF_AE" / "html",
            companion_output_dir(source, app_suffix="AE", query_type="html"),
        )


if __name__ == "__main__":
    unittest.main()
