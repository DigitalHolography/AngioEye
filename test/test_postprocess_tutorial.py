# ruff: noqa: E402

import json
import sys
import tempfile
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from postprocess.core.base import PostprocessContext
from postprocess.tutorial_postprocess import PostprocessTutorial


class PostprocessTutorialTests(unittest.TestCase):
    def test_tutorial_generates_minimal_json_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_dir = tmp_path / "outputs"
            output_dir.mkdir()
            result_file = output_dir / "sample_result.h5"
            result_file.write_text("placeholder", encoding="utf-8")

            context = PostprocessContext(
                output_dir=output_dir,
                processed_files=(result_file,),
                selected_pipelines=("Basic Stats",),
                input_path=tmp_path / "input_folder",
                zip_outputs=False,
                input_h5_paths=(tmp_path / "input_folder" / "sample.h5",),
            )

            result = PostprocessTutorial().run(context)

            json_path = output_dir / "postprocess_tutorial.json"

            self.assertEqual(
                [str(json_path)],
                result.generated_paths,
            )
            self.assertEqual(1, result.metadata["processed_file_count"])
            self.assertTrue(json_path.exists())
            self.assertEqual(
                ["Basic Stats"],
                result.metadata["selected_pipelines"],
            )

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual("Postprocess Tutorial", payload["postprocess_name"])
            self.assertEqual(
                str(output_dir),
                payload["context_fields"]["output_dir"],
            )
            self.assertEqual(
                [str(result_file)],
                payload["context_fields"]["processed_files"],
            )
            self.assertEqual(
                [str(tmp_path / "input_folder" / "sample.h5")],
                payload["context_fields"]["input_h5_paths"],
            )
            self.assertEqual(
                ["Basic Stats"],
                payload["context_fields"]["selected_pipelines"],
            )
            self.assertEqual(
                str(tmp_path / "input_folder"),
                payload["context_fields"]["input_path"],
            )
            self.assertFalse(payload["context_fields"]["zip_outputs"])
            self.assertEqual(
                "Generated postprocess_tutorial.json.",
                payload["result_format"]["summary"],
            )
            self.assertEqual(
                [str(json_path)],
                payload["result_format"]["generated_paths"],
            )
            self.assertEqual(
                {
                    "processed_file_count": 1,
                    "selected_pipelines": ["Basic Stats"],
                },
                payload["result_format"]["metadata"],
            )


if __name__ == "__main__":
    unittest.main()
