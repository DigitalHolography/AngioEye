import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cli  # noqa: E402
from workflows import RunWorkflowResult, WorkflowDispatchResult  # noqa: E402


class CliWorkflowRequestTests(unittest.TestCase):
    def _dispatch_result(self, tmp_path: Path) -> WorkflowDispatchResult:
        return WorkflowDispatchResult(
            workflow_result=RunWorkflowResult(
                output_dir=tmp_path / "outputs",
                processed_outputs=[tmp_path / "outputs" / "sample.h5"],
                failures=[],
                summary_message="done",
            )
        )

    def test_doppler_manager_style_call_builds_file_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_h5 = tmp_path / "sample_EF.h5"
            input_h5.write_text("h5", encoding="utf-8")
            pipelines_file = tmp_path / "pipelines.txt"
            pipelines_file.write_text("waveform_shape_metrics\n", encoding="utf-8")
            output_dir = tmp_path / "outputs"
            pipeline = SimpleNamespace(name="waveform_shape_metrics")
            captured_requests = []

            def _dispatch(request, _callbacks):
                captured_requests.append(request)
                return self._dispatch_result(tmp_path)

            with (
                mock.patch.object(
                    cli,
                    "_build_pipeline_registry",
                    return_value={"waveform_shape_metrics": pipeline},
                ),
                mock.patch.object(cli, "_build_postprocess_registry", return_value={}),
                mock.patch.object(cli, "dispatch_workflow", side_effect=_dispatch),
            ):
                result = cli.main(
                    [
                        "--data",
                        str(input_h5),
                        "--pipelines",
                        str(pipelines_file),
                        "--output",
                        str(output_dir),
                        "--trim-source",
                    ]
                )

            self.assertEqual(0, result)
            request = captured_requests[0]
            self.assertEqual("file", request.mode)
            self.assertEqual((input_h5,), request.input_plan.h5_paths)
            self.assertEqual((pipeline,), tuple(request.pipelines))
            self.assertEqual(
                ("waveform_shape_metrics",),
                tuple(request.selected_pipeline_names),
            )
            self.assertTrue(request.trim_source)

    def test_holo_path_list_builds_holo_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            holo_list = tmp_path / "missing_AE.txt"
            holo_list.write_text("sample.holo\n", encoding="utf-8")
            pipelines_file = tmp_path / "pipelines.txt"
            pipelines_file.write_text("waveform_shape_metrics\n", encoding="utf-8")
            output_dir = tmp_path / "outputs"
            pipeline = SimpleNamespace(name="waveform_shape_metrics")
            captured_requests = []

            def _dispatch(request, _callbacks):
                captured_requests.append(request)
                return self._dispatch_result(tmp_path)

            with (
                mock.patch.object(
                    cli,
                    "_build_pipeline_registry",
                    return_value={"waveform_shape_metrics": pipeline},
                ),
                mock.patch.object(cli, "_build_postprocess_registry", return_value={}),
                mock.patch.object(cli, "dispatch_workflow", side_effect=_dispatch),
            ):
                result = cli.main(
                    [
                        "--data",
                        str(holo_list),
                        "--pipelines",
                        str(pipelines_file),
                        "--output",
                        str(output_dir),
                    ]
                )

            self.assertEqual(0, result)
            request = captured_requests[0]
            self.assertEqual("holo", request.mode)
            self.assertIsNone(request.input_plan)
            self.assertEqual((holo_list,), tuple(request.holo_paths))

    def test_inline_pipeline_name_does_not_require_pipeline_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_h5 = tmp_path / "sample_EF.h5"
            input_h5.write_text("h5", encoding="utf-8")
            output_dir = tmp_path / "outputs"
            pipeline = SimpleNamespace(name="waveform_shape_metrics")

            with (
                mock.patch.object(
                    cli,
                    "_build_pipeline_registry",
                    return_value={"waveform_shape_metrics": pipeline},
                ),
                mock.patch.object(cli, "_build_postprocess_registry", return_value={}),
                mock.patch.object(
                    cli,
                    "dispatch_workflow",
                    return_value=self._dispatch_result(tmp_path),
                ),
            ):
                result = cli.main(
                    [
                        "--data",
                        str(input_h5),
                        "--pipeline",
                        "waveform_shape_metrics",
                        "--output",
                        str(output_dir),
                    ]
                )

            self.assertEqual(0, result)


if __name__ == "__main__":
    unittest.main()
