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

    def test_cli_log_flushes_terminal_output_immediately(self) -> None:
        with mock.patch.object(cli, "print") as printer:
            cli._log_cli("[PROCESS] Pipeline finished")

        printer.assert_called_once_with("[PROCESS] Pipeline finished", flush=True)

    def test_cli_warning_log_flushes_stderr_immediately(self) -> None:
        with mock.patch.object(cli, "print") as printer:
            cli._log_cli("[POST WARN] optional report skipped")

        printer.assert_called_once_with(
            "[POST WARN] optional report skipped",
            file=sys.stderr,
            flush=True,
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
            self.assertFalse(request.persist_source)

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
                        "--pipelines",
                        "waveform_shape_metrics",
                        "--output",
                        str(output_dir),
                    ]
                )

            self.assertEqual(0, result)

    def test_repeated_selection_names_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_h5 = tmp_path / "sample_EF.h5"
            input_h5.write_text("h5", encoding="utf-8")
            first = SimpleNamespace(name="waveform_shape_metrics")
            second = SimpleNamespace(name="second_pipeline")
            postprocess = SimpleNamespace(name="HTML summary")
            captured_requests = []

            def _dispatch(request, _callbacks):
                captured_requests.append(request)
                return self._dispatch_result(tmp_path)

            with (
                mock.patch.object(
                    cli,
                    "_build_pipeline_registry",
                    return_value={first.name: first, second.name: second},
                ),
                mock.patch.object(
                    cli,
                    "_build_postprocess_registry",
                    return_value={postprocess.name: postprocess},
                ),
                mock.patch.object(cli, "dispatch_workflow", side_effect=_dispatch),
            ):
                result = cli.main(
                    [
                        "--data",
                        str(input_h5),
                        "--pipelines",
                        first.name,
                        second.name,
                        "--postprocesses",
                        postprocess.name,
                    ]
                )

            self.assertEqual(0, result)
            request = captured_requests[0]
            self.assertEqual((first, second), tuple(request.pipelines))
            self.assertEqual((postprocess,), tuple(request.postprocesses))

    def test_list_literal_selection_names_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_h5 = tmp_path / "sample_EF.h5"
            input_h5.write_text("h5", encoding="utf-8")
            pipeline = SimpleNamespace(name="waveform_shape_metrics")
            second_pipeline = SimpleNamespace(name="second_pipeline")
            postprocess = SimpleNamespace(name="HTML summary")
            captured_requests = []

            def _dispatch(request, _callbacks):
                captured_requests.append(request)
                return self._dispatch_result(tmp_path)

            with (
                mock.patch.object(
                    cli,
                    "_build_pipeline_registry",
                    return_value={
                        pipeline.name: pipeline,
                        second_pipeline.name: second_pipeline,
                    },
                ),
                mock.patch.object(
                    cli,
                    "_build_postprocess_registry",
                    return_value={postprocess.name: postprocess},
                ),
                mock.patch.object(cli, "dispatch_workflow", side_effect=_dispatch),
            ):
                result = cli.main(
                    [
                        "--data",
                        str(input_h5),
                        "--pipelines",
                        '["waveform_shape_metrics", "second_pipeline"]',
                        "--postprocesses",
                        '["HTML summary"]',
                    ]
                )

            self.assertEqual(0, result)
            self.assertEqual(
                (pipeline, second_pipeline),
                tuple(captured_requests[0].pipelines),
            )
            self.assertEqual((postprocess,), tuple(captured_requests[0].postprocesses))

    def test_missing_postprocess_selection_runs_no_postprocesses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_h5 = tmp_path / "sample_EF.h5"
            input_h5.write_text("h5", encoding="utf-8")
            pipeline = SimpleNamespace(name="waveform_shape_metrics")
            postprocess = SimpleNamespace(name="HTML summary")
            captured_requests = []

            class _Settings:
                def load_pipeline_visibility(self):
                    return {"waveform_shape_metrics": True}

            def _dispatch(request, _callbacks):
                captured_requests.append(request)
                return self._dispatch_result(tmp_path)

            with (
                mock.patch.object(
                    cli,
                    "_build_pipeline_registry",
                    return_value={pipeline.name: pipeline},
                ),
                mock.patch.object(
                    cli,
                    "_build_postprocess_registry",
                    return_value={postprocess.name: postprocess},
                ),
                mock.patch.object(cli, "AppSettingsStore", return_value=_Settings()),
                mock.patch.object(cli, "dispatch_workflow", side_effect=_dispatch),
            ):
                result = cli.main(["--data", str(input_h5)])
                keep_result = cli.main(
                    ["--data", str(input_h5), "--keep-source"]
                )

            self.assertEqual(0, result)
            self.assertEqual(0, keep_result)
            request = captured_requests[0]
            self.assertEqual(tmp_path, request.base_output_dir)
            self.assertEqual((pipeline,), tuple(request.pipelines))
            self.assertEqual((), tuple(request.postprocesses))
            self.assertFalse(request.persist_source)
            self.assertTrue(captured_requests[-1].persist_source)

    def test_modern_ef_h5_uses_ae_output_tree_and_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            holo_path = tmp_path / "sample.holo"
            input_h5 = tmp_path / "sample" / "sample_EF" / "h5" / "sample.h5"
            holo_path.write_text("holo", encoding="utf-8")
            input_h5.parent.mkdir(parents=True)
            input_h5.write_text("h5", encoding="utf-8")
            pipelines_file = tmp_path / "pipelines.txt"
            pipelines_file.write_text("waveform_shape_metrics\n", encoding="utf-8")
            pipeline = SimpleNamespace(name="waveform_shape_metrics")
            captured_requests = []

            def _dispatch(request, _callbacks):
                captured_requests.append(request)
                return self._dispatch_result(tmp_path)

            with (
                mock.patch.object(
                    cli,
                    "_build_pipeline_registry",
                    return_value={pipeline.name: pipeline},
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
                    ]
                )

            self.assertEqual(0, result)
            request = captured_requests[0]
            self.assertEqual(
                tmp_path / "sample" / "sample_AE",
                request.base_output_dir,
            )
            self.assertEqual(
                "sample_AE.h5",
                request.output_filename_for_run(input_h5, (input_h5,)),
            )

    def test_required_postprocess_option_enables_source_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_h5 = tmp_path / "sample_EF.h5"
            input_h5.write_text("h5", encoding="utf-8")
            pipeline = SimpleNamespace(name="waveform_shape_metrics")
            postprocess = SimpleNamespace(
                name="HTML summary",
                required_option=["persist_eyeflow_data"],
            )
            captured_requests = []

            def _dispatch(request, _callbacks):
                captured_requests.append(request)
                return self._dispatch_result(tmp_path)

            with (
                mock.patch.object(
                    cli,
                    "_build_pipeline_registry",
                    return_value={pipeline.name: pipeline},
                ),
                mock.patch.object(
                    cli,
                    "_build_postprocess_registry",
                    return_value={postprocess.name: postprocess},
                ),
                mock.patch.object(cli, "dispatch_workflow", side_effect=_dispatch),
            ):
                result = cli.main(
                    [
                        "--data",
                        str(input_h5),
                        "--pipelines",
                        pipeline.name,
                        "--postprocesses",
                        postprocess.name,
                    ]
                )

            self.assertEqual(0, result)
            self.assertTrue(captured_requests[0].persist_source)


if __name__ == "__main__":
    unittest.main()
