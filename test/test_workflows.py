import sys
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path
from unittest import mock

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from input_output import list_h5_members  # noqa: E402
from workflows import (  # noqa: E402
    HoloInputContext,
    RunWorkflowResult,
    WorkflowCallbacks,
    WorkflowInputError,
    WorkflowRunRequest,
    ZipBatchSettings,
    dispatch_workflow,
    prepare_run_input,
    prepare_run_inputs,
    run_filesystem_workflow,
    run_holo_workflow,
)
from workflows._pipeline_runs import (  # noqa: E402
    run_filesystem_pipeline_run,
    run_zip_pipeline_run,
)
from workflows._postprocess_requirements import (  # noqa: E402
    compatible_postprocess_files,
    missing_required_pipeline_errors,
)


def process_pool_run_pipeline_file(
    h5_path,
    _pipelines,
    output_root,
    output_relative_parent=Path("."),
    output_filename=None,
):
    target_dir = output_root / output_relative_parent
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / (output_filename or f"{h5_path.stem}_result.h5")
    output_path.write_text("result", encoding="utf-8")
    return output_path


class PostprocessRequirementTests(unittest.TestCase):
    def test_selected_required_pipeline_keeps_processed_outputs(self):
        processed_output = Path("new_result.h5")
        input_h5 = Path("input.h5")

        result = compatible_postprocess_files(
            processed_outputs=(processed_output,),
            input_h5_paths=(input_h5,),
            required_pipelines=("waveform_shape_metrics",),
            selected_pipeline_names=("waveform_shape_metrics",),
        )

        self.assertEqual((processed_output,), result.files)
        self.assertEqual((), result.skipped)

    def test_unselected_required_pipeline_still_skips_incompatible_files(self):
        processed_output = Path("new_result.h5")
        input_h5 = Path("input.h5")

        result = compatible_postprocess_files(
            processed_outputs=(processed_output,),
            input_h5_paths=(input_h5,),
            required_pipelines=("waveform_shape_metrics",),
            selected_pipeline_names=(),
        )

        self.assertEqual((), result.files)
        self.assertEqual((input_h5,), result.skipped)

    def test_selected_alternative_required_pipeline_keeps_processed_outputs(self):
        processed_output = Path("new_result.h5")
        input_h5 = Path("input.h5")

        result = compatible_postprocess_files(
            processed_outputs=(processed_output,),
            input_h5_paths=(input_h5,),
            required_pipelines=(
                "waveform_shape_metrics",
                "waveform_shape_metrics_denoised",
            ),
            required_pipeline_options=(
                ("waveform_shape_metrics",),
                ("waveform_shape_metrics_denoised",),
            ),
            selected_pipeline_names=("waveform_shape_metrics",),
        )

        self.assertEqual((processed_output,), result.files)
        self.assertEqual((), result.skipped)

    def test_alternative_required_pipeline_errors_accept_either_selection(self):
        postprocess = type(
            "Postprocess",
            (),
            {
                "name": "Variability",
                "required_pipeline_options": (
                    ("waveform_shape_metrics",),
                    ("waveform_shape_metrics_denoised",),
                ),
            },
        )()

        errors = missing_required_pipeline_errors(
            postprocesses=(postprocess,),
            selected_pipeline_names=("waveform_shape_metrics",),
        )

        self.assertEqual([], errors)


class FilesystemWorkflowTests(unittest.TestCase):
    def _run_workflow(
        self,
        tmp_path: Path,
        *,
        zip_should_fail: bool,
        postprocesses=None,
    ):
        logs: list[str] = []
        status: list[str] = []
        progress: list[float] = []
        final_progress: list[tuple[float, str]] = []
        zip_errors: list[str] = []
        zip_source_folders: list[Path] = []

        input_root = tmp_path / "inputs"
        input_root.mkdir()
        h5_path = input_root / "sample.h5"
        h5_path.write_text("h5", encoding="utf-8")
        base_output_dir = tmp_path / "outputs"
        base_output_dir.mkdir()

        def run_pipeline_file(
            _h5_path,
            _pipelines,
            output_dir,
            output_relative_parent=Path("."),
            output_filename=None,
        ):
            target_dir = output_dir / output_relative_parent
            target_dir.mkdir(parents=True, exist_ok=True)
            png_dir = output_dir / "png"
            png_dir.mkdir(parents=True, exist_ok=True)
            (png_dir / "composite_scoring_raw_rwas_by_cohort.png").write_text(
                "png",
                encoding="utf-8",
            )
            output_path = target_dir / (output_filename or "sample_result.h5")
            output_path.write_text("result", encoding="utf-8")
            return output_path

        def zip_output_dir(folder, target_path=None, progress_callback=None):
            zip_source_folders.append(folder)
            if zip_should_fail:
                raise RuntimeError("zip failed")
            if progress_callback is not None:
                progress_callback(1, 1, Path("sample_result.h5"))
            assert target_path is not None
            target_path.write_text("archive", encoding="utf-8")
            return target_path

        result = run_filesystem_workflow(
            inputs=[h5_path],
            data_root=input_root,
            pipelines=[object()],
            postprocesses=list(postprocesses or []),
            selected_pipeline_names=["Demo"],
            input_path=input_root,
            base_output_dir=base_output_dir,
            zip_outputs=True,
            zip_name="outputs.zip",
            output_filename=None,
            settings=ZipBatchSettings(batch_size=4),
            run_pipeline_file=run_pipeline_file,
            run_postprocesses=lambda *args, **kwargs: None,
            relative_parent=lambda *_args: Path("."),
            zip_output_dir=zip_output_dir,
            log=logs.append,
            advance_progress=progress.append,
            start_final_progress=lambda units, text: final_progress.append(
                (units, text)
            ),
            set_status=status.append,
            make_zip_progress_callback=lambda: None,
            on_zip_error=zip_errors.append,
        )

        return (
            base_output_dir,
            result,
            logs,
            status,
            progress,
            final_progress,
            zip_errors,
            zip_source_folders,
        )

    def test_run_filesystem_workflow_removes_work_dir_after_zip_success(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            (
                base_output_dir,
                result,
                logs,
                status,
                progress,
                final_progress,
                zip_errors,
                zip_source_folders,
            ) = self._run_workflow(Path(tmp_dir), zip_should_fail=False)

            self.assertFalse(result.zip_failed)
            self.assertEqual(base_output_dir / "outputs.zip", result.zip_path)
            self.assertTrue((base_output_dir / "outputs.zip").exists())
            self.assertTrue(
                (
                    base_output_dir
                    / "png"
                    / "composite_scoring_raw_rwas_by_cohort.png"
                ).exists()
            )
            self.assertFalse(zip_source_folders[0].exists())
            self.assertIn("ZIP archive:", result.summary_message)
            self.assertIn("Companion outputs:", result.summary_message)
            self.assertIn("Creating ZIP...", status)
            self.assertEqual([(1, "Creating ZIP...")], final_progress)
            self.assertEqual([1], progress)
            self.assertEqual([], zip_errors)
            self.assertTrue(
                any("[ZIP] Archive created:" in message for message in logs)
            )
            self.assertTrue(
                any(
                    message.startswith("[TIME] Pipeline phase completed in ")
                    for message in logs
                )
            )
            self.assertTrue(
                any(
                    message.startswith("[TIME] ZIP finalization completed in ")
                    for message in logs
                )
            )

    def test_run_filesystem_workflow_logs_postprocess_duration(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            (
                _base_output_dir,
                _result,
                logs,
                _status,
                _progress,
                _final_progress,
                _zip_errors,
                _zip_source_folders,
            ) = self._run_workflow(
                Path(tmp_dir),
                zip_should_fail=False,
                postprocesses=[object()],
            )

            self.assertTrue(
                any(
                    message.startswith("[TIME] Postprocess phase completed in ")
                    for message in logs
                )
            )

    def test_run_filesystem_workflow_keeps_work_dir_after_zip_failure(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            (
                base_output_dir,
                result,
                logs,
                _status,
                progress,
                _final_progress,
                zip_errors,
                zip_source_folders,
            ) = self._run_workflow(Path(tmp_dir), zip_should_fail=True)

            self.assertTrue(result.zip_failed)
            self.assertIsNone(result.zip_path)
            self.assertFalse((base_output_dir / "outputs.zip").exists())
            self.assertTrue(zip_source_folders[0].exists())
            self.assertTrue((zip_source_folders[0] / "sample_result.h5").exists())
            self.assertEqual("zip failed", result.zip_error)
            self.assertEqual([1, 1.0], progress)
            self.assertEqual(["zip failed"], zip_errors)
            self.assertTrue(
                any("[ZIP FAIL] zip failed" == message for message in logs)
            )
            self.assertTrue(
                any(
                    message.startswith("[TIME] Pipeline phase completed in ")
                    for message in logs
                )
            )
            self.assertTrue(
                any(
                    message.startswith("[TIME] ZIP finalization completed in ")
                    for message in logs
                )
            )


class ZipPipelineParallelismTests(unittest.TestCase):
    def test_run_filesystem_pipeline_run_uses_process_pool_for_picklable_runner(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_dir = tmp_path / "inputs"
            input_dir.mkdir()
            input_paths = []
            for index in range(4):
                input_path = input_dir / f"sample_{index}.h5"
                input_path.write_text("h5", encoding="utf-8")
                input_paths.append(input_path)
            output_dir = tmp_path / "outputs"
            output_dir.mkdir()
            logs: list[str] = []
            progress: list[float] = []

            result = run_filesystem_pipeline_run(
                inputs=input_paths,
                data_root=input_dir,
                pipelines=["waveform_shape_metrics"],
                output_dir=output_dir,
                output_filename=None,
                settings=ZipBatchSettings(
                    batch_size=2,
                    process_workers=2,
                ),
                run_pipeline_file=process_pool_run_pipeline_file,
                relative_parent=lambda *_args: Path("."),
                log=logs.append,
                advance_progress=progress.append,
            )

            self.assertEqual(4, len(result.processed_outputs))
            self.assertEqual([], result.failures)
            self.assertEqual([1, 1, 1, 1], progress)
            self.assertTrue(
                any(
                    "Starting ProcessPoolExecutor(max_workers=2)" in message
                    for message in logs
                ),
                logs,
            )
            self.assertTrue(
                any(message.startswith("[BATCH OK]") for message in logs),
                logs,
            )

    def test_run_zip_pipeline_run_uses_process_pool_for_picklable_runner(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            zip_path = tmp_path / "inputs.zip"
            output_dir = tmp_path / "outputs"
            output_dir.mkdir()
            with zipfile.ZipFile(zip_path, "w") as archive:
                for index in range(4):
                    archive.writestr(f"CTRL/sample_{index}.h5", "h5")

            members = list_h5_members(zip_path)
            logs: list[str] = []
            progress: list[float] = []

            result = run_zip_pipeline_run(
                zip_path=zip_path,
                members=members,
                member_count=len(members),
                pipelines=["waveform_shape_metrics"],
                output_dir=output_dir,
                settings=ZipBatchSettings(
                    batch_size=2,
                    process_workers=2,
                ),
                run_pipeline_file=process_pool_run_pipeline_file,
                log=logs.append,
                advance_progress=progress.append,
            )

            self.assertEqual(4, len(result.processed_outputs))
            self.assertEqual([], result.failures)
            self.assertEqual([1, 1, 1, 1], progress)
            self.assertTrue(
                any(
                    "Starting ProcessPoolExecutor(max_workers=2)" in message
                    for message in logs
                ),
                logs,
            )
            self.assertTrue(
                any(message.startswith("[ZIP] Streaming 4 file(s)") for message in logs),
                logs,
            )
            self.assertTrue(
                any(message.startswith("[PROCESS] Queued ZIP batch 2/2") for message in logs),
                logs,
            )

    def test_run_holo_workflow_uses_process_pool_for_picklable_runner(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            contexts: list[HoloInputContext] = []
            for index in range(4):
                holo_path = tmp_path / f"sample_{index}.holo"
                holo_path.write_text("holo", encoding="utf-8")
                ef_dir = tmp_path / f"sample_{index}" / f"sample_{index}_EF"
                h5_dir = ef_dir / "h5"
                h5_dir.mkdir(parents=True)
                h5_path = h5_dir / f"sample_{index}.h5"
                h5_path.write_text("h5", encoding="utf-8")
                contexts.append(
                    HoloInputContext(
                        holo_path=holo_path,
                        ef_dir=ef_dir,
                        h5_path=h5_path,
                        output_dir=tmp_path
                        / f"sample_{index}"
                        / f"sample_{index}_AE",
                    )
                )

            logs: list[str] = []
            progress: list[float] = []

            result = run_holo_workflow(
                contexts=contexts,
                pipelines=["waveform_shape_metrics"],
                postprocesses=[],
                selected_pipeline_names=["waveform_shape_metrics"],
                run_pipeline_file=process_pool_run_pipeline_file,
                run_postprocesses=lambda *args, **kwargs: None,
                log=logs.append,
                advance_progress=progress.append,
                start_final_progress=lambda _units, _status: None,
                settings=ZipBatchSettings(
                    batch_size=2,
                    process_workers=2,
                ),
            )

            self.assertEqual(4, len(result.processed_outputs))
            self.assertEqual([], result.failures)
            self.assertEqual([1, 1, 1, 1], progress)
            self.assertTrue(
                any(
                    "Starting ProcessPoolExecutor(max_workers=2)" in message
                    for message in logs
                ),
                logs,
            )
            self.assertTrue(
                any(
                    message.startswith("[PROCESS] Queued holo batch 2/2")
                    for message in logs
                ),
                logs,
            )

    def test_run_filesystem_pipeline_run_parallelizes_files_inside_batch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_dir = tmp_path / "inputs"
            input_dir.mkdir()
            input_paths = []
            for index in range(4):
                input_path = input_dir / f"sample_{index}.h5"
                input_path.write_text("h5", encoding="utf-8")
                input_paths.append(input_path)
            output_dir = tmp_path / "outputs"
            output_dir.mkdir()
            active = 0
            max_active = 0
            progress: list[float] = []
            idle_calls = 0
            lock = threading.Lock()

            def run_pipeline_file(
                h5_path,
                _pipelines,
                output_root,
                output_relative_parent=Path("."),
                output_filename=None,
            ):
                nonlocal active, max_active
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                try:
                    time.sleep(0.12)
                    target_dir = output_root / output_relative_parent
                    target_dir.mkdir(parents=True, exist_ok=True)
                    output_path = target_dir / (
                        output_filename or f"{h5_path.stem}_result.h5"
                    )
                    output_path.write_text("result", encoding="utf-8")
                    return output_path
                finally:
                    with lock:
                        active -= 1

            def _idle_callback():
                nonlocal idle_calls
                idle_calls += 1

            result = run_filesystem_pipeline_run(
                inputs=input_paths,
                data_root=input_dir,
                pipelines=[object()],
                output_dir=output_dir,
                output_filename=None,
                settings=ZipBatchSettings(batch_size=4),
                run_pipeline_file=run_pipeline_file,
                relative_parent=lambda *_args: Path("."),
                log=lambda _message: None,
                advance_progress=progress.append,
                idle_callback=_idle_callback,
            )

            self.assertEqual(4, len(result.processed_outputs))
            self.assertEqual([], result.failures)
            self.assertEqual([1, 1, 1, 1], progress)
            self.assertGreater(max_active, 1)
            self.assertGreater(idle_calls, 0)

    def test_run_zip_pipeline_run_parallelizes_files_inside_batch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            zip_path = tmp_path / "inputs.zip"
            output_dir = tmp_path / "outputs"
            output_dir.mkdir()
            with zipfile.ZipFile(zip_path, "w") as archive:
                for index in range(4):
                    archive.writestr(f"CTRL/sample_{index}.h5", "h5")

            members = list_h5_members(zip_path)
            active = 0
            max_active = 0
            idle_calls = 0
            lock = threading.Lock()

            def run_pipeline_file(
                h5_path,
                _pipelines,
                output_root,
                output_relative_parent=Path("."),
                output_filename=None,
            ):
                nonlocal active, max_active
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                try:
                    time.sleep(0.12)
                    target_dir = output_root / output_relative_parent
                    target_dir.mkdir(parents=True, exist_ok=True)
                    output_path = target_dir / (
                        output_filename or f"{h5_path.stem}_result.h5"
                    )
                    output_path.write_text("result", encoding="utf-8")
                    return output_path
                finally:
                    with lock:
                        active -= 1

            result = run_zip_pipeline_run(
                zip_path=zip_path,
                members=members,
                member_count=len(members),
                pipelines=[object()],
                output_dir=output_dir,
                settings=ZipBatchSettings(
                    batch_size=4,
                    extract_workers=1,
                ),
                run_pipeline_file=run_pipeline_file,
                log=lambda _message: None,
                advance_progress=lambda _units: None,
            )

            self.assertEqual(4, len(result.processed_outputs))
            self.assertEqual([], result.failures)
            self.assertGreater(max_active, 1)

            def _idle_callback():
                nonlocal idle_calls
                idle_calls += 1

            run_zip_pipeline_run(
                zip_path=zip_path,
                members=members,
                member_count=len(members),
                pipelines=[object()],
                output_dir=output_dir,
                settings=ZipBatchSettings(
                    batch_size=4,
                    extract_workers=1,
                ),
                run_pipeline_file=run_pipeline_file,
                log=lambda _message: None,
                advance_progress=lambda _units: None,
                idle_callback=_idle_callback,
            )
            self.assertGreater(idle_calls, 0)


class WorkflowDispatchTests(unittest.TestCase):
    def _request(self, tmp_path: Path, **overrides) -> WorkflowRunRequest:
        values = {
            "mode": "folder",
            "pipelines": [object()],
            "postprocesses": [],
            "selected_pipeline_names": ["Demo"],
            "base_output_dir": tmp_path / "outputs",
            "zip_outputs": False,
            "zip_name": "outputs.zip",
            "trim_source": True,
            "zip_output_dir": lambda folder, target_path=None, progress_callback=None: (
                target_path or folder / "outputs.zip"
            ),
            "input_plan": None,
        }
        values.update(overrides)
        values["base_output_dir"].mkdir(parents=True, exist_ok=True)
        return WorkflowRunRequest(**values)

    def _callbacks(self) -> WorkflowCallbacks:
        return WorkflowCallbacks(
            log=lambda _message: None,
            start_primary_progress=lambda _units, _status: None,
            start_final_progress=lambda _units, _status: None,
            advance_progress=lambda _units=1.0: None,
            set_progress_units=lambda _units: None,
            set_status=lambda _status: None,
            make_zip_progress_callback=lambda: None,
        )

    def _result(self, tmp_path: Path) -> RunWorkflowResult:
        return RunWorkflowResult(
            output_dir=tmp_path / "outputs",
            processed_outputs=[],
            failures=[],
            summary_message="done",
        )

    def test_dispatch_routes_file_inputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            h5_path = tmp_path / "sample.h5"
            h5_path.write_text("h5", encoding="utf-8")
            request = self._request(
                tmp_path,
                mode="file",
                input_plan=prepare_run_input(h5_path),
            )

            with mock.patch(
                "workflows.dispatch.run_filesystem_workflow",
                return_value=self._result(tmp_path),
            ) as run_filesystem:
                result = dispatch_workflow(request, self._callbacks())

            self.assertIsNotNone(result.workflow_result)
            self.assertEqual([h5_path], list(run_filesystem.call_args.kwargs["inputs"]))

    def test_prepare_run_inputs_groups_multiple_hdf5_files_as_file_run(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            first_path = tmp_path / "first.h5"
            second_path = tmp_path / "second.hdf5"
            first_path.write_text("h5", encoding="utf-8")
            second_path.write_text("hdf5", encoding="utf-8")

            input_plan = prepare_run_inputs([first_path, second_path])

            self.assertEqual("file", input_plan.kind)
            self.assertEqual(tmp_path, input_plan.input_path)
            self.assertEqual((first_path, second_path), input_plan.h5_paths)

    def test_dispatch_routes_folder_inputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_dir = tmp_path / "inputs"
            input_dir.mkdir()
            (input_dir / "sample.h5").write_text("h5", encoding="utf-8")
            request = self._request(
                tmp_path,
                mode="folder",
                input_plan=prepare_run_input(input_dir),
            )

            with mock.patch(
                "workflows.dispatch.run_filesystem_workflow",
                return_value=self._result(tmp_path),
            ) as run_filesystem:
                result = dispatch_workflow(request, self._callbacks())

            self.assertIsNotNone(result.workflow_result)
            run_filesystem.assert_called_once()

    def test_dispatch_routes_zip_inputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            zip_path = tmp_path / "inputs.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("sample.h5", "h5")
            request = self._request(
                tmp_path,
                mode="zip",
                input_plan=prepare_run_input(zip_path),
            )

            with mock.patch(
                "workflows.dispatch.run_zip_workflow",
                return_value=self._result(tmp_path),
            ) as run_zip:
                result = dispatch_workflow(request, self._callbacks())

            self.assertIsNotNone(result.workflow_result)
            self.assertEqual(1, run_zip.call_args.kwargs["member_count"])

    def test_dispatch_routes_holo_inputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            holo_path = tmp_path / "sample.holo"
            holo_path.write_text("holo", encoding="utf-8")
            ef_h5_dir = tmp_path / "sample" / "sample_EF" / "h5"
            ef_h5_dir.mkdir(parents=True)
            (ef_h5_dir / "sample.h5").write_text("h5", encoding="utf-8")
            request = self._request(
                tmp_path,
                mode="holo",
                holo_paths=[holo_path],
            )

            with mock.patch(
                "workflows.dispatch.run_holo_workflow",
                return_value=self._result(tmp_path),
            ) as run_holo:
                result = dispatch_workflow(request, self._callbacks())

            self.assertIsNotNone(result.workflow_result)
            self.assertEqual(1, len(run_holo.call_args.kwargs["contexts"]))
            self.assertTrue((tmp_path / "sample" / "sample_AE").is_dir())

    def test_dispatch_rejects_empty_zip_without_tk(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            zip_path = tmp_path / "empty.zip"
            with zipfile.ZipFile(zip_path, "w"):
                pass
            request = self._request(
                tmp_path,
                mode="zip",
                input_plan=prepare_run_input(zip_path),
            )

            with self.assertRaises(WorkflowInputError):
                dispatch_workflow(request, self._callbacks())

    def test_dispatch_rejects_mismatched_mode_and_input_kind(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            h5_path = tmp_path / "sample.h5"
            h5_path.write_text("h5", encoding="utf-8")
            request = self._request(
                tmp_path,
                mode="folder",
                input_plan=prepare_run_input(h5_path),
            )

            with self.assertRaises(WorkflowInputError):
                dispatch_workflow(request, self._callbacks())


if __name__ == "__main__":
    unittest.main()
