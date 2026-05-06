import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

fake_h5py = types.ModuleType("h5py")
fake_h5py.File = object
sys.modules.setdefault("h5py", fake_h5py)

fake_pipelines = types.ModuleType("pipelines")
fake_pipelines.PipelineDescriptor = object
fake_pipelines.ProcessResult = object
fake_pipelines.load_pipeline_catalog = lambda: ([], [])
fake_pipelines.process_results_to_metric_trees = lambda *args, **kwargs: []
sys.modules.setdefault("pipelines", fake_pipelines)
sys.modules.setdefault("pipelines.core", types.ModuleType("pipelines.core"))

fake_pipeline_errors = types.ModuleType("pipelines.core.errors")
fake_pipeline_errors.format_pipeline_exception = lambda exc, _pipeline: str(exc)
sys.modules.setdefault("pipelines.core.errors", fake_pipeline_errors)

fake_postprocess = types.ModuleType("postprocess")
fake_postprocess.PostprocessContext = object
fake_postprocess.PostprocessDescriptor = object
fake_postprocess.load_postprocess_catalog = lambda: ([], [])
sys.modules.setdefault("postprocess", fake_postprocess)

from angioeye_io import (  # noqa: E402
    default_output_dir_for_input,
    default_work_h5_name_for_input,
    get_h5_stem,
)
from angio_eye import (  # noqa: E402
    ProcessApp,
)

for _module_name in (
    "angioeye_io",
    "angioeye_io.archive_io",
    "angioeye_io.hdf5_io",
    "angioeye_io.hdf5_schema",
):
    sys.modules.pop(_module_name, None)

for _module_name in (
    "h5py",
    "pipelines",
    "pipelines.core",
    "pipelines.core.errors",
    "postprocess",
):
    _module = sys.modules.get(_module_name)
    if _module is not None and getattr(_module, "__file__", None) is None:
        sys.modules.pop(_module_name, None)


class _Var:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


class AngioEyeOutputNamingTests(unittest.TestCase):
    def test_get_h5_stem_strips_eyeflow_result_suffix(self) -> None:
        self.assertEqual(
            "251031_ALA_L_1",
            get_h5_stem(Path("251031_ALA_L_1_eyeflow_pipelines_result.h5")),
        )

    def test_default_work_h5_name_for_eyeflow_result(self) -> None:
        self.assertEqual(
            "251031_ALA_L_1_AE.h5",
            default_work_h5_name_for_input(
                Path("251031_ALA_L_1_eyeflow_pipelines_result.h5")
            ),
        )

    def test_default_output_dir_for_h5_input(self) -> None:
        input_path = Path("root") / "251031_ALA_L_1.h5"

        self.assertEqual(
            Path("root") / "251031_ALA_L_1" / "251031_ALA_L_1_AE",
            default_output_dir_for_input(input_path),
        )

    def test_default_output_dir_for_ef_sibling_input(self) -> None:
        input_path = (
            Path("root")
            / "251031_ALA_L_1"
            / "251031_ALA_L_1_EF"
            / "251031_ALA_L_1_EF.h5"
        )

        self.assertEqual(
            Path("root") / "251031_ALA_L_1" / "251031_ALA_L_1_AE",
            default_output_dir_for_input(input_path),
        )

    def test_default_output_dir_climbs_to_base_folder(self) -> None:
        input_path = (
            Path("root")
            / "251031_ALA_L_1"
            / "251031_ALA_L_1_EF"
            / "nested"
            / "251031_ALA_L_1_EF.h5"
        )

        self.assertEqual(
            Path("root") / "251031_ALA_L_1" / "251031_ALA_L_1_AE",
            default_output_dir_for_input(input_path),
        )

    @mock.patch("angio_eye.h5py.File")
    def test_run_pipelines_on_file_overwrites_ae_h5_path(self, h5_file) -> None:
        class _H5Context:
            def __enter__(self):
                return object()

            def __exit__(self, *_args):
                return False

        pipeline_result = SimpleNamespace(output_h5_path=None)
        pipeline = SimpleNamespace(name="Demo", run=lambda _h5file: pipeline_result)
        pipeline_desc = SimpleNamespace(instantiate=lambda: pipeline)
        written_paths: list[Path] = []

        h5_file.return_value = _H5Context()
        app = SimpleNamespace(
            _log_batch=lambda _message: None,
            _advance_progress=lambda: None,
            _write_combined_results_with_ui_pump=lambda **kwargs: written_paths.append(
                kwargs["combined_h5_out"]
            ),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "251031_ALA_L_1_AE"
            output_root.mkdir()
            existing_output = output_root / "251031_ALA_L_1_AE.h5"
            existing_output.write_text("old", encoding="utf-8")

            result_path = ProcessApp._run_pipelines_on_file(
                app,
                Path(tmp_dir) / "251031_ALA_L_1_eyeflow_pipelines_result.h5",
                [pipeline_desc],
                output_root,
            )

            self.assertEqual(existing_output, result_path)
            self.assertEqual([existing_output], written_paths)
            self.assertFalse((output_root / "251031_ALA_L_1_AE_1.h5").exists())
            self.assertEqual(str(existing_output), pipeline_result.output_h5_path)


class BatchZipCleanupTests(unittest.TestCase):
    def _make_fake_app(
        self,
        *,
        input_path: Path,
        base_output_dir: Path,
        zip_should_fail: bool,
    ):
        logs: list[str] = []
        pipeline_target_dirs: list[Path] = []
        data_root = input_path.parent / "extracted"
        h5_path = data_root / "sample.h5"

        def _run_pipelines_on_file(
            _h5_path,
            _pipelines,
            target_dir,
            output_filename=None,
        ):
            pipeline_target_dirs.append(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            result_path = target_dir / (output_filename or "sample_pipelines_result.h5")
            result_path.write_text("result", encoding="utf-8")
            return result_path

        def _zip_output_dir(_folder, target_path=None, progress_callback=None):
            if zip_should_fail:
                raise RuntimeError("zip failed")
            if progress_callback is not None:
                progress_callback(1, 1, Path("sample_pipelines_result.h5"))
            assert target_path is not None
            target_path.write_text("archive", encoding="utf-8")
            return target_path

        return SimpleNamespace(
            batch_input_var=_Var(str(input_path)),
            batch_output_var=_Var(str(base_output_dir)),
            batch_zip_var=_Var(True),
            batch_zip_name_var=_Var("outputs.zip"),
            ui_mode="advanced",
            _progress_total_units=1.0,
            _progress_completed_units=0.0,
            _progress_primary_style="primary",
            _progress_final_style="final",
            pipeline_rows=[SimpleNamespace(name="Demo", available=True)],
            postprocess_rows=[],
            pipeline_visibility={"Demo": True},
            postprocess_visibility={},
            pipeline_registry={"Demo": object()},
            postprocess_registry={},
            _validate_postprocess_selection=lambda *args, **kwargs: [],
            _reset_batch_output=lambda *args, **kwargs: None,
            _prepare_data_root=lambda _path: (data_root, None),
            _find_h5_inputs=lambda _path: [h5_path],
            _relative_input_parent=lambda *args, **kwargs: Path("."),
            _run_pipelines_on_file=_run_pipelines_on_file,
            _run_postprocesses=lambda *args, **kwargs: None,
            _zip_output_dir=_zip_output_dir,
            _log_batch=logs.append,
            _show_batch_error_dialog=lambda *args, **kwargs: None,
            _reset_progress=lambda: None,
            _start_progress=lambda total_units, **_kwargs: None,
            _set_progress_units=lambda completed_units: None,
            _advance_progress=lambda units=1.0: None,
            _set_minimal_status=lambda _message: None,
            _minimal_output_filename_for_run=lambda _data_path, _inputs: None,
            update=lambda: None,
            logs=logs,
            pipeline_target_dirs=pipeline_target_dirs,
        )

    @mock.patch("angio_eye.messagebox.showwarning")
    @mock.patch("angio_eye.messagebox.showerror")
    @mock.patch("angio_eye.messagebox.showinfo")
    def test_run_batch_removes_temp_output_dir_after_successful_zip(
        self,
        _showinfo,
        _showerror,
        _showwarning,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "input.zip"
            input_path.write_text("dummy", encoding="utf-8")
            base_output_dir = tmp_path / "outputs"
            base_output_dir.mkdir()

            app = self._make_fake_app(
                input_path=input_path,
                base_output_dir=base_output_dir,
                zip_should_fail=False,
            )

            ProcessApp.run_batch(app)

            self.assertTrue((base_output_dir / "outputs.zip").exists())
            self.assertEqual(
                [base_output_dir / "outputs.zip"],
                sorted(base_output_dir.iterdir()),
            )
            self.assertTrue(
                any("Completed. ZIP archive:" in message for message in app.logs),
            )

    @mock.patch("angio_eye.messagebox.showwarning")
    @mock.patch("angio_eye.messagebox.showerror")
    @mock.patch("angio_eye.messagebox.showinfo")
    def test_run_batch_keeps_work_dir_when_zip_creation_fails(
        self,
        _showinfo,
        showerror,
        _showwarning,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "input.zip"
            input_path.write_text("dummy", encoding="utf-8")
            base_output_dir = tmp_path / "outputs"
            base_output_dir.mkdir()

            app = self._make_fake_app(
                input_path=input_path,
                base_output_dir=base_output_dir,
                zip_should_fail=True,
            )

            ProcessApp.run_batch(app)

            work_dirs = [path for path in base_output_dir.iterdir() if path.is_dir()]
            self.assertEqual(1, len(work_dirs))
            self.assertTrue(
                (work_dirs[0] / "sample_pipelines_result.h5").exists(),
            )
            self.assertFalse((base_output_dir / "outputs.zip").exists())
            self.assertTrue(
                any(str(work_dirs[0]) in message for message in app.logs),
            )
            self.assertEqual("Zip failed", showerror.call_args.args[0])

    @mock.patch("angio_eye.messagebox.showwarning")
    @mock.patch("angio_eye.messagebox.showerror")
    @mock.patch("angio_eye.messagebox.showinfo")
    def test_run_batch_resolves_target_dir_before_pipeline_run(
        self,
        _showinfo,
        _showerror,
        _showwarning,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "input.zip"
            input_path.write_text("dummy", encoding="utf-8")
            base_output_dir = tmp_path / "outputs"
            base_output_dir.mkdir()

            app = self._make_fake_app(
                input_path=input_path,
                base_output_dir=base_output_dir,
                zip_should_fail=False,
            )
            nested_parent = Path("nested")
            app._relative_input_parent = lambda *_args, **_kwargs: nested_parent

            ProcessApp.run_batch(app)

            self.assertEqual(1, len(app.pipeline_target_dirs))
            self.assertEqual(nested_parent.name, app.pipeline_target_dirs[0].name)
            self.assertTrue(app.pipeline_target_dirs[0].parent.parent == base_output_dir)

    def test_apply_input_defaults_for_zip_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "sample.zip"
            input_path.write_text("dummy", encoding="utf-8")

            app = SimpleNamespace(
                batch_output_var=_Var(""),
                batch_zip_var=_Var(False),
                batch_zip_name_var=_Var("outputs.zip"),
                _default_archive_name=lambda path: f"{path.stem}_AE.zip",
                _reset_progress=lambda: None,
                _set_minimal_status=lambda _message: None,
            )

            ProcessApp._apply_input_defaults(app, input_path)

            self.assertEqual(str(tmp_path), app.batch_output_var.get())
            self.assertTrue(app.batch_zip_var.get())
            self.assertEqual("sample_AE.zip", app.batch_zip_name_var.get())

    def test_apply_input_defaults_for_h5_input_uses_ae_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "sample.h5"
            input_path.write_text("dummy", encoding="utf-8")

            app = SimpleNamespace(
                batch_output_var=_Var(""),
                batch_zip_var=_Var(True),
                batch_zip_name_var=_Var("outputs.zip"),
                _default_archive_name=lambda path: f"{path.stem}_AE.zip",
                _reset_progress=lambda: None,
                _set_minimal_status=lambda _message: None,
            )

            ProcessApp._apply_input_defaults(app, input_path)

            self.assertEqual(
                str(tmp_path / "sample" / "sample_AE"),
                app.batch_output_var.get(),
            )
            self.assertFalse(app.batch_zip_var.get())
            self.assertEqual("sample_AE.zip", app.batch_zip_name_var.get())

    def test_minimal_output_filename_for_single_h5(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "sample.h5"
            input_path.write_text("dummy", encoding="utf-8")

            app = SimpleNamespace(
                ui_mode="minimal",
                batch_zip_var=_Var(False),
                _default_output_artifact_name=lambda path: f"{path.stem}_AE.h5",
            )

            output_name = ProcessApp._minimal_output_filename_for_run(
                app,
                input_path,
                [input_path],
            )

            self.assertEqual("sample_AE.h5", output_name)

    def test_handle_dropped_paths_accepts_supported_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "sample.zip"
            input_path.write_text("dummy", encoding="utf-8")
            applied_paths: list[Path] = []
            logs: list[str] = []

            app = SimpleNamespace(
                batch_input_var=_Var(""),
                _apply_input_defaults=lambda path: applied_paths.append(path),
                _log_batch=logs.append,
            )

            accepted = ProcessApp._handle_dropped_paths(app, [input_path])

            self.assertTrue(accepted)
            self.assertEqual(str(input_path), app.batch_input_var.get())
            self.assertEqual([input_path], applied_paths)
            self.assertIn("Drag and drop", logs[0])


class MouseWheelBindingTests(unittest.TestCase):
    def test_mousewheel_scroll_units_handles_delta_and_button_events(self) -> None:
        self.assertEqual(
            -1, ProcessApp._mousewheel_scroll_units(SimpleNamespace(delta=120))
        )
        self.assertEqual(
            1, ProcessApp._mousewheel_scroll_units(SimpleNamespace(delta=-120))
        )
        self.assertEqual(
            -2, ProcessApp._mousewheel_scroll_units(SimpleNamespace(delta=240))
        )
        self.assertEqual(
            -1, ProcessApp._mousewheel_scroll_units(SimpleNamespace(delta=1))
        )
        self.assertEqual(
            -1, ProcessApp._mousewheel_scroll_units(SimpleNamespace(delta=0, num=4))
        )
        self.assertEqual(
            1, ProcessApp._mousewheel_scroll_units(SimpleNamespace(delta=0, num=5))
        )

    def test_bind_vertical_mousewheel_registers_handlers_that_scroll_canvas(self):
        app = ProcessApp.__new__(ProcessApp)
        widget = mock.Mock()
        canvas = mock.Mock()

        ProcessApp._bind_vertical_mousewheel(app, widget, canvas)

        self.assertEqual(
            ["<MouseWheel>", "<Button-4>", "<Button-5>"],
            [call.args[0] for call in widget.bind.call_args_list],
        )
        self.assertTrue(
            all(call.kwargs.get("add") == "+" for call in widget.bind.call_args_list)
        )

        mousewheel_handler = widget.bind.call_args_list[0].args[1]
        result = mousewheel_handler(SimpleNamespace(delta=-120))

        canvas.yview_scroll.assert_called_once_with(1, "units")
        self.assertEqual("break", result)


if __name__ == "__main__":
    unittest.main()
