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
sys.modules.setdefault("pipelines", fake_pipelines)
sys.modules.setdefault("pipelines.core", types.ModuleType("pipelines.core"))

fake_pipeline_errors = types.ModuleType("pipelines.core.errors")
fake_pipeline_errors.format_pipeline_exception = lambda exc, _pipeline: str(exc)
sys.modules.setdefault("pipelines.core.errors", fake_pipeline_errors)

fake_pipeline_utils = types.ModuleType("pipelines.core.utils")
fake_pipeline_utils.write_combined_results_h5 = lambda *args, **kwargs: None
sys.modules.setdefault("pipelines.core.utils", fake_pipeline_utils)

fake_postprocess = types.ModuleType("postprocess")
fake_postprocess.PostprocessContext = object
fake_postprocess.PostprocessDescriptor = object
fake_postprocess.load_postprocess_catalog = lambda: ([], [])
sys.modules.setdefault("postprocess", fake_postprocess)

from angio_eye import ProcessApp


class _Var:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class BatchZipCleanupTests(unittest.TestCase):
    def _make_fake_app(
        self,
        *,
        input_path: Path,
        base_output_dir: Path,
        zip_should_fail: bool,
    ):
        logs: list[str] = []
        data_root = input_path.parent / "extracted"
        h5_path = data_root / "sample.h5"

        def _run_pipelines_on_file(
            _h5_path,
            _pipelines,
            output_dir,
            output_relative_parent=Path("."),
        ):
            target_dir = output_dir / output_relative_parent
            target_dir.mkdir(parents=True, exist_ok=True)
            result_path = target_dir / "sample_pipelines_result.h5"
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
            pipeline_check_vars={"Demo": _Var(True)},
            postprocess_check_vars={},
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
            update=lambda: None,
            logs=logs,
        )

    @mock.patch("angio_eye.messagebox.showwarning")
    @mock.patch("angio_eye.messagebox.showerror")
    @mock.patch("angio_eye.messagebox.showinfo")
    def test_run_batch_removes_temp_output_dir_after_successful_zip(
        self,
        showinfo,
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
            self.assertIn("outputs.zip", showinfo.call_args.args[1])

    @mock.patch("angio_eye.messagebox.showwarning")
    @mock.patch("angio_eye.messagebox.showerror")
    @mock.patch("angio_eye.messagebox.showinfo")
    def test_run_batch_keeps_work_dir_when_zip_creation_fails(
        self,
        showinfo,
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
            self.assertIn(str(work_dirs[0]), showinfo.call_args.args[1])
            self.assertEqual("Zip failed", showerror.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
