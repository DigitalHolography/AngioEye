import sys
import tempfile
import unittest
from pathlib import Path
from threading import get_ident
from types import SimpleNamespace
from unittest import mock

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ui.controllers.run import RunTabController  # noqa: E402
from ui.drag_drop import DragDropMixin  # noqa: E402
from ui.services import UiServices  # noqa: E402
from ui.settings import SettingsMixin  # noqa: E402


class Var:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def make_app():
    pipeline = SimpleNamespace(name="Summary", available=True)
    return SimpleNamespace(
        batch_input_var=Var(),
        batch_output_var=Var(),
        batch_input_paths=[],
        batch_selection_label="",
        batch_zip_var=Var(False),
        batch_zip_name_var=Var("statistics.zip"),
        processing_branch_var=Var("ProcessingAlt"),
        minimal_input_path_var=Var(),
        minimal_output_path_var=Var(),
        pipeline_rows=[pipeline],
        pipeline_visibility={"Summary": True},
        _batch_running=False,
        _progress_total_units=1.0,
        _reset_progress=mock.Mock(),
        _set_progress_units=mock.Mock(),
        _set_minimal_status=mock.Mock(),
        _reset_batch_output=mock.Mock(),
        _log_batch=mock.Mock(),
        _show_batch_error_dialog=mock.Mock(),
        update=mock.Mock(),
        after=mock.Mock(),
        destroy=mock.Mock(),
        withdraw=mock.Mock(),
        _persist_ui_mode=mock.Mock(),
        ui_services=UiServices(dialogs=mock.Mock(), file_dialogs=mock.Mock()),
    )


class BatchControllerTests(unittest.TestCase):
    def test_explicit_selection_submits_whole_batch_once(self):
        app = make_app()
        controller = RunTabController(app)
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / "a_EF.h5", Path(directory) / "b_EF.h5"]
            controller.apply_inputs(paths)
            result = SimpleNamespace(failures=[], summary_message="Analyzed 2 files.")
            with mock.patch(
                "ui.controllers.run.run_batch", return_value=result
            ) as runner:
                controller.run()
                controller._worker_thread.join(timeout=5)
                controller._poll_worker_events()
            self.assertEqual(runner.call_count, 1)
            call = runner.call_args
            self.assertEqual(call.args[0], tuple(paths))
            self.assertEqual(call.args[1], app.pipeline_rows)
            self.assertEqual(call.kwargs["processing_branch"], "ProcessingAlt")
            self.assertFalse(app._batch_running)
            app._set_minimal_status.assert_called_with("Completed.")

    def test_manual_input_edit_overrides_multiselection(self):
        app = make_app()
        controller = RunTabController(app)
        controller.apply_inputs([Path("a.h5"), Path("b.h5")])
        app.batch_input_var.set("replacement.h5")
        self.assertEqual(controller.selected_inputs(), (Path("replacement.h5"),))

    def test_no_selected_pipeline_does_not_run(self):
        app = make_app()
        app.batch_input_var.set("sample.h5")
        app.pipeline_visibility["Summary"] = False
        with mock.patch("ui.controllers.run.run_batch") as runner:
            RunTabController(app).run()
        runner.assert_not_called()
        app.ui_services.dialogs.showwarning.assert_called_once()

    def test_run_failure_restores_controls_and_reports_error(self):
        app = make_app()
        app.batch_input_var.set("sample.h5")
        app.batch_output_var.set("outputs")
        app.run_button = mock.Mock()
        with mock.patch(
            "ui.controllers.run.run_batch",
            side_effect=ValueError("Invalid EyeFlow output"),
        ):
            controller = RunTabController(app)
            controller.run()
            controller._worker_thread.join(timeout=5)
            controller._poll_worker_events()
        self.assertFalse(app._batch_running)
        app.run_button.configure.assert_called_with(state="normal")
        app.ui_services.dialogs.showerror.assert_called_with(
            "Batch statistics failed", "Invalid EyeFlow output"
        )

    def test_reentrant_run_is_ignored(self):
        app = make_app()
        app._batch_running = True
        with mock.patch("ui.controllers.run.run_batch") as runner:
            RunTabController(app).run()
        runner.assert_not_called()

    def test_worker_callbacks_reach_tk_only_when_main_thread_polls(self):
        app = make_app()
        app.batch_input_var.set("sample.h5")
        app.batch_output_var.set("outputs")
        calls = []
        app._log_batch.side_effect = lambda message: calls.append(
            (message, get_ident())
        )
        app._set_minimal_status.side_effect = lambda message: calls.append(
            (message, get_ident())
        )

        def fake_batch(*_args, log, progress, **_kwargs):
            log("Batch log")
            progress(1, 2, "Analyzing")
            return SimpleNamespace(failures=[], summary_message="Complete")

        controller = RunTabController(app)
        with mock.patch("ui.controllers.run.run_batch", side_effect=fake_batch):
            controller.run()
            controller._worker_thread.join(timeout=5)
        self.assertEqual(calls, [])
        self.assertTrue(app._batch_running)
        controller._poll_worker_events()
        self.assertEqual(
            [message for message, _thread in calls],
            ["Batch log", "Analyzing", "Completed."],
        )
        self.assertTrue(all(thread == get_ident() for _message, thread in calls))
        self.assertFalse(app._batch_running)

    def test_closing_window_finishes_worker_without_showing_dialog(self):
        app = make_app()
        app.batch_input_var.set("sample.h5")
        app.batch_output_var.set("outputs")
        controller = RunTabController(app)
        with mock.patch(
            "ui.controllers.run.run_batch", side_effect=ValueError("Failed")
        ):
            controller.run()
            controller._worker_thread.join(timeout=5)
        SettingsMixin._on_close(app)
        app.withdraw.assert_called_once()
        app.destroy.assert_not_called()
        controller._poll_worker_events()
        app.destroy.assert_called_once()
        app.ui_services.dialogs.showerror.assert_not_called()
        self.assertFalse(app._batch_running)
        self.assertFalse(controller._worker_thread.daemon)

    def test_drop_accepts_multiple_hdf5_files_or_single_folder(self):
        app = make_app()
        app.run_controller = RunTabController(app)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "a.h5"
            second = root / "b.hdf5"
            first.touch()
            second.touch()
            self.assertTrue(DragDropMixin._handle_dropped_paths(app, [first, second]))
            self.assertEqual(app.run_controller.selected_inputs(), (first, second))
            self.assertFalse(DragDropMixin._handle_dropped_paths(app, [first, root]))
            self.assertTrue(DragDropMixin._handle_dropped_paths(app, [root]))
            self.assertEqual(app.run_controller.selected_inputs(), (root,))
            unsupported = root / "sample.holo"
            unsupported.touch()
            self.assertFalse(DragDropMixin._handle_dropped_paths(app, [unsupported]))


class GuiConstructionTests(unittest.TestCase):
    def test_gui_exposes_only_batch_pipelines_with_valid_defaults(self):
        import tkinter as tk

        import ui.app as app_module
        from app_settings import AppSettingsStore

        original_init = app_module.BaseAppTk.__init__

        def hidden_init(window, *args, **kwargs):
            original_init(window, *args, **kwargs)
            window.withdraw()

        with tempfile.TemporaryDirectory() as directory:
            store = AppSettingsStore(
                Path(directory) / "settings.json",
                Path(__file__).resolve().parents[1] / "default_settings.json",
            )
            app = None
            try:
                with (
                    mock.patch.object(
                        app_module, "AppSettingsStore", return_value=store
                    ),
                    mock.patch.object(app_module.BaseAppTk, "__init__", hidden_init),
                    mock.patch.object(
                        app_module.ProcessApp, "_ensure_window_size_for_mode"
                    ),
                ):
                    try:
                        app = app_module.ProcessApp()
                    except tk.TclError as exc:
                        self.skipTest(f"Tk display unavailable: {exc}")
                self.assertEqual(
                    [app.notebook.tab(tab, "text") for tab in app.notebook.tabs()],
                    ["Run", "Pipelines"],
                )
                selected = app.run_controller.selected_pipelines()
                self.assertIn(
                    "Pipeline Metrics Manifest", [item.name for item in selected]
                )
                self.assertTrue(all(item.available for item in selected))
                self.assertEqual(app.processing_branch_var.get(), "Processing")
                self.assertEqual(app.state(), "withdrawn")
            finally:
                if app is not None:
                    app.destroy()
