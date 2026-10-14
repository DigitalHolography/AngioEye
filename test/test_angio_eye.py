import sys
import tempfile
import tkinter as tk
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
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def holo(self, name="sample.holo"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    def test_explicit_selection_submits_whole_batch_once(self):
        app = make_app()
        controller = RunTabController(app)
        paths = [self.holo("a.holo"), self.holo("b.holo")]
        self.assertTrue(controller.apply_inputs(paths))
        result = SimpleNamespace(failures=[], summary_message="Analyzed 2 files.")
        with mock.patch("ui.controllers.run.run_batch", return_value=result) as runner:
            controller.run()
            controller._worker_thread.join(timeout=5)
            controller._poll_worker_events()
        self.assertEqual(runner.call_count, 1)
        call = runner.call_args
        self.assertEqual(call.args[0], tuple(paths))
        self.assertEqual(call.args[1], app.pipeline_rows)
        self.assertEqual(call.args[2], self.root)
        self.assertEqual(call.kwargs["processing_branch"], "ProcessingAlt")
        self.assertFalse(app._batch_running)
        app._set_minimal_status.assert_called_with("Completed.")

    def test_manual_input_edit_overrides_multiselection(self):
        app = make_app()
        controller = RunTabController(app)
        controller.apply_inputs([self.holo("a.holo"), self.holo("b.holo")])
        replacement = self.holo("replacement.holo")
        app.batch_input_var.set(str(replacement))
        self.assertEqual(controller.selected_inputs(), (replacement,))

    def test_browse_only_offers_holo_files_and_accepts_recursive_folder(self):
        app = make_app()
        controller = RunTabController(app)
        holo = self.holo("nested/a.HOLO")
        app.ui_services.file_dialogs.askopenfilenames.return_value = [str(holo)]
        controller.choose_file()
        self.assertEqual(controller.selected_inputs(), (holo,))
        self.assertEqual(app.batch_output_var.get(), str(holo.parent))
        options = app.ui_services.file_dialogs.askopenfilenames.call_args.kwargs
        self.assertEqual(options["filetypes"], [("Hologram recordings", "*.holo")])
        app.ui_services.file_dialogs.askdirectory.return_value = str(self.root)
        controller.choose_folder()
        self.assertEqual(controller.selected_inputs(), (self.root,))
        self.assertEqual(app.batch_output_var.get(), str(self.root))
        app.ui_services.dialogs.showwarning.assert_not_called()

    def test_cancelled_browse_preserves_selection(self):
        app = make_app()
        controller = RunTabController(app)
        holo = self.holo()
        controller.apply_inputs([holo])
        app.ui_services.file_dialogs.askopenfilenames.return_value = ()
        app.ui_services.file_dialogs.askdirectory.return_value = ""
        controller.choose_file()
        controller.choose_folder()
        self.assertEqual(controller.selected_inputs(), (holo,))
        app.ui_services.dialogs.showwarning.assert_not_called()

    def test_invalid_selection_preserves_input_and_output(self):
        app = make_app()
        controller = RunTabController(app)
        holo = self.holo()
        controller.apply_inputs([holo])
        previous_output = app.batch_output_var.get()
        previous_archive = app.batch_zip_name_var.get()
        app._reset_progress.reset_mock()
        empty = self.root / "empty"
        empty.mkdir()
        h5 = self.root / "sample_EF.h5"
        h5.touch()
        invalid_selections = [
            [h5],
            [empty],
            [self.root / "missing.holo"],
            [holo, self.root],
            [holo, h5],
            [self.root, empty],
        ]
        for selection in invalid_selections:
            with self.subTest(selection=selection):
                app.ui_services.dialogs.showwarning.reset_mock()
                self.assertFalse(controller.apply_inputs(selection))
                self.assertEqual(controller.selected_inputs(), (holo,))
                self.assertEqual(app.batch_output_var.get(), previous_output)
                self.assertEqual(app.batch_zip_name_var.get(), previous_archive)
                app.ui_services.dialogs.showwarning.assert_called_once()
        app._reset_progress.assert_not_called()

    def test_typed_hdf5_zip_or_missing_holo_does_not_start_worker(self):
        app = make_app()
        app.batch_output_var.set(str(self.root / "outputs"))
        controller = RunTabController(app)
        for name in ("sample.h5", "sample.hdf5", "sample.zip", "missing.holo"):
            path = self.root / name
            if name != "missing.holo":
                path.touch()
            app.batch_input_var.set(str(path))
            with (
                self.subTest(name=name),
                mock.patch("ui.controllers.run.run_batch") as runner,
            ):
                app.ui_services.dialogs.showwarning.reset_mock()
                controller.run()
                runner.assert_not_called()
                self.assertFalse(app._batch_running)
                self.assertFalse(hasattr(controller, "_worker_thread"))
                app.ui_services.dialogs.showwarning.assert_called_once()

    def test_holo_without_processed_result_is_selected_then_reported_by_worker(self):
        app = make_app()
        controller = RunTabController(app)
        holo = self.holo()
        self.assertTrue(controller.apply_inputs([holo]))
        app.ui_services.dialogs.showwarning.assert_not_called()
        controller.run()
        controller._worker_thread.join(timeout=5)
        controller._poll_worker_events()
        self.assertFalse(app._batch_running)
        app.ui_services.dialogs.showerror.assert_called_once()
        message = app.ui_services.dialogs.showerror.call_args.args[1]
        self.assertIn("sample", message)
        self.assertIn("EyeFlow", message)

    def test_no_selected_pipeline_does_not_run(self):
        app = make_app()
        app.batch_input_var.set(str(self.holo()))
        app.pipeline_visibility["Summary"] = False
        with mock.patch("ui.controllers.run.run_batch") as runner:
            RunTabController(app).run()
        runner.assert_not_called()
        app.ui_services.dialogs.showwarning.assert_called_once()

    def test_run_failure_restores_controls_and_reports_error(self):
        app = make_app()
        app.batch_input_var.set(str(self.holo()))
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
        app.batch_input_var.set(str(self.holo()))
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
        app.batch_input_var.set(str(self.holo()))
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

    def test_drop_accepts_multiple_holo_files_or_single_recursive_folder(self):
        app = make_app()
        app.run_controller = RunTabController(app)
        first = self.holo("nested/a.holo")
        second = self.holo("nested/b.HOLO")
        self.assertTrue(DragDropMixin._handle_dropped_paths(app, [first, second]))
        self.assertEqual(app.run_controller.selected_inputs(), (first, second))
        self.assertFalse(DragDropMixin._handle_dropped_paths(app, [first, self.root]))
        self.assertTrue(DragDropMixin._handle_dropped_paths(app, [self.root]))
        self.assertEqual(app.run_controller.selected_inputs(), (self.root,))

    def make_drop_app(self):
        app = make_app()
        app.run_controller = RunTabController(app)
        app.tk = tk.Tcl()
        app._handle_dropped_paths = lambda paths: DragDropMixin._handle_dropped_paths(
            app, paths
        )
        return app

    def test_drop_tcl_list_preserves_spaces_unicode_and_reports_copy(self):
        app = self.make_drop_app()
        paths = [self.holo("Patient space/échantillon.holo"), self.holo("other.holo")]
        event = SimpleNamespace(data=" ".join(f"{{{path}}}" for path in paths))
        self.assertEqual(DragDropMixin._on_input_drop(app, event), "copy")
        self.assertEqual(app.run_controller.selected_inputs(), tuple(paths))
        app._log_batch.assert_called_once()
        app.ui_services.dialogs.showwarning.assert_not_called()

    def test_invalid_drop_refuses_with_one_warning_and_preserves_selection(self):
        app = self.make_drop_app()
        selected = self.holo()
        app.run_controller.apply_inputs([selected])
        empty = self.root / "empty"
        empty.mkdir()
        h5 = empty / "sample_EF.h5"
        h5.touch()
        archive = self.root / "results.zip"
        archive.touch()
        for paths in ([h5], [archive], [empty], [selected, self.root], []):
            with self.subTest(paths=paths):
                app.ui_services.dialogs.showwarning.reset_mock()
                event = SimpleNamespace(data=" ".join(f"{{{path}}}" for path in paths))
                self.assertEqual(
                    DragDropMixin._on_input_drop(app, event), "refuse_drop"
                )
                self.assertEqual(app.run_controller.selected_inputs(), (selected,))
                app.ui_services.dialogs.showwarning.assert_called_once()
        app._log_batch.assert_not_called()

    def test_malformed_drop_list_reports_one_warning(self):
        app = self.make_drop_app()
        event = SimpleNamespace(data="{unclosed path.holo")
        self.assertEqual(DragDropMixin._on_input_drop(app, event), "refuse_drop")
        app.ui_services.dialogs.showwarning.assert_called_once()
        self.assertEqual(app.run_controller.selected_inputs(), ())

    def test_running_batch_preserves_selection_and_refuses_browse_and_drop(self):
        app = self.make_drop_app()
        selected = self.holo()
        replacement = self.holo("replacement.holo")
        app.run_controller.apply_inputs([selected])
        app._batch_running = True
        app.run_controller.choose_file()
        app.run_controller.choose_folder()
        self.assertFalse(app.run_controller.apply_inputs([replacement]))
        self.assertEqual(DragDropMixin._on_input_drag(app, None), "refuse_drop")
        self.assertEqual(
            DragDropMixin._on_input_drop(
                app, SimpleNamespace(data=f"{{{replacement}}}")
            ),
            "refuse_drop",
        )
        self.assertEqual(app.run_controller.selected_inputs(), (selected,))
        app.ui_services.file_dialogs.askopenfilenames.assert_not_called()
        app.ui_services.file_dialogs.askdirectory.assert_not_called()
        app.ui_services.dialogs.showwarning.assert_not_called()


class GuiConstructionTests(unittest.TestCase):
    def test_gui_exposes_only_batch_pipelines_with_valid_defaults(self):
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
                self.assertEqual(
                    app.minimal_select_button.cget("text"), "Select .holo files"
                )
                self.assertEqual(
                    app.minimal_folder_button.cget("text"), "Select folder"
                )
                self.assertEqual(
                    app.minimal_input_hint.cget("text"), "drop folder or .holo file"
                )
                self.assertEqual(
                    app.input_hint.cget("text"), "drop folder or .holo file"
                )
                app.run_controller._set_running(True)
                self.assertEqual(
                    str(app.minimal_folder_button.cget("state")), "disabled"
                )
                self.assertEqual(str(app.input_entry.cget("state")), "disabled")
                app.run_controller._set_running(False)
                self.assertEqual(str(app.minimal_folder_button.cget("state")), "normal")
                self.assertEqual(str(app.input_entry.cget("state")), "normal")
                self.assertEqual(app.state(), "withdrawn")
            finally:
                if app is not None:
                    app.destroy()
