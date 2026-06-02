from pathlib import Path
from tkinter import messagebox

from app_settings import LAST_BATCH_LOG_FILENAME

class ProgressMixin:
    def _set_minimal_status(self, text: str) -> None:
        self.minimal_status_var.set(text)
        self.update_idletasks()

    def _batch_log_path(self) -> Path:
        return self.settings_store.path.with_name(LAST_BATCH_LOG_FILENAME)

    def _persist_batch_log_snapshot(self) -> None:
        log_path = self._batch_log_path()
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(
                self.batch_output.get("1.0", "end-1c"),
                encoding="utf-8",
            )
        except OSError:
            self._last_saved_batch_log_path = None
            return
        self._last_saved_batch_log_path = log_path

    def _set_progress_style(self, style_name: str) -> None:
        if hasattr(self, "minimal_progress"):
            self.minimal_progress.configure(style=style_name)

    def _reset_progress(self) -> None:
        self._progress_total_units = 1.0
        self._progress_completed_units = 0.0
        self._set_progress_style(self._progress_primary_style)
        self.batch_progress_var.set(0.0)
        self.update_idletasks()

    def _start_progress(
        self,
        total_units: float,
        *,
        style_name: str | None = None,
        status_text: str | None = None,
    ) -> None:
        self._progress_total_units = max(float(total_units), 1.0)
        self._progress_completed_units = 0.0
        self._set_progress_style(style_name or self._progress_primary_style)
        self.batch_progress_var.set(0.0)
        if status_text is not None:
            self._set_minimal_status(status_text)
        self.update_idletasks()

    def _set_progress_units(self, completed_units: float) -> None:
        clamped_units = min(
            max(float(completed_units), 0.0),
            max(self._progress_total_units, 1.0),
        )
        self._progress_completed_units = clamped_units
        self.batch_progress_var.set(
            (clamped_units / max(self._progress_total_units, 1.0)) * 100.0
        )
        self.update_idletasks()

    def _advance_progress(self, units: float = 1.0) -> None:
        self._set_progress_units(self._progress_completed_units + units)

    def _reset_batch_output(
        self,
        message: str = (
            "Select an input/output path, choose pipelines in Pipeline Library, "
            "choose optional postprocess steps in Postprocess Library, then run."
        ),
    ) -> None:
        self.batch_output.configure(state="normal")
        self.batch_output.delete("1.0", "end")
        self.batch_output.insert("end", message)
        self.batch_output.configure(state="disabled")
        self._persist_batch_log_snapshot()

    def _log_batch(self, text: str) -> None:
        self.batch_output.configure(state="normal")
        self.batch_output.insert("end", f"{text}\n")
        self.batch_output.see("end")
        self.batch_output.configure(state="disabled")
        self._persist_batch_log_snapshot()
        self.batch_output.update_idletasks()
        self.update_idletasks()

    def _show_batch_error_dialog(self, message: str) -> None:
        self.bell()
        if self._last_saved_batch_log_path is not None:
            message = (
                f"{message}\n\nLatest log saved to:\n{self._last_saved_batch_log_path}"
            )
        messagebox.showwarning(
            "Batch completed with errors",
            message,
        )
