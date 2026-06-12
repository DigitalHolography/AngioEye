import tkinter as tk
from .services import services_for

class SettingsMixin:
    def _ensure_default_settings(self) -> None:
        try:
            self.settings_store.initialize_from_defaults()
        except OSError as exc:
            self._show_settings_warning(
                "Settings not initialized",
                f"Could not create default settings file:\n{exc}",
            )

    def _show_settings_warning(self, title: str, details: str) -> None:
        if self._settings_warning_shown:
            return
        self._settings_warning_shown = True
        services_for(self).dialogs.showwarning(title, details)

    def _persist_ui_mode(self) -> None:
        try:
            self.settings_store.save_ui_mode(self.ui_mode)
        except OSError as exc:
            self._show_settings_warning(
                "Settings not saved",
                f"Could not save UI mode preference:\n{exc}",
            )

    def _persist_trim_h5source(self) -> None:
        try:
            self.settings_store.save_trim_h5source(
                not self._persist_eyeflow_data.get()
            )
        except OSError as exc:
            self._show_settings_warning(
                "Settings not saved",
                f"Could not save trim preference:\n{exc}",
            )

    def _window_size_for_mode(self, mode: str) -> tuple[int, int, int, int]:
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        if mode == "advanced":
            width = min(900, max(760, screen_width - 240), screen_width)
            height = min(640, max(520, screen_height - 240), screen_height)
            min_width = min(620, width)
            min_height = min(520, height)
        else:
            width = max(560, min(660, screen_width - 260))
            height = max(480, min(440, screen_height - 260))
            min_width = min(500, width)
            min_height = min(460, height)
        return width, height, min_width, min_height

    def _ensure_window_size_for_mode(
        self,
        mode: str,
        *,
        force_target_size: bool = False,
    ) -> None:
        target_width, target_height, min_width, min_height = self._window_size_for_mode(
            mode
        )
        self.minsize(min_width, min_height)

        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        current_width = max(self.winfo_width(), 1)
        current_height = max(self.winfo_height(), 1)
        if (
            not force_target_size
            and current_width >= min_width
            and current_height >= min_height
        ):
            return

        if force_target_size:
            if mode == "minimal":
                try:
                    if self.state() != "normal":
                        self.state("normal")
                except tk.TclError:
                    pass
                self.minimal_view.update_idletasks()
                requested_width = self.minimal_view.winfo_reqwidth() + 24
                requested_height = self.minimal_view.winfo_reqheight() + 24
                width = min(
                    max(requested_width, min_width),
                    min(target_width, screen_width),
                )
                height = min(
                    max(requested_height, min_height),
                    min(target_height, screen_height),
                )
            else:
                width = min(target_width, screen_width)
                height = min(target_height, screen_height)
        else:
            width = min(max(current_width, min_width), screen_width)
            height = min(max(current_height, min_height), screen_height)
        x = max(min(self.winfo_x(), screen_width - width), 0)
        y = max(min(self.winfo_y(), screen_height - height), 0)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def _apply_ui_mode(self, mode: str, *, persist: bool = True) -> None:
        normalized_mode = (
            "advanced" if mode in {"advanced", "sandbox_advanced"} else "minimal"
        )
        previous_mode = self.ui_mode
        self.ui_mode = normalized_mode
        self.ui_mode_var.set(normalized_mode)

        self.minimal_view.pack_forget()
        self.advanced_view.pack_forget()
        if normalized_mode == "advanced":
            self.advanced_view.pack(fill="both", expand=True)
        else:
            self.minimal_view.pack(fill="both", expand=True)

        self.update_idletasks()

        self._ensure_window_size_for_mode(
            normalized_mode,
            force_target_size=(
                normalized_mode == "minimal"
                and (previous_mode == "advanced" or not persist)
            ),
        )
        if persist:
            self._persist_ui_mode()

    def _on_close(self) -> None:
        self._persist_ui_mode()
        self._persist_trim_h5source()
        self.destroy()
