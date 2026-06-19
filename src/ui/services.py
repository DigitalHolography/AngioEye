from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Protocol


class DialogService(Protocol):
    def showwarning(self, title: str, message: str) -> object: ...

    def showerror(self, title: str, message: str) -> object: ...

    def showinfo(self, title: str, message: str) -> object: ...


class FileDialogService(Protocol):
    def askdirectory(self, **options) -> str: ...

    def askopenfilenames(self, **options) -> Sequence[str]: ...


class FolderOpener(Protocol):
    def open_folder(self, folder: Path) -> None: ...


class FileOpener(Protocol):
    def open_file(self, file_path: Path) -> None: ...


class TkDialogService:
    def showwarning(self, title: str, message: str) -> object:
        return messagebox.showwarning(title, message)

    def showerror(self, title: str, message: str) -> object:
        return messagebox.showerror(title, message)

    def showinfo(self, title: str, message: str) -> object:
        return messagebox.showinfo(title, message)


class TkFileDialogService:
    def askdirectory(self, **options) -> str:
        return filedialog.askdirectory(**options)

    def askopenfilenames(self, **options) -> Sequence[str]:
        return filedialog.askopenfilenames(**options)


class SystemFolderOpener:
    def open_folder(self, folder: Path) -> None:
        if sys.platform.startswith("win"):
            os.startfile(str(folder))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(folder)], check=False)
        else:
            subprocess.run(["xdg-open", str(folder)], check=False)


class SystemFileOpener:
    def open_file(self, file_path: Path) -> None:
        if sys.platform.startswith("win"):
            os.startfile(str(file_path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(file_path)], check=False)
        else:
            subprocess.run(["xdg-open", str(file_path)], check=False)


@dataclass(frozen=True)
class UiServices:
    dialogs: DialogService = field(default_factory=TkDialogService)
    file_dialogs: FileDialogService = field(default_factory=TkFileDialogService)
    folder_opener: FolderOpener = field(default_factory=SystemFolderOpener)
    file_opener: FileOpener = field(default_factory=SystemFileOpener)


def services_for(app) -> UiServices:
    return getattr(app, "ui_services", UiServices())
