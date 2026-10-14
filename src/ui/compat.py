import tkinter as tk

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:  # optional dependency
    DND_FILES = None
    TkinterDnD = None

try:
    import sv_ttk
except ImportError:  #  optional dependency
    sv_ttk = None


if TkinterDnD is None:
    BaseAppTk = tk.Tk
else:

    class BaseAppTk(TkinterDnD.Tk):
        def __init__(self, *args, **kwargs):
            try:
                super().__init__(*args, **kwargs)
            except RuntimeError:
                # A wheel can import correctly while its native tkdnd library
                # does not support the installed Tcl version. The Tk root has
                # already been created; keep the GUI usable without dropping.
                if "tk" not in self.__dict__:
                    raise
                self._drag_drop_available = False
