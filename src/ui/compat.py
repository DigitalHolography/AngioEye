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


BaseAppTk = TkinterDnD.Tk if TkinterDnD is not None else tk.Tk
