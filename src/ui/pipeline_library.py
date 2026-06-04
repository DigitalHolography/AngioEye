import tkinter as tk
from tkinter import ttk


class PipelineLibraryTab(ttk.Frame):
    def __init__(self, parent: tk.Misc, controller) -> None:
        super().__init__(parent, padding=10)
        self.controller = controller
        self._build()

    def _build(self) -> None:
        controller = self.controller
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        ttk.Label(
            self,
            text="Select the pipelines to run. "
            "This preference is saved between app launches.",
        ).grid(row=0, column=0, sticky="w")

        controls = ttk.Frame(self)
        controls.grid(row=1, column=0, sticky="ew", pady=(8, 4))
        controls.columnconfigure(4, weight=1)
        ttk.Button(
            controls,
            text="Select all",
            command=controller.select_all,
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(
            controls,
            text="Deselect all",
            command=controller.deselect_all,
        ).grid(row=0, column=1, sticky="w", padx=(4, 0))
        ttk.Button(
            controls,
            text="Reload pipelines",
            command=controller.refresh,
        ).grid(row=0, column=2, sticky="w", padx=(4, 0))
        ttk.Button(
            controls,
            text="Open folder",
            command=controller.open_folder,
        ).grid(row=0, column=3, sticky="w", padx=(4, 0))
        ttk.Label(controls, textvariable=controller.summary_var).grid(
            row=0, column=4, sticky="e"
        )

        library_container = ttk.Frame(self)
        library_container.grid(row=2, column=0, sticky="nsew")
        library_container.columnconfigure(0, weight=1)
        library_container.rowconfigure(0, weight=1)

        canvas = tk.Canvas(
            library_container, highlightthickness=0, bg=controller.bg_color
        )
        canvas.grid(row=0, column=0, sticky="nsew")
        library_scroll = ttk.Scrollbar(
            library_container,
            orient="vertical",
            command=canvas.yview,
        )
        library_scroll.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=library_scroll.set)
        inner = ttk.Frame(canvas)
        window = canvas.create_window((0, 0), window=inner, anchor="nw")
        controller.set_canvas_widgets(canvas, inner, window)
        inner.bind(
            "<Configure>",
            lambda _evt: canvas.configure(
                scrollregion=canvas.bbox("all")
            ),
        )
        canvas.bind(
            "<Configure>",
            lambda evt: canvas.itemconfigure(
                window, width=evt.width
            ),
        )
        controller.bind_mousewheel(canvas, canvas)
        controller.bind_mousewheel(inner, canvas)
        controller.bind_mousewheel(library_scroll, canvas)
