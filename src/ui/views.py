import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

from .pipeline_library import PipelineLibraryTab
from .postprocess_library import PostprocessLibraryTab


class MinimalView(ttk.Frame):
    def __init__(self, parent: tk.Misc, controller) -> None:
        super().__init__(parent, padding=10)
        self.controller = controller
        self._build()

    def _build(self) -> None:
        app = self.controller
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.grid_anchor("center")

        content = ttk.Frame(self, padding=(24, 24))
        content.grid(row=0, column=0)
        content.columnconfigure(0, minsize=420)
        app.minimal_content = content

        app.minimal_title_label = ttk.Label(
            content,
            text="AngioEye",
            font=app._get_minimal_title_font(),
        )
        app.minimal_title_label.grid(row=0, column=0, pady=(0, 10))

        minimal_logo = app._load_scaled_logo_image(max_width=360, max_height=144)
        if minimal_logo is not None:
            app._minimal_logo_image = minimal_logo
            app.minimal_logo_label = ttk.Label(content, image=app._minimal_logo_image)
            app.minimal_logo_label.grid(row=1, column=0, pady=(0, 18))

        app.minimal_browse_button = ttk.Button(
            content,
            text="Browse .h5, .holo, or zip archive",
            command=app.choose_batch_file,
        )
        app.minimal_browse_button.grid(row=2, column=0, pady=(0, 10))
        app.minimal_input_path_label = tk.Label(
            content,
            textvariable=app.minimal_input_path_var,
            bg=app._bg_color,
            fg=app._muted_fg,
            justify="center",
            wraplength=420,
        )
        app.minimal_input_path_label.grid(row=3, column=0, pady=(0, 8), sticky="ew")

        app.minimal_holo_status_label = tk.Label(
            content,
            textvariable=app.holo_status_var,
            bg=app._bg_color,
            fg="#d65f5f",
            justify="center",
            wraplength=420,
        )
        app.minimal_holo_status_label.grid(row=4, column=0, pady=(0, 8), sticky="ew")
        app.minimal_holo_status_label.grid_remove()
        app.minimal_holo_output_label = tk.Label(
            content,
            textvariable=app.holo_output_path_var,
            bg=app._bg_color,
            fg=app._text_fg,
            justify="center",
            wraplength=420,
        )
        app.minimal_holo_output_label.grid(row=5, column=0, pady=(0, 8), sticky="ew")
        app.minimal_holo_output_label.grid_remove()

        app.minimal_output_button = ttk.Button(
            content,
            text="Select output folder",
            command=app.choose_batch_output,
        )
        app.minimal_output_button.grid(row=6, column=0, pady=(0, 10))
        app.minimal_output_path_label = tk.Label(
            content,
            textvariable=app.minimal_output_path_var,
            bg=app._bg_color,
            fg=app._muted_fg,
            justify="center",
            wraplength=420,
        )
        app.minimal_output_path_label.grid(row=7, column=0, pady=(0, 6), sticky="ew")
        app.minimal_output_name_label = tk.Label(
            content,
            textvariable=app.minimal_output_name_var,
            bg=app._bg_color,
            fg=app._text_fg,
            justify="center",
            wraplength=420,
        )
        app.minimal_output_name_label.grid(row=8, column=0, pady=(0, 18), sticky="ew")

        app.minimal_run_button = ttk.Button(
            content, text="Run", command=app.run_batch
        )
        app.minimal_run_button.grid(row=9, column=0, pady=(0, 18))

        app.minimal_progress = ttk.Progressbar(
            content,
            orient="horizontal",
            mode="determinate",
            maximum=100,
            variable=app.batch_progress_var,
            length=340,
            style=app._progress_primary_style,
        )
        app.minimal_progress.grid(row=10, column=0, sticky="ew")
        app.minimal_status_label = tk.Label(
            content,
            textvariable=app.minimal_status_var,
            bg=app._bg_color,
            fg=app._text_fg,
            justify="center",
            wraplength=420,
        )
        app.minimal_status_label.grid(row=11, column=0, pady=(8, 0), sticky="ew")


class RunTab(ttk.Frame):
    def __init__(self, parent: tk.Misc, controller) -> None:
        super().__init__(parent, padding=(18, 14, 14, 14))
        self.controller = controller
        self._build()

    def _build(self) -> None:
        app = self.controller
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        app.advanced_input_frame = ttk.Frame(self)
        app.advanced_input_frame.grid(row=0, column=0, sticky="ew")
        app.advanced_input_frame.columnconfigure(
            0, minsize=app._ADVANCED_FORM_LABEL_WIDTH
        )
        app.advanced_input_frame.columnconfigure(1, weight=1)

        ttk.Label(app.advanced_input_frame, text="Input").grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 16),
        )
        input_entry = ttk.Entry(
            app.advanced_input_frame, textvariable=app.batch_input_var
        )
        input_entry.grid(row=0, column=1, sticky="ew", padx=(0, 4))
        input_btn_frame = ttk.Frame(app.advanced_input_frame)
        input_btn_frame.grid(row=0, column=2, sticky="w")
        ttk.Button(
            input_btn_frame, text="Browse folder", command=app.choose_batch_folder
        ).pack(side="left")
        ttk.Button(
            input_btn_frame, text="Browse file", command=app.choose_batch_file
        ).pack(side="left", padx=(4, 0))

        app.advanced_holo_status_label = tk.Label(
            app.advanced_input_frame,
            textvariable=app.holo_status_var,
            bg=app._bg_color,
            fg="#d65f5f",
            justify="left",
            anchor="w",
        )
        app.advanced_holo_status_label.grid(
            row=1, column=1, columnspan=2, sticky="ew", pady=(8, 0)
        )
        app.advanced_holo_status_label.grid_remove()
        app.advanced_holo_output_label = tk.Label(
            app.advanced_input_frame,
            textvariable=app.holo_output_path_var,
            bg=app._bg_color,
            fg=app._text_fg,
            justify="left",
            anchor="w",
        )
        app.advanced_holo_output_label.grid(
            row=2, column=1, columnspan=2, sticky="ew", pady=(4, 10)
        )
        app.advanced_holo_output_label.grid_remove()

        ttk.Label(app.advanced_input_frame, text="Output").grid(
            row=3,
            column=0,
            sticky="w",
            padx=(0, 16),
            pady=(10, 0),
        )
        batch_output_entry = ttk.Entry(
            app.advanced_input_frame, textvariable=app.batch_output_var
        )
        batch_output_entry.grid(
            row=3, column=1, sticky="ew", padx=(0, 4), pady=(10, 0)
        )
        ttk.Button(
            app.advanced_input_frame,
            text="Browse",
            command=app.choose_batch_output,
        ).grid(row=3, column=2, sticky="w", pady=(10, 0))

        run_btn = ttk.Button(
            app.advanced_input_frame, text="Run", command=app.run_batch
        )
        run_btn.grid(
            row=4,
            column=0,
            sticky="w",
            padx=(0, 16),
            pady=(12, 0),
        )

        persist_eyeflow_data_btn = ttk.Checkbutton(
            app.advanced_input_frame,
            text="Persist Eyeflow Data",
            variable=app._persist_eyeflow_data,
            command=app._persist_trim_h5source,
        )
        persist_eyeflow_data_btn.grid(row=4, column=1, sticky="w", pady=(12, 0))

        ttk.Label(self, text="BatchLog").grid(
            row=1, column=0, sticky="w", pady=(16, 4)
        )
        batch_output_frame = ttk.Frame(self)
        batch_output_frame.grid(row=2, column=0, sticky="nsew")
        batch_output_frame.columnconfigure(0, weight=1)
        batch_output_frame.rowconfigure(0, weight=1)
        app.batch_output = tk.Text(
            batch_output_frame,
            height=14,
            state="disabled",
            bg=app._text_bg,
            fg=app._text_fg,
            insertbackground=app._text_fg,
        )
        batch_output_scroll = ttk.Scrollbar(
            batch_output_frame, orient="vertical", command=app.batch_output.yview
        )
        app.batch_output.configure(yscrollcommand=batch_output_scroll.set)
        app.batch_output.grid(row=0, column=0, sticky="nsew")
        batch_output_scroll.grid(row=0, column=1, sticky="ns")


class AdvancedView(ttk.Frame):
    def __init__(self, parent: tk.Misc, controller) -> None:
        super().__init__(parent, padding=10)
        self.controller = controller
        self._build()

    def _build(self) -> None:
        app = self.controller
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        app.notebook = ttk.Notebook(self, style="Advanced.TNotebook")
        app.notebook.grid(row=0, column=0, sticky="nsew")

        app.batch_tab = RunTab(app.notebook, app)
        app.pipeline_library_tab = PipelineLibraryTab(app.notebook, app)
        app.postprocess_library_tab = PostprocessLibraryTab(app.notebook, app)
        app.notebook.add(app.batch_tab, text="Run")
        app.notebook.add(app.pipeline_library_tab, text="Pipeline Library")
        app.notebook.add(app.postprocess_library_tab, text="Postprocess Library")


class ViewBuilderMixin:
    def _build_ui(self) -> None:
        self._build_menu()

        container = ttk.Frame(self, padding=10)
        container.pack(fill="both", expand=True)
        self.main_container = container

        self.minimal_view = MinimalView(container, self)
        self.advanced_view = AdvancedView(container, self)

    def _build_menu(self) -> None:
        self.ui_mode_var = tk.StringVar(value=self.ui_mode)
        menu_bar = tk.Menu(self)
        view_menu = tk.Menu(menu_bar, tearoff=False)
        view_menu.add_radiobutton(
            label="Minimal Mode",
            value="minimal",
            variable=self.ui_mode_var,
            command=lambda: self._apply_ui_mode(self.ui_mode_var.get()),
        )
        view_menu.add_radiobutton(
            label="Advanced Mode",
            value="advanced",
            variable=self.ui_mode_var,
            command=lambda: self._apply_ui_mode(self.ui_mode_var.get()),
        )
        menu_bar.add_cascade(label="View", menu=view_menu)
        self.configure(menu=menu_bar)

    def _get_minimal_title_font(self) -> tkfont.Font:
        if self._minimal_title_font is None:
            title_font = tkfont.nametofont("TkDefaultFont").copy()
            base_size = int(title_font.cget("size")) or 10
            title_font.configure(size=base_size * 2)
            self._minimal_title_font = title_font
        return self._minimal_title_font
