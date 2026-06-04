import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

from .controllers import AdvancedViewController, MinimalViewController
from .pipeline_library import PipelineLibraryTab
from .postprocess_library import PostprocessLibraryTab


class MinimalView(ttk.Frame):
    def __init__(self, parent: tk.Misc, controller) -> None:
        super().__init__(parent, padding=10)
        self.controller = controller
        self._build()

    def _build(self) -> None:
        controller = self.controller
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.grid_anchor("center")

        content = ttk.Frame(self, padding=(24, 24))
        content.grid(row=0, column=0)
        content.columnconfigure(0, minsize=420)
        controller.set_widget("minimal_content", content)

        minimal_title_label = ttk.Label(
            content,
            text="AngioEye",
            font=controller.title_font(),
        )
        minimal_title_label.grid(row=0, column=0, pady=(0, 10))
        controller.set_widget("minimal_title_label", minimal_title_label)

        minimal_logo = controller.load_logo()
        if minimal_logo is not None:
            controller.keep_logo(minimal_logo)
            minimal_logo_label = ttk.Label(content, image=minimal_logo)
            minimal_logo_label.grid(row=1, column=0, pady=(0, 18))
            controller.set_widget("minimal_logo_label", minimal_logo_label)

        minimal_browse_button = ttk.Button(
            content,
            text="Browse .h5, .holo, or zip archive",
            command=controller.choose_input,
        )
        minimal_browse_button.grid(row=2, column=0, pady=(0, 10))
        controller.set_widget("minimal_browse_button", minimal_browse_button)
        minimal_input_path_label = tk.Label(
            content,
            textvariable=controller.input_path_var,
            bg=controller.bg_color,
            fg=controller.muted_fg,
            justify="center",
            wraplength=420,
        )
        minimal_input_path_label.grid(row=3, column=0, pady=(0, 8), sticky="ew")
        controller.set_widget("minimal_input_path_label", minimal_input_path_label)

        minimal_holo_status_label = tk.Label(
            content,
            textvariable=controller.holo_status_var,
            bg=controller.bg_color,
            fg="#d65f5f",
            justify="center",
            wraplength=420,
        )
        minimal_holo_status_label.grid(row=4, column=0, pady=(0, 8), sticky="ew")
        minimal_holo_status_label.grid_remove()
        controller.set_widget("minimal_holo_status_label", minimal_holo_status_label)
        minimal_holo_output_label = tk.Label(
            content,
            textvariable=controller.holo_output_path_var,
            bg=controller.bg_color,
            fg=controller.text_fg,
            justify="center",
            wraplength=420,
        )
        minimal_holo_output_label.grid(row=5, column=0, pady=(0, 8), sticky="ew")
        minimal_holo_output_label.grid_remove()
        controller.set_widget("minimal_holo_output_label", minimal_holo_output_label)

        minimal_output_button = ttk.Button(
            content,
            text="Select output folder",
            command=controller.choose_output,
        )
        minimal_output_button.grid(row=6, column=0, pady=(0, 10))
        controller.set_widget("minimal_output_button", minimal_output_button)
        minimal_output_path_label = tk.Label(
            content,
            textvariable=controller.output_path_var,
            bg=controller.bg_color,
            fg=controller.muted_fg,
            justify="center",
            wraplength=420,
        )
        minimal_output_path_label.grid(row=7, column=0, pady=(0, 6), sticky="ew")
        controller.set_widget("minimal_output_path_label", minimal_output_path_label)
        minimal_output_name_label = tk.Label(
            content,
            textvariable=controller.output_name_var,
            bg=controller.bg_color,
            fg=controller.text_fg,
            justify="center",
            wraplength=420,
        )
        minimal_output_name_label.grid(row=8, column=0, pady=(0, 18), sticky="ew")
        controller.set_widget("minimal_output_name_label", minimal_output_name_label)

        minimal_run_button = ttk.Button(
            content, text="Run", command=controller.run
        )
        minimal_run_button.grid(row=9, column=0, pady=(0, 18))
        controller.set_widget("minimal_run_button", minimal_run_button)

        minimal_progress = ttk.Progressbar(
            content,
            orient="horizontal",
            mode="determinate",
            maximum=100,
            variable=controller.progress_var,
            length=340,
            style=controller.progress_style,
        )
        minimal_progress.grid(row=10, column=0, sticky="ew")
        controller.set_widget("minimal_progress", minimal_progress)
        minimal_status_label = tk.Label(
            content,
            textvariable=controller.status_var,
            bg=controller.bg_color,
            fg=controller.text_fg,
            justify="center",
            wraplength=420,
        )
        minimal_status_label.grid(row=11, column=0, pady=(8, 0), sticky="ew")
        controller.set_widget("minimal_status_label", minimal_status_label)


class RunTab(ttk.Frame):
    def __init__(self, parent: tk.Misc, controller) -> None:
        super().__init__(parent, padding=(18, 14, 14, 14))
        self.controller = controller
        self._build()

    def _build(self) -> None:
        controller = self.controller
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        advanced_input_frame = ttk.Frame(self)
        advanced_input_frame.grid(row=0, column=0, sticky="ew")
        advanced_input_frame.columnconfigure(
            0, minsize=controller.form_label_width
        )
        advanced_input_frame.columnconfigure(1, weight=1)
        controller.set_widget("advanced_input_frame", advanced_input_frame)

        ttk.Label(advanced_input_frame, text="Input").grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 16),
        )
        input_entry = ttk.Entry(
            advanced_input_frame, textvariable=controller.input_var
        )
        input_entry.grid(row=0, column=1, sticky="ew", padx=(0, 4))
        input_btn_frame = ttk.Frame(advanced_input_frame)
        input_btn_frame.grid(row=0, column=2, sticky="w")
        ttk.Button(
            input_btn_frame, text="Browse folder", command=controller.choose_folder
        ).pack(side="left")
        ttk.Button(
            input_btn_frame, text="Browse file", command=controller.choose_file
        ).pack(side="left", padx=(4, 0))

        advanced_holo_status_label = tk.Label(
            advanced_input_frame,
            textvariable=controller.holo_status_var,
            bg=controller.bg_color,
            fg="#d65f5f",
            justify="left",
            anchor="w",
        )
        advanced_holo_status_label.grid(
            row=1, column=1, columnspan=2, sticky="ew", pady=(8, 0)
        )
        advanced_holo_status_label.grid_remove()
        controller.set_widget("advanced_holo_status_label", advanced_holo_status_label)
        advanced_holo_output_label = tk.Label(
            advanced_input_frame,
            textvariable=controller.holo_output_path_var,
            bg=controller.bg_color,
            fg=controller.text_fg,
            justify="left",
            anchor="w",
        )
        advanced_holo_output_label.grid(
            row=2, column=1, columnspan=2, sticky="ew", pady=(4, 10)
        )
        advanced_holo_output_label.grid_remove()
        controller.set_widget("advanced_holo_output_label", advanced_holo_output_label)

        ttk.Label(advanced_input_frame, text="Output").grid(
            row=3,
            column=0,
            sticky="w",
            padx=(0, 16),
            pady=(10, 0),
        )
        batch_output_entry = ttk.Entry(
            advanced_input_frame, textvariable=controller.output_var
        )
        batch_output_entry.grid(
            row=3, column=1, sticky="ew", padx=(0, 4), pady=(10, 0)
        )
        ttk.Button(
            advanced_input_frame,
            text="Browse",
            command=controller.choose_output,
        ).grid(row=3, column=2, sticky="w", pady=(10, 0))

        run_btn = ttk.Button(
            advanced_input_frame, text="Run", command=controller.run
        )
        run_btn.grid(
            row=4,
            column=0,
            sticky="w",
            padx=(0, 16),
            pady=(12, 0),
        )

        persist_eyeflow_data_btn = ttk.Checkbutton(
            advanced_input_frame,
            text="Persist Eyeflow Data",
            variable=controller.persist_eyeflow_data_var,
            command=controller.persist_trim_h5source,
        )
        persist_eyeflow_data_btn.grid(row=4, column=1, sticky="w", pady=(12, 0))

        ttk.Label(self, text="BatchLog").grid(
            row=1, column=0, sticky="w", pady=(16, 4)
        )
        batch_output_frame = ttk.Frame(self)
        batch_output_frame.grid(row=2, column=0, sticky="nsew")
        batch_output_frame.columnconfigure(0, weight=1)
        batch_output_frame.rowconfigure(0, weight=1)
        batch_output = tk.Text(
            batch_output_frame,
            height=14,
            state="disabled",
            bg=controller.text_bg,
            fg=controller.text_fg,
            insertbackground=controller.text_fg,
        )
        batch_output_scroll = ttk.Scrollbar(
            batch_output_frame, orient="vertical", command=batch_output.yview
        )
        batch_output.configure(yscrollcommand=batch_output_scroll.set)
        batch_output.grid(row=0, column=0, sticky="nsew")
        batch_output_scroll.grid(row=0, column=1, sticky="ns")
        controller.set_widget("batch_output", batch_output)


class AdvancedView(ttk.Frame):
    def __init__(self, parent: tk.Misc, controller) -> None:
        super().__init__(parent, padding=10)
        self.controller = controller
        self._build()

    def _build(self) -> None:
        controller = self.controller
        app = controller.app
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        app.notebook = ttk.Notebook(self, style="Advanced.TNotebook")
        app.notebook.grid(row=0, column=0, sticky="nsew")

        app.batch_tab = RunTab(app.notebook, controller.create_run_controller())
        app.pipeline_library_tab = PipelineLibraryTab(
            app.notebook,
            controller.create_pipeline_library_controller(),
        )
        app.postprocess_library_tab = PostprocessLibraryTab(
            app.notebook,
            controller.create_postprocess_library_controller(),
        )
        app.notebook.add(app.batch_tab, text="Run")
        app.notebook.add(app.pipeline_library_tab, text="Pipeline Library")
        app.notebook.add(app.postprocess_library_tab, text="Postprocess Library")


class ViewBuilderMixin:
    def _build_ui(self) -> None:
        self._build_menu()

        container = ttk.Frame(self, padding=10)
        container.pack(fill="both", expand=True)
        self.main_container = container

        self.minimal_view = MinimalView(container, MinimalViewController(self))
        self.advanced_view = AdvancedView(container, AdvancedViewController(self))

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
