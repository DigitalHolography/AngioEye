import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

class ViewBuilderMixin:
    def _build_ui(self) -> None:
        self._build_menu()

        container = ttk.Frame(self, padding=10)
        container.pack(fill="both", expand=True)
        self.main_container = container

        self.minimal_view = ttk.Frame(container, padding=10)
        self.advanced_view = ttk.Frame(container, padding=10)

        self._build_minimal_view(self.minimal_view)
        self._build_advanced_view(self.advanced_view)

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

    def _build_minimal_view(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        parent.grid_anchor("center")

        content = ttk.Frame(parent, padding=(24, 24))
        content.grid(row=0, column=0)
        content.columnconfigure(0, minsize=420)
        self.minimal_content = content

        self.minimal_title_label = ttk.Label(
            content,
            text="AngioEye",
            font=self._get_minimal_title_font(),
        )
        self.minimal_title_label.grid(row=0, column=0, pady=(0, 10))

        minimal_logo = self._load_scaled_logo_image(max_width=360, max_height=144)
        if minimal_logo is not None:
            self._minimal_logo_image = minimal_logo
            self.minimal_logo_label = ttk.Label(content, image=self._minimal_logo_image)
            self.minimal_logo_label.grid(row=1, column=0, pady=(0, 18))

        self.minimal_browse_button = ttk.Button(
            content,
            text="Browse .h5, .holo, or zip archive",
            command=self.choose_batch_file,
        )
        self.minimal_browse_button.grid(row=2, column=0, pady=(0, 10))
        self.minimal_input_path_label = tk.Label(
            content,
            textvariable=self.minimal_input_path_var,
            bg=self._bg_color,
            fg=self._muted_fg,
            justify="center",
            wraplength=420,
        )
        self.minimal_input_path_label.grid(row=3, column=0, pady=(0, 8), sticky="ew")

        self.minimal_holo_status_label = tk.Label(
            content,
            textvariable=self.holo_status_var,
            bg=self._bg_color,
            fg="#d65f5f",
            justify="center",
            wraplength=420,
        )
        self.minimal_holo_status_label.grid(row=4, column=0, pady=(0, 8), sticky="ew")
        self.minimal_holo_status_label.grid_remove()
        self.minimal_holo_output_label = tk.Label(
            content,
            textvariable=self.holo_output_path_var,
            bg=self._bg_color,
            fg=self._text_fg,
            justify="center",
            wraplength=420,
        )
        self.minimal_holo_output_label.grid(row=5, column=0, pady=(0, 8), sticky="ew")
        self.minimal_holo_output_label.grid_remove()

        self.minimal_output_button = ttk.Button(
            content,
            text="Select output folder",
            command=self.choose_batch_output,
        )
        self.minimal_output_button.grid(row=6, column=0, pady=(0, 10))
        self.minimal_output_path_label = tk.Label(
            content,
            textvariable=self.minimal_output_path_var,
            bg=self._bg_color,
            fg=self._muted_fg,
            justify="center",
            wraplength=420,
        )
        self.minimal_output_path_label.grid(row=7, column=0, pady=(0, 6), sticky="ew")
        self.minimal_output_name_label = tk.Label(
            content,
            textvariable=self.minimal_output_name_var,
            bg=self._bg_color,
            fg=self._text_fg,
            justify="center",
            wraplength=420,
        )
        self.minimal_output_name_label.grid(row=8, column=0, pady=(0, 18), sticky="ew")

        self.minimal_run_button = ttk.Button(
            content, text="Run", command=self.run_batch
        )
        self.minimal_run_button.grid(row=9, column=0, pady=(0, 18))

        self.minimal_progress = ttk.Progressbar(
            content,
            orient="horizontal",
            mode="determinate",
            maximum=100,
            variable=self.batch_progress_var,
            length=340,
            style=self._progress_primary_style,
        )
        self.minimal_progress.grid(row=10, column=0, sticky="ew")
        self.minimal_status_label = tk.Label(
            content,
            textvariable=self.minimal_status_var,
            bg=self._bg_color,
            fg=self._text_fg,
            justify="center",
            wraplength=420,
        )
        self.minimal_status_label.grid(row=11, column=0, pady=(8, 0), sticky="ew")

    def _get_minimal_title_font(self) -> tkfont.Font:
        if self._minimal_title_font is None:
            title_font = tkfont.nametofont("TkDefaultFont").copy()
            base_size = int(title_font.cget("size")) or 10
            title_font.configure(size=base_size * 2)
            self._minimal_title_font = title_font
        return self._minimal_title_font

    def _build_advanced_view(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(parent, style="Advanced.TNotebook")
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self.batch_tab = ttk.Frame(self.notebook, padding=(18, 14, 14, 14))
        self.pipeline_library_tab = ttk.Frame(self.notebook, padding=10)
        self.postprocess_library_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.batch_tab, text="Run")
        self.notebook.add(self.pipeline_library_tab, text="Pipeline Library")
        self.notebook.add(self.postprocess_library_tab, text="Postprocess Library")

        self._build_batch_tab(self.batch_tab)
        self._build_pipeline_library_tab(self.pipeline_library_tab)
        self._build_postprocess_library_tab(self.postprocess_library_tab)

    def _build_batch_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)

        self.advanced_input_frame = ttk.Frame(parent)
        self.advanced_input_frame.grid(row=0, column=0, sticky="ew")
        self.advanced_input_frame.columnconfigure(
            0, minsize=self._ADVANCED_FORM_LABEL_WIDTH
        )
        self.advanced_input_frame.columnconfigure(1, weight=1)

        ttk.Label(self.advanced_input_frame, text="Input").grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 16),
        )
        input_entry = ttk.Entry(
            self.advanced_input_frame, textvariable=self.batch_input_var
        )
        input_entry.grid(row=0, column=1, sticky="ew", padx=(0, 4))
        input_btn_frame = ttk.Frame(self.advanced_input_frame)
        input_btn_frame.grid(row=0, column=2, sticky="w")
        ttk.Button(
            input_btn_frame, text="Browse folder", command=self.choose_batch_folder
        ).pack(side="left")
        ttk.Button(
            input_btn_frame, text="Browse file", command=self.choose_batch_file
        ).pack(side="left", padx=(4, 0))

        self.advanced_holo_status_label = tk.Label(
            self.advanced_input_frame,
            textvariable=self.holo_status_var,
            bg=self._bg_color,
            fg="#d65f5f",
            justify="left",
            anchor="w",
        )
        self.advanced_holo_status_label.grid(
            row=1, column=1, columnspan=2, sticky="ew", pady=(8, 0)
        )
        self.advanced_holo_status_label.grid_remove()
        self.advanced_holo_output_label = tk.Label(
            self.advanced_input_frame,
            textvariable=self.holo_output_path_var,
            bg=self._bg_color,
            fg=self._text_fg,
            justify="left",
            anchor="w",
        )
        self.advanced_holo_output_label.grid(
            row=2, column=1, columnspan=2, sticky="ew", pady=(4, 10)
        )
        self.advanced_holo_output_label.grid_remove()

        ttk.Label(self.advanced_input_frame, text="Output").grid(
            row=3,
            column=0,
            sticky="w",
            padx=(0, 16),
            pady=(10, 0),
        )
        batch_output_entry = ttk.Entry(
            self.advanced_input_frame, textvariable=self.batch_output_var
        )
        batch_output_entry.grid(
            row=3, column=1, sticky="ew", padx=(0, 4), pady=(10, 0)
        )
        ttk.Button(
            self.advanced_input_frame,
            text="Browse",
            command=self.choose_batch_output,
        ).grid(row=3, column=2, sticky="w", pady=(10, 0))

        run_btn = ttk.Button(
            self.advanced_input_frame, text="Run", command=self.run_batch
        )
        run_btn.grid(
            row=4,
            column=0,
            sticky="w",
            padx=(0, 16),
            pady=(12, 0),
        )

        persist_eyeflow_data_btn = ttk.Checkbutton(
            self.advanced_input_frame,
            text="Persist Eyeflow Data",
            variable=self._persist_eyeflow_data,
            command=self._persist_trim_h5source,
        )
        persist_eyeflow_data_btn.grid(row=4, column=1, sticky="w", pady=(12, 0))

        ttk.Label(parent, text="BatchLog").grid(
            row=1, column=0, sticky="w", pady=(16, 4)
        )
        batch_output_frame = ttk.Frame(parent)
        batch_output_frame.grid(row=2, column=0, sticky="nsew")
        batch_output_frame.columnconfigure(0, weight=1)
        batch_output_frame.rowconfigure(0, weight=1)
        self.batch_output = tk.Text(
            batch_output_frame,
            height=14,
            state="disabled",
            bg=self._text_bg,
            fg=self._text_fg,
            insertbackground=self._text_fg,
        )
        batch_output_scroll = ttk.Scrollbar(
            batch_output_frame, orient="vertical", command=self.batch_output.yview
        )
        self.batch_output.configure(yscrollcommand=batch_output_scroll.set)
        self.batch_output.grid(row=0, column=0, sticky="nsew")
        batch_output_scroll.grid(row=0, column=1, sticky="ns")
