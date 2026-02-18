import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import h5py
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


class H5Viewer(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("HDF5 Viewer")
        self.geometry("1200x800")
        self.h5_file: h5py.File | None = None
        self.current_dataset: h5py.Dataset | None = None
        self.current_dataset_path: str | None = None
        self.axis_label_to_index: dict[str, int] = {}
        self.slider_vars: dict[int, tuple[tk.IntVar, ttk.Label]] = {}
        self.colorbar = None

        self._build_ui()
        self._show_placeholder("Load a .h5 file to get started")

    def _build_ui(self) -> None:
        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(1, weight=1)

        top_bar = ttk.Frame(container, padding=8)
        top_bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        open_btn = ttk.Button(top_bar, text="Open .h5 file", command=self.open_file)
        open_btn.pack(side="left")
        self.file_label = ttk.Label(top_bar, text="No file loaded", wraplength=700)
        self.file_label.pack(side="left", padx=8)

        sidebar = ttk.Frame(container, padding=8)
        sidebar.grid(row=1, column=0, sticky="ns")
        sidebar.rowconfigure(4, weight=1)

        ttk.Label(sidebar, text="Datasets").grid(row=0, column=0, sticky="w")
        tree_frame = ttk.Frame(sidebar)
        tree_frame.grid(row=1, column=0, sticky="nsew", pady=(4, 8))
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)

        self.tree = ttk.Treeview(
            tree_frame, columns=("path",), show="tree", selectmode="browse"
        )
        tree_scroll = ttk.Scrollbar(
            tree_frame, orient="vertical", command=self.tree.yview
        )
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        tree_scroll.grid(row=0, column=1, sticky="ns")
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

        self.info_label = ttk.Label(sidebar, text="No dataset selected", wraplength=260)
        self.info_label.grid(row=2, column=0, sticky="ew")

        axis_frame = ttk.LabelFrame(sidebar, text="Axes", padding=8)
        axis_frame.grid(row=3, column=0, sticky="ew", pady=(8, 4))
        axis_frame.columnconfigure(1, weight=1)

        self.x_axis_var = tk.StringVar()
        self.y_axis_var = tk.StringVar()

        ttk.Label(axis_frame, text="X axis").grid(row=0, column=0, sticky="w")
        self.x_combo = ttk.Combobox(
            axis_frame, textvariable=self.x_axis_var, state="readonly", width=24
        )
        self.x_combo.grid(row=0, column=1, sticky="ew", padx=4, pady=2)

        ttk.Label(axis_frame, text="Y axis").grid(row=1, column=0, sticky="w")
        self.y_combo = ttk.Combobox(
            axis_frame, textvariable=self.y_axis_var, state="readonly", width=24
        )
        self.y_combo.grid(row=1, column=1, sticky="ew", padx=4, pady=2)

        self.x_combo.bind("<<ComboboxSelected>>", self.on_axis_change)
        self.y_combo.bind("<<ComboboxSelected>>", self.on_axis_change)

        self.slider_frame = ttk.LabelFrame(
            sidebar, text="Other axes sliders", padding=8
        )
        self.slider_frame.grid(row=4, column=0, sticky="nsew", pady=(8, 8))
        self.slider_frame.columnconfigure(1, weight=1)

        plot_frame = ttk.Frame(container, padding=8)
        plot_frame.grid(row=1, column=1, sticky="nsew")
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(1, weight=1)

        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=0, column=0, sticky="ew")
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")

    def _axis_label(self, axis: int) -> str:
        size = (
            self.current_dataset.shape[axis]
            if self.current_dataset is not None
            else "?"
        )
        return f"Dim {axis} (size {size})"

    def _selected_axis(self, label: str) -> int | None:
        if label == "(none)":
            return None
        return self.axis_label_to_index.get(label)

    def _show_placeholder(self, message: str) -> None:
        if self.colorbar is not None:
            try:
                self.colorbar.remove()
            except Exception:
                pass
            self.colorbar = None
        self.ax.clear()
        self.ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
        self.ax.axis("off")
        self.canvas.draw_idle()

    def open_file(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("HDF5", "*.h5 *.hdf5"), ("All files", "*.*")],
            initialdir=os.path.abspath("h5_example"),
        )
        if not path:
            return
        try:
            if self.h5_file is not None:
                self.h5_file.close()
            self.h5_file = h5py.File(path, "r")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", f"Cannot open {path}: {exc}")
            return

        self.file_label.config(text=path)
        self._populate_tree()
        self.current_dataset = None
        self.current_dataset_path = None
        self._show_placeholder("Select a dataset in the tree")

    def _populate_tree(self) -> None:
        self.tree.delete(*self.tree.get_children())
        if self.h5_file is None:
            return
        root_id = self.tree.insert(
            "",
            "end",
            text=os.path.basename(self.h5_file.filename),
            open=True,
            values=("/"),
        )
        self._add_tree_items(root_id, self.h5_file)

    def _add_tree_items(self, parent: str, obj: h5py.Group) -> None:
        for key, item in obj.items():
            if isinstance(item, h5py.Group):
                node_id = self.tree.insert(
                    parent,
                    "end",
                    text=key,
                    open=False,
                    values=(item.name,),
                    tags=("group",),
                )
                self._add_tree_items(node_id, item)
            else:
                label = f"{key} {item.shape}"
                self.tree.insert(
                    parent, "end", text=label, values=(item.name,), tags=("dataset",)
                )

    def on_tree_select(self, _event: tk.Event) -> None:
        item_id = self.tree.focus()
        if not item_id:
            return
        tags = self.tree.item(item_id, "tags")
        if "dataset" not in tags:
            return
        path = self.tree.item(item_id, "values")[0]
        self.load_dataset(path)

    def load_dataset(self, path: str) -> None:
        if self.h5_file is None:
            return
        dataset = self.h5_file[path]
        self.current_dataset = dataset
        self.current_dataset_path = path
        self.info_label.config(
            text=f"{path}\nshape={dataset.shape} dtype={dataset.dtype}"
        )

        dims = dataset.ndim
        labels = [self._axis_label(i) for i in range(dims)]
        self.axis_label_to_index = {label: idx for idx, label in enumerate(labels)}

        if dims == 0:
            self.x_combo["values"] = ()
            self.y_combo["values"] = ()
            self.x_axis_var.set("")
            self.y_axis_var.set("")
            self._clear_sliders()
            self.update_plot()
            return

        self.x_combo["values"] = labels
        y_values = labels.copy() if dims > 1 else ["(none)"]
        self.y_combo["values"] = y_values

        if dims >= 2:
            x_default = labels[1] if len(labels) > 1 else labels[0]
            y_default = labels[0]
        else:
            x_default = labels[0]
            y_default = "(none)"

        self.x_axis_var.set(x_default)
        self.y_axis_var.set(y_default)
        self.refresh_sliders()
        self.update_plot()

    def _clear_sliders(self) -> None:
        for child in self.slider_frame.winfo_children():
            child.destroy()
        self.slider_vars = {}

    def refresh_sliders(self) -> None:
        self._clear_sliders()
        if self.current_dataset is None:
            return
        dims = self.current_dataset.ndim
        if dims <= 2:
            return
        x_axis = self._selected_axis(self.x_axis_var.get())
        y_axis = self._selected_axis(self.y_axis_var.get())
        for axis, size in enumerate(self.current_dataset.shape):
            if axis == x_axis or axis == y_axis:
                continue
            row = len(self.slider_vars)
            ttk.Label(self.slider_frame, text=self._axis_label(axis)).grid(
                row=row, column=0, sticky="w"
            )
            var = tk.IntVar(value=0)
            scale = tk.Scale(
                self.slider_frame,
                from_=0,
                to=max(size - 1, 0),
                orient="horizontal",
                resolution=1,
                variable=var,
                command=lambda val, ax=axis: self.on_slider_change(ax, val),
            )
            scale.grid(row=row, column=1, sticky="ew", padx=6, pady=2)
            value_label = ttk.Label(self.slider_frame, text=f"0 / {max(size - 1, 0)}")
            value_label.grid(row=row, column=2, sticky="e")
            self.slider_vars[axis] = (var, value_label)

    def on_axis_change(self, _event: tk.Event | None = None) -> None:
        if self.current_dataset is None:
            return
        self.refresh_sliders()
        self.update_plot()

    def on_slider_change(self, axis: int, value: str) -> None:
        if axis not in self.slider_vars:
            return
        var, label = self.slider_vars[axis]
        var.set(int(float(value)))
        label.config(text=f"{var.get()} / {self.current_dataset.shape[axis] - 1}")
        self.update_plot()

    def update_plot(self) -> None:
        if self.colorbar is not None:
            try:
                self.colorbar.remove()
            except Exception:
                pass
            self.colorbar = None
        self.ax.clear()

        if self.current_dataset is None:
            self._show_placeholder("Select a dataset")
            return

        ds = self.current_dataset
        dims = ds.ndim
        x_axis = self._selected_axis(self.x_axis_var.get())
        y_axis = self._selected_axis(self.y_axis_var.get())
        slider_indices = {
            axis: int(var.get()) for axis, (var, _) in self.slider_vars.items()
        }

        if dims == 0:
            value = ds[()]
            self.ax.text(0.5, 0.5, str(value), ha="center", va="center", fontsize=12)
            self.ax.axis("off")
        elif dims == 1 or y_axis is None or x_axis is None:
            data = np.asarray(ds[...]).squeeze()
            self.ax.plot(np.arange(data.shape[0]), data)
            self.ax.set_xlabel(self._axis_label(x_axis if x_axis is not None else 0))
            self.ax.set_ylabel(ds.name)
        else:
            if x_axis == y_axis:
                self._show_placeholder("Choose two different axes")
                return
            slices: list[int | slice] = []
            for axis in range(dims):
                if axis == x_axis or axis == y_axis:
                    slices.append(slice(None))
                else:
                    slices.append(slider_indices.get(axis, 0))
            data = np.asarray(ds[tuple(slices)])
            if data.ndim == 1:
                self.ax.plot(np.arange(data.shape[0]), data)
                self.ax.set_xlabel(self._axis_label(x_axis))
                self.ax.set_ylabel(ds.name)
            else:
                kept_axes = [
                    idx for idx, slc in enumerate(slices) if isinstance(slc, slice)
                ]
                order = [kept_axes.index(y_axis), kept_axes.index(x_axis)]
                if order != [0, 1]:
                    data = np.transpose(data, order)
                im = self.ax.imshow(data, aspect="auto", origin="lower")
                self.ax.set_xlabel(self._axis_label(x_axis))
                self.ax.set_ylabel(self._axis_label(y_axis))
                self.ax.set_title(ds.name)
                self.colorbar = self.figure.colorbar(im, ax=self.ax, pad=0.02)

        self.canvas.draw_idle()


if __name__ == "__main__":
    app = H5Viewer()
    app.mainloop()
