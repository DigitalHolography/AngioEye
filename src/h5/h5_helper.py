import os
from pathlib import Path

import h5py

from src.colors.color_class import col
from src.logger.logger_class import Logger


def inspect_h5_structure(file_path):
    """
    Prints the hierarchy of an HDF5 file with indentation.
    Shows shapes and data types for datasets.
    """
    title_str = f"Structure of: {file_path}"
    sep = "-" * (len(title_str))
    print(f"+-{sep}-+")
    print(f"| {col.PUR}{title_str}{col.RES} |")
    print(f"+-{sep}-+")

    try:
        with h5py.File(file_path, "r") as f:

            def print_node_info(name, obj):
                depth = name.count("/")
                indent = "    " * depth

                item_name = name.split("/")[-1]

                if isinstance(obj, h5py.Group):
                    print(f"{indent}📂 {col.CYA}{item_name}/{col.RES}")

                elif isinstance(obj, h5py.Dataset):
                    p_str = f"{indent}📄 {item_name}"
                    shape = f"{col.PUR}Shape:{col.RES} {obj.shape}"
                    type = f"{col.PUR}Type:{col.RES} {obj.dtype}"
                    print(f"{p_str:<70}  [{shape:<28} | {type:<22}]")

            f.visititems(print_node_info)

    except FileNotFoundError:
        Logger.error("File not found.", "H5")
    except Exception as e:
        Logger.error(f"Error: {e}", "H5")


def get_h5_value(
    filename: Path, key_path: str, default=None, doGroup: bool = False
):
    """
    Safely retrieves a value from an H5 file.

    Args:
        filename (Path): Path to the .h5 file.
        key_path (str) : The internal path (e.g., "group1/dataset1").
        default  (any) : Value to return if key or file is missing.
        doGroup  (bool): Allows to treat folders

    Returns:
        NumPy Array (if dataset), List of keys (if group), or default value.
    """
    if not os.path.exists(filename):
        Logger.warn(f"File '{filename}' does not exist.", "H5")
        return default

    def recursively_load(h5_object):
        data_dict = {}

        for key, item in h5_object.items():
            if isinstance(item, h5py.Dataset):
                data_dict[key] = item[:]

            elif isinstance(item, h5py.Group):
                data_dict[key] = recursively_load(item)

        return data_dict

    try:
        with h5py.File(filename, "r") as f:
            if key_path not in f:
                Logger.warn(f"Key '{key_path}' not found in file.", "H5")
                return default

            obj = f[key_path]

            if isinstance(obj, h5py.Dataset):
                # Copy data
                return obj[:]

            if not doGroup:
                Logger.error(
                    f"The key_path is a Group: {col.YEL}{key_path}", "H5"
                )
                return []
            return recursively_load(obj)

    except Exception as e:
        Logger.error(f"Error reading file: {e}", "H5")
        return default

    return default
