from src.utils.colors.colorClass import col

import os
import h5py

def inspect_h5_structure(file_path):
    """
    Prints the hierarchy of an HDF5 file with indentation.
    Shows shapes and data types for datasets.
    """
    title_str = f"Structure of: {file_path}"
    print(f"+-{"-" * (len(title_str))}-+")
    print(f"| {col.PUR}{title_str}{col.RES} |")
    print(f"+-{"-" * (len(title_str))}-+")

    try:
        with h5py.File(file_path, 'r') as f:
            
            def print_node_info(name, obj):
                depth = name.count('/')
                indent = '    ' * depth
                
                item_name = name.split('/')[-1]

                if isinstance(obj, h5py.Group):
                    print(f"{indent}📂 {col.CYA}{item_name}/{col.RES}")
                
                elif isinstance(obj, h5py.Dataset):
                    p_str   = f"{indent}📄 {item_name}"
                    shape   = f"{col.PUR}Shape:{col.RES} {obj.shape}"
                    type    = f"{col.PUR}Type:{col.RES} {obj.dtype}"
                    print(f"{p_str:<70}  [{shape:<28} | {type:<22}]")

            f.visititems(print_node_info)
            
    except FileNotFoundError:
        print("Error: File not found.")
    except Exception as e:
        print(f"Error: {e}")
