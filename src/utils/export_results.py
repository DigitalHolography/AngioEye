import os
from pathlib import Path

from logger.logger_class import Logger


def save_womersley_results(
    dfs_dict: dict, subname: str | None = None, output_dir: Path = "."
) -> None:
    """
    Saves the dictionary of DataFrames to separate CSV files.
    """
    if not dfs_dict:
        Logger.warn("No data to save.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for name, df in dfs_dict.items():
        file_name = f"{name}_{subname}.csv" if subname else f"{name}.csv"
        file_path = os.path.join(output_dir, file_name)

        try:
            df.to_csv(file_path, index=False)
            Logger.info(f"Saved: {file_path}")
        except Exception as e:
            Logger.error(f"Could not save {file_name}: {e}")
