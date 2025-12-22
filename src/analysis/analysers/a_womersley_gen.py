import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from analysis.visitors.visitor_BT import visitor_two_depth
from logger.logger_class import Logger


def a_womersley_gen(
    file_path: Path,
    WomersleyPath: str = "/Array3D_FrequencyAnalysis/Womersley",
    percentiles: list[tuple[int, int]] = [(10, 90), (25, 75), (40, 60)],
) -> dict:
    """
    Analyzes H5 data and separates it into DataFrames based on the folder
    structure found.

    Returns:
        dict: { "Folder": pd.DataFrame, ... }
    """

    _visitor, raw_buckets = visitor_two_depth()

    try:
        with h5py.File(file_path, "r") as f:
            if WomersleyPath not in f:
                Logger.error(f"Group {WomersleyPath} not found.")
                return {}
            f[WomersleyPath].visititems(_visitor)
    except Exception as e:
        Logger.error(f"File error: {e}")
        return {}

    final_dfs = {}

    for wall_type, raw_slices in raw_buckets.items():
        if not raw_slices:
            continue

        processed_data = {}
        processed_keys = set()

        for key, data in raw_slices.items():
            if "_real_H" in key:
                imag_key = key.replace("_real_", "_imag_")
                if imag_key in raw_slices:
                    base_key = key.replace("_real_", "_")
                    complex_arr = data + 1j * raw_slices[imag_key]

                    processed_data[
                        re.sub(r"_H(\d+)$", r"_abs_H\1", base_key)
                    ] = np.abs(complex_arr)
                    processed_data[
                        re.sub(r"_H(\d+)$", r"_arg_H\1", base_key)
                    ] = np.angle(complex_arr)

                    processed_keys.add(key)
                    processed_keys.add(imag_key)

        for key, val in raw_slices.items():
            if key not in processed_keys:
                processed_data[key] = val

        stats_list = []
        percentile_pairs = percentiles if percentiles else []

        for name, data in processed_data.items():
            if np.issubdtype(data.dtype, np.number):
                flat = data.flatten()
                flat = flat[~np.isnan(flat)]

                if len(flat) > 0:
                    stat_entry = {
                        "Metric": re.sub(r"_H\d+$", "", name),
                        "Harmonic": int(name.split("_H")[-1])
                        if "_H" in name
                        else 0,
                        "NbSegments": len(flat),
                        "Min": np.min(flat),
                        "Max": np.max(flat),
                        "Average": np.mean(flat),
                        "Median": np.median(flat),
                        "StdDev": np.std(flat),
                    }

                    for p_low, p_high in percentile_pairs:
                        pl_val = np.percentile(flat, p_low)
                        ph_val = np.percentile(flat, p_high)

                        stat_entry[f"P{p_low}"] = pl_val
                        stat_entry[f"P{p_high}"] = ph_val

                        subset = flat[(flat >= pl_val) & (flat <= ph_val)]
                        suffix = f"_P{p_low}P{p_high}"

                        if len(subset) > 0:
                            stat_entry[f"Average{suffix}"] = np.mean(subset)
                            stat_entry[f"Median{suffix}"] = np.median(subset)
                            stat_entry[f"StdDev{suffix}"] = np.std(subset)
                        else:
                            stat_entry[f"Average{suffix}"] = np.nan
                            stat_entry[f"Median{suffix}"] = np.nan
                            stat_entry[f"StdDev{suffix}"] = np.nan

                    stats_list.append(stat_entry)

        df = pd.DataFrame(stats_list)
        if not df.empty:
            df = df.sort_values(by=["Metric", "Harmonic"]).reset_index(
                drop=True
            )
            final_dfs[wall_type] = df

    return final_dfs
