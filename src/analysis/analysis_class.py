import os
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from colors.color_class import col
from h5.h5_helper import get_dimension_idx
from logger.logger_class import Logger


class AnalysisClass:
    def __init__(self):
        pass

    def generate_df(
        filename: Path,
        WomersleyPath: str = "/Array3D_FrequencyAnalysis/Womersley",
        percentiles: list[(int, int)] = [(10, 90), (25, 75), (40, 60)],
    ):
        """
        Analyzes H5 data and separates it into DataFrames based on the folder
        structure found.

        Returns:
            dict: { "Folder": pd.DataFrame, ... }
        """

        raw_buckets = {}

        def _get_harmonic_axis_index(dset: h5py.Dataset) -> int:
            return get_dimension_idx(dset, "harmonic")

        def _visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                parts = name.split("/")

                if len(parts) >= 3:
                    vessel_type = parts[0]
                    wall_code = parts[1]
                    var_name = parts[-1]

                    bucket_key = wall_code

                    if bucket_key not in raw_buckets:
                        raw_buckets[bucket_key] = {}

                    category = f"{vessel_type}_{bucket_key}"

                    if "RnR0_complex_" in var_name:
                        var_name = var_name.replace("RnR0_complex_", "RnR0_")

                    data = obj[:]
                    h_idx = _get_harmonic_axis_index(obj)

                    target_dict = raw_buckets[bucket_key]

                    if h_idx != -1:
                        for h in range(data.shape[h_idx]):
                            key = f"{category}_{var_name}_H{h + 1}"
                            target_dict[key] = np.take(data, h, axis=h_idx)
                    elif data.ndim == 3:
                        for h in range(data.shape[2]):
                            key = f"{category}_{var_name}_H{h + 1}"
                            target_dict[key] = data[:, :, h]
                    else:
                        key = f"{category}_{var_name}_H0"
                        target_dict[key] = data

        try:
            with h5py.File(filename, "r") as f:
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
                                stat_entry[f"Median{suffix}"] = np.median(
                                    subset
                                )
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
