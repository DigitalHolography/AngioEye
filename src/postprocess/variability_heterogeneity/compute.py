import re
from collections import defaultdict
from pathlib import Path
from tkinter import Tk, filedialog

import numpy as np

from input_output.hdf5_io import MetricsTree, iter_h5_arrays
from math_utils import compute_axis_statistics, nanmedian_or_nan

from ..core.grouped_batch import iter_grouped_h5_files_in_zip
from .constants import (
    CONTROL_GROUP_PATTERNS,
    EPS,
    INPUT_METRICS,
    SEGMENT_METRIC_FOLDERS,
    SEGMENT_MODE,
)


def choose_zip():
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilename(filetypes=[("ZIP", "*.zip")])


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(name)).strip("_")


def extract_sort_key(filename):
    name = Path(filename).name
    date_match = re.search(r"(\d{6})", name)
    date = int(date_match.group(1)) if date_match else 0
    hd_match = re.search(r"_(\d+)_HD", name)
    hd_index = int(hd_match.group(1)) if hd_match else 0
    return date, hd_index


def iter_segment_metrics(
    h5_path,
    metric_names,
    mode=SEGMENT_MODE,
    *,
    metric_folders=SEGMENT_METRIC_FOLDERS,
):
    """Yield available 3D metric arrays while opening the HDF5 file only once."""
    group_paths = [f"{folder.rstrip('/')}/{mode}" for folder in metric_folders]
    yield from iter_h5_arrays(
        h5_path,
        metric_names,
        group_paths=group_paths,
        dtype=float,
        ndim=3,
    )


def compute_file_higher_metrics_from_segment_array(arr, eps=EPS):
    """Reduce a beat/branch/disk metric array to spatial and temporal summaries."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 3:
        return None

    segment_count = arr.shape[1] * arr.shape[2]
    beat_by_segment = arr.reshape(arr.shape[0], segment_count)
    spatial = compute_axis_statistics(beat_by_segment, axis=1, eps=eps)
    temporal = compute_axis_statistics(beat_by_segment, axis=0, eps=eps)

    return {
        "MED_seg_medbeat": nanmedian_or_nan(spatial["median"]),
        "STD_seg_medbeat": nanmedian_or_nan(spatial["std"]),
        "IQR_seg_medbeat": nanmedian_or_nan(spatial["iqr"]),
        "MAD_seg_medbeat": nanmedian_or_nan(spatial["mad"]),
        "CV_seg_medbeat": nanmedian_or_nan(spatial["cv"]),
        "STD_beat_medseg": nanmedian_or_nan(temporal["std"]),
        "IQR_beat_medseg": nanmedian_or_nan(temporal["iqr"]),
        "MAD_beat_medseg": nanmedian_or_nan(temporal["mad"]),
        "CV_beat_medseg": nanmedian_or_nan(temporal["cv"]),
    }


def compute_file_higher_metric_blocks(
    file_path,
    metrics=INPUT_METRICS,
    mode=SEGMENT_MODE,
):
    blocks = {}
    for metric_name, arr in iter_segment_metrics(file_path, metrics, mode=mode):
        high = compute_file_higher_metrics_from_segment_array(arr)
        if high is not None:
            blocks[metric_name] = high
    return blocks


def add_file_blocks_to_results(results, group_name, blocks):
    for metric_name, high in blocks.items():
        for high_name, value in high.items():
            results[group_name][metric_name][high_name].append(value)


def variability_tree_from_blocks(blocks):
    metrics = {
        f"{high_name}/{metric_name}": np.asarray(value, dtype=float)
        for metric_name, high in blocks.items()
        for high_name, value in high.items()
    }
    if not metrics:
        return None
    return MetricsTree(
        name="Variability",
        metrics=metrics,
        attrs={"kind": "postprocess", "source": "segment_metrics"},
    )


def analyze_zip(zip_path, metrics=INPUT_METRICS, mode=SEGMENT_MODE):
    """Collect higher-order metrics by cohort from a ZIP archive."""
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for grouped_file in iter_grouped_h5_files_in_zip(
        zip_path,
        sort_key=lambda record: (
            record.group_name,
            extract_sort_key(record.file_name),
        ),
    ):
        blocks = compute_file_higher_metric_blocks(
            grouped_file.file_path,
            metrics=metrics,
            mode=mode,
        )
        add_file_blocks_to_results(results, grouped_file.group_name, blocks)
    return results


def normalize_group_name(group_name):
    return re.sub(r"[^a-z0-9]+", "_", str(group_name).strip().lower()).strip("_")


def is_control_group(group_name, patterns=CONTROL_GROUP_PATTERNS):
    normalized = normalize_group_name(group_name)
    return any(
        re.match(pattern, normalized, flags=re.IGNORECASE) for pattern in patterns
    )


def find_control_group(results):
    groups = list(results.keys())
    candidates = [group for group in groups if is_control_group(group)]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        priority = {"control": 0, "controle": 1, "ctrl": 2, "ctl": 3}
        return min(
            candidates,
            key=lambda group: priority.get(normalize_group_name(group), 100),
        )
    raise ValueError(
        "No control group found. Expected a group folder named like: "
        "control, controle, ctrl, ctl, healthy_control. "
        f"Groups found: {groups}"
    )
