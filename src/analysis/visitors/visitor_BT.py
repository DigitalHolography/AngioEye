from typing import Callable

import h5py
import numpy as np

from h5.h5_helper import get_dimension_idx


def _get_harmonic_axis_index(dset: h5py.Dataset) -> int:
    return get_dimension_idx(dset, "harmonic")


def visitor_two_depth() -> tuple[Callable[[str, object], None], dict]:
    """Returns a visitor and the dictionnary with data results

    It will Handle values stored inside 2 levels of depth.
    Usually for VesselType/FitName/metric

    Returns:
        tuple[Callable[[str, object], None], dict]: _description_
    """
    raw_buckets = {}

    def _visitor(name: str, obj) -> None:
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

    return _visitor, raw_buckets
