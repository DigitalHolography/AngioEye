from __future__ import annotations

import re
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_SRC))


def _iter_ef_files(input_dir: Path):
    input_dir = Path(input_dir)
    for path in sorted(input_dir.rglob("*.h5")):
        if path.is_file():
            yield path


def _load_source_waveform(path: Path):
    with h5py.File(path, "r") as h:
        v_per_beat = np.asarray(
            h["/Artery/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegment/value"],
            dtype=float,
        )
        beat_period = np.asarray(
            h["/Artery/VelocityPerBeat/beatPeriodSeconds/value"], dtype=float
        )
        if "/Meta/Time/value" in h:
            phase = np.asarray(h["/Meta/Time/value"], dtype=float).reshape(-1)
        else:
            phase = None

    if v_per_beat.ndim != 4:
        raise ValueError(f"Expected 4D segment waveform block, got {v_per_beat.shape}")

    n_t, n_beats, n_branches, n_radii = v_per_beat.shape
    if phase is None:
        phase = np.arange(n_t, dtype=float) / float(n_t)
    if phase.size != n_t:
        phase = _normalize_phase_axis(phase, n_t)

    if beat_period.ndim == 2 and beat_period.shape[0] == 1:
        beat_period = beat_period[0]
    if beat_period.shape[0] != n_beats:
        raise ValueError(
            f"Beat period length {beat_period.shape[0]} "
            f"does not match waveform beats {n_beats}"
        )

    return v_per_beat, phase, beat_period


def _make_svd_like_arrays(v_per_beat: np.ndarray):
    a = np.asarray(v_per_beat, dtype=float)
    a_bar = np.nanmean(a, axis=0)
    p_a = (a - a_bar[None, :, :, :]) ** 2

    return a, p_a


def _make_p_a_stack(a_stack: np.ndarray):
    a_stack = np.asarray(a_stack, dtype=float)
    a_bar = np.nanmean(a_stack, axis=1, keepdims=True)
    return (a_stack - a_bar) ** 2


def _infer_flicker_group_name(path: Path):
    stem = path.stem.lower()
    if "bl1" in stem:
        return "bl1"
    if "bl2" in stem:
        return "bl2"

    for token in stem.split("_"):
        try:
            index = int(token)
        except ValueError:
            continue
        if 0 <= index <= 7:
            return "bl1"
        if 8 <= index <= 19:
            return "flicker"
        if 20 <= index <= 27:
            return "bl2"

    return "flicker"


def _normalize_time_axis(waveform: np.ndarray, target_timepoints: int):
    waveform = np.asarray(waveform, dtype=float)
    if waveform.shape[0] == target_timepoints:
        return waveform

    source_axis = np.arange(waveform.shape[0], dtype=float)
    target_axis = np.linspace(0, waveform.shape[0] - 1, target_timepoints)
    interpolated = np.empty(
        (target_timepoints, waveform.shape[1], waveform.shape[2], waveform.shape[3]),
        dtype=float,
    )

    for beat in range(waveform.shape[1]):
        for segment in range(waveform.shape[2]):
            for branch in range(waveform.shape[3]):
                interpolated[:, beat, segment, branch] = np.interp(
                    target_axis,
                    source_axis,
                    waveform[:, beat, segment, branch],
                )

    return interpolated


def _normalize_beats(waveform: np.ndarray, target_beats: int):
    waveform = np.asarray(waveform, dtype=float)
    if waveform.shape[1] == target_beats:
        return waveform

    source_axis = np.arange(waveform.shape[1], dtype=float)
    target_axis = np.linspace(0, waveform.shape[1] - 1, target_beats)
    interpolated = np.empty(
        (waveform.shape[0], target_beats, waveform.shape[2], waveform.shape[3]),
        dtype=float,
    )

    for timepoint in range(waveform.shape[0]):
        for segment in range(waveform.shape[2]):
            for branch in range(waveform.shape[3]):
                interpolated[timepoint, :, segment, branch] = np.interp(
                    target_axis,
                    source_axis,
                    waveform[timepoint, :, segment, branch],
                )

    return interpolated


def _normalize_segment_axis(waveform: np.ndarray, target_segments: int):
    waveform = np.asarray(waveform, dtype=float)
    if waveform.ndim == 5:
        interpolated = np.empty(
            (
                waveform.shape[0],
                waveform.shape[1],
                waveform.shape[2],
                target_segments,
                waveform.shape[4],
            ),
            dtype=float,
        )
        for record in range(waveform.shape[0]):
            interpolated[record] = _normalize_segment_axis(
                waveform[record], target_segments
            )
        return interpolated

    if waveform.ndim != 4:
        raise ValueError(f"Expected 4D or 5D waveform block, got {waveform.shape}")

    if waveform.shape[2] == target_segments:
        return waveform

    source_axis = np.arange(waveform.shape[2], dtype=float)
    target_axis = np.linspace(0, waveform.shape[2] - 1, target_segments)
    interpolated = np.empty(
        (waveform.shape[0], waveform.shape[1], target_segments, waveform.shape[3]),
        dtype=float,
    )

    for timepoint in range(waveform.shape[0]):
        for beat in range(waveform.shape[1]):
            for branch in range(waveform.shape[3]):
                interpolated[timepoint, beat, :, branch] = np.interp(
                    target_axis,
                    source_axis,
                    waveform[timepoint, beat, :, branch],
                )

    return interpolated


def _normalize_radius_axis(waveform: np.ndarray, target_radii: int):
    waveform = np.asarray(waveform, dtype=float)
    if waveform.ndim == 5:
        interpolated = np.empty(
            (
                waveform.shape[0],
                waveform.shape[1],
                waveform.shape[2],
                waveform.shape[3],
                target_radii,
            ),
            dtype=float,
        )
        for record in range(waveform.shape[0]):
            interpolated[record] = _normalize_radius_axis(
                waveform[record], target_radii
            )
        return interpolated

    if waveform.ndim != 4:
        raise ValueError(f"Expected 4D or 5D waveform block, got {waveform.shape}")

    if waveform.shape[3] == target_radii:
        return waveform

    source_axis = np.arange(waveform.shape[3], dtype=float)
    target_axis = np.linspace(0, waveform.shape[3] - 1, target_radii)
    interpolated = np.empty(
        (waveform.shape[0], waveform.shape[1], waveform.shape[2], target_radii),
        dtype=float,
    )

    for timepoint in range(waveform.shape[0]):
        for beat in range(waveform.shape[1]):
            for segment in range(waveform.shape[2]):
                interpolated[timepoint, beat, segment, :] = np.interp(
                    target_axis,
                    source_axis,
                    waveform[timepoint, beat, segment, :],
                )

    return interpolated


def _normalize_array_axis(arr: np.ndarray, axis: int, target_size: int):
    arr = np.asarray(arr, dtype=float)
    if arr.shape[axis] == target_size:
        return arr

    moved = np.moveaxis(arr, axis, 0)
    source_axis = np.arange(moved.shape[0], dtype=float)
    target_axis = np.linspace(0, moved.shape[0] - 1, target_size)
    flat = moved.reshape(moved.shape[0], -1)
    out = np.empty((target_size, flat.shape[1]), dtype=float)
    for column in range(flat.shape[1]):
        out[:, column] = np.interp(target_axis, source_axis, flat[:, column])
    out = out.reshape((target_size, *moved.shape[1:]))
    return np.moveaxis(out, 0, axis)


def _normalize_waveform_axes(
    waveform: np.ndarray,
    target_timepoints: int,
    target_beats: int,
    target_segments: int,
    target_radii: int,
):
    waveform = _normalize_time_axis(waveform, target_timepoints)
    waveform = _normalize_beats(waveform, target_beats)
    waveform = _normalize_segment_axis(waveform, target_segments)
    waveform = _normalize_radius_axis(waveform, target_radii)
    return waveform


def _normalize_waveform_stack_axes(
    waveform_stack: np.ndarray,
    target_timepoints: int,
    target_beats: int,
    target_segments: int,
    target_radii: int,
):
    waveform_stack = np.asarray(waveform_stack, dtype=float)
    if waveform_stack.ndim != 5:
        raise ValueError(f"Expected 5D waveform stack, got {waveform_stack.shape}")
    return np.stack(
        [
            _normalize_waveform_axes(
                waveform,
                target_timepoints,
                target_beats,
                target_segments,
                target_radii,
            )
            for waveform in waveform_stack
        ],
        axis=0,
    )


def _normalize_valid_stack_axes(
    valid_stack: np.ndarray,
    target_beats: int,
    target_segments: int,
    target_radii: int,
):
    valid_stack = np.asarray(valid_stack, dtype=float)
    if valid_stack.ndim != 4:
        raise ValueError(f"Expected 4D validity stack, got {valid_stack.shape}")
    normalized = np.stack(
        [
            _normalize_array_axis(
                _normalize_array_axis(
                    _normalize_array_axis(valid, 0, target_beats),
                    1,
                    target_segments,
                ),
                2,
                target_radii,
            )
            for valid in valid_stack
        ],
        axis=0,
    )
    return normalized >= 0.5


def _normalize_phase_axis(phase: np.ndarray, target_timepoints: int):
    phase = np.asarray(phase, dtype=float).reshape(-1)
    if phase.size == target_timepoints:
        return phase
    source_axis = np.arange(phase.size, dtype=float)
    target_axis = np.linspace(0, phase.size - 1, target_timepoints)
    return np.interp(target_axis, source_axis, phase)


def _write_group_h5(
    output_dir: Path,
    prefix: str,
    a_stack: np.ndarray,
    p_a_stack: np.ndarray,
    valid_stack: np.ndarray,
    names: np.ndarray,
    phase: np.ndarray,
):
    with h5py.File(output_dir / f"{prefix}_asymmetry_A.h5", "w") as h:
        h.create_group("Asymmetry")
        h["Asymmetry"].create_dataset("A/value", data=a_stack)
        h.create_group("QualityControl")
        h["QualityControl"].create_dataset(
            "ParabolicFitValid/value", data=valid_stack
        )
        h.create_group("Axes")
        h["Axes"].create_dataset("CardiacPhase/value", data=phase)
        h.create_group("Files")
        h["Files"].create_dataset("Names/value", data=names)

    with h5py.File(output_dir / f"{prefix}_partition_functions_pA.h5", "w") as h:
        h.create_group("PartitionFunctions")
        h["PartitionFunctions"].create_dataset("p_A/value", data=p_a_stack)
        h.create_group("QualityControl")
        h["QualityControl"].create_dataset(
            "ParabolicFitValid/value", data=valid_stack
        )
        h.create_group("Axes")
        h["Axes"].create_dataset("CardiacPhase/value", data=phase)
        h.create_group("Files")
        h["Files"].create_dataset("Names/value", data=names)


def _collect_recordings_from_dir(input_dir: Path, include_dir_prefix: bool = False):
    input_dir = Path(input_dir)
    source_paths = list(_iter_ef_files(input_dir))
    if not source_paths:
        raise FileNotFoundError(f"No .h5 files found in {input_dir}")

    recordings = []
    for path in source_paths:
        v_per_beat, phase, _ = _load_source_waveform(path)
        recordings.append((path, v_per_beat, phase))

    a_list = []
    valid_list = []
    names = []
    phases = []

    target_timepoints = max(waveform.shape[0] for _, waveform, _ in recordings)
    target_beats = max(waveform.shape[1] for _, waveform, _ in recordings)
    target_segments = max(waveform.shape[2] for _, waveform, _ in recordings)
    target_radii = max(waveform.shape[3] for _, waveform, _ in recordings)

    for path, v_per_beat, phase in recordings:
        a = _normalize_waveform_axes(
            v_per_beat,
            target_timepoints,
            target_beats,
            target_segments,
            target_radii,
        )
        phase = _normalize_phase_axis(phase, target_timepoints)

        a_list.append(a[None, ...])
        valid_list.append(np.ones((1, a.shape[1], a.shape[2], a.shape[3]), dtype=bool))
        phases.append(np.asarray(phase, dtype=float))
        name = path.stem if not include_dir_prefix else f"{input_dir.name}/{path.stem}"
        names.append(name)

    a_stack = np.concatenate(a_list, axis=0)
    p_a_stack = _make_p_a_stack(a_stack)
    valid_stack = np.concatenate(valid_list, axis=0)

    phase_values = np.asarray(phases[0], dtype=float)
    if any(
        phase.shape != phase_values.shape or not np.allclose(phase, phase_values)
        for phase in phases[1:]
    ):
        phase_values = np.arange(target_timepoints, dtype=float) / float(
            target_timepoints
        )

    return a_stack, p_a_stack, valid_stack, np.array(names, dtype="S"), phase_values


def _resolve_group_prefix(group: str) -> str:
    group_name = group.lower()
    if group_name in {"ctrl", "control", "baseline"}:
        return "ctrl"
    if group_name in {"bl1", "baseline1", "baseline_1"}:
        return "bl1"
    if group_name in {"bl2", "baseline2", "baseline_2"}:
        return "bl2"
    if group_name in {
        "pressure",
        "indentation",
        "pressure-flicker",
        "flicker-pressure",
    }:
        return "pressure"
    if group_name in {"flicker", "f"}:
        return "flicker"
    raise ValueError(f"Unknown group: {group}")


def convert_ef_directory_to_svd_inputs(
    input_dir: Path, output_dir: Path, group: str = "flicker"
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = _resolve_group_prefix(group)
    a_stack, p_a_stack, valid_stack, names, phase = _collect_recordings_from_dir(
        input_dir, include_dir_prefix=False
    )
    _write_group_h5(output_dir, prefix, a_stack, p_a_stack, valid_stack, names, phase)

    return output_dir


def _has_token(tokens: set[str], *values: str) -> bool:
    return bool(tokens.intersection(values))


def _classify_acquisition_dir(name: str) -> str | None:
    normalized = name.lower()
    tokens = {token for token in re.split(r"[^a-z0-9]+", normalized) if token}

    if normalized in {"bl1", "baseline1", "baseline_1"} or _has_token(
        tokens, "bl1", "baseline1"
    ):
        return "bl1"
    if normalized in {"bl2", "baseline2", "baseline_2"} or _has_token(
        tokens, "bl2", "baseline2"
    ):
        return "bl2"
    if normalized in {
        "pressure",
        "indentation",
        "pressure-flicker",
        "flicker-pressure",
    } or _has_token(tokens, "pressure", "indentation"):
        return "pressure"
    if normalized in {"f", "flicker"} or _has_token(tokens, "f", "flicker"):
        return "flicker"
    if normalized in {"ctrl", "control", "baseline"} or _has_token(
        tokens, "ctrl", "control", "baseline", "base"
    ):
        return "ctrl"
    return None


GroupData = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _combine_groups(group_results: list[GroupData]):
    combined_a_list = []
    combined_valid_list = []
    names = []
    phases = []

    target_timepoints = max(group[0].shape[1] for group in group_results)
    target_beats = max(group[0].shape[2] for group in group_results)
    target_segments = max(group[0].shape[3] for group in group_results)
    target_radii = max(group[0].shape[4] for group in group_results)

    for a_stack, _p_a_stack, valid_stack, group_names, phase in group_results:
        a_stack = _normalize_waveform_stack_axes(
            a_stack,
            target_timepoints,
            target_beats,
            target_segments,
            target_radii,
        )
        valid_stack = _normalize_valid_stack_axes(
            valid_stack,
            target_beats,
            target_segments,
            target_radii,
        )
        combined_a_list.append(a_stack)
        combined_valid_list.append(valid_stack)
        names.extend(
            [
                name.decode("utf-8") if isinstance(name, bytes) else str(name)
                for name in group_names
            ]
        )
        phases.append(phase)

    phase_values = phases[0]
    if any(
        phase.shape != phase_values.shape or not np.allclose(phase, phase_values)
        for phase in phases[1:]
    ):
        phase_values = np.arange(target_timepoints, dtype=float) / float(
            target_timepoints
        )

    combined_a = np.concatenate(combined_a_list, axis=0)
    return (
        combined_a,
        _make_p_a_stack(combined_a),
        np.concatenate(combined_valid_list, axis=0),
        np.array(names, dtype="S"),
        phase_values,
    )


def convert_acquisition_groups_to_svd_inputs(input_root: Path, output_dir: Path):
    input_root = Path(input_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    acquisition_groups: dict[str, list[Path]] = {}
    for child in sorted(input_root.iterdir()):
        if not child.is_dir():
            continue
        group_name = _classify_acquisition_dir(child.name)
        if group_name is not None:
            acquisition_groups.setdefault(group_name, []).append(child)

    if not acquisition_groups:
        raise FileNotFoundError(f"No acquisition folders found in {input_root}")

    grouped_results = {}
    for group_name, dirs in acquisition_groups.items():
        dir_results = [
            _collect_recordings_from_dir(acquisition_dir, include_dir_prefix=True)
            for acquisition_dir in dirs
        ]
        grouped_results[group_name] = _combine_groups(dir_results)
        _write_group_h5(output_dir, group_name, *grouped_results[group_name])

    control_sources = [
        grouped_results[group_name]
        for group_name in ("ctrl", "bl1", "bl2")
        if group_name in grouped_results
    ]
    if control_sources:
        _write_group_h5(output_dir, "ctrl", *_combine_groups(control_sources))

    # Keep the pressure datasource alias available for provocation folder F.
    if "flicker" in grouped_results and "pressure" not in grouped_results:
        _write_group_h5(output_dir, "pressure", *grouped_results["flicker"])

    return output_dir


def _collect_flicker_recordings(input_dir: Path):
    source_paths = list(_iter_ef_files(input_dir))
    if not source_paths:
        raise FileNotFoundError(f"No .h5 files found in {input_dir}")

    recordings = []
    for path in source_paths:
        try:
            waveform, phase, _ = _load_source_waveform(path)
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
            continue

        recordings.append((path, waveform, phase))

    if not recordings:
        raise FileNotFoundError(f"No usable EF waveform files found in {input_dir}")

    grouped_recordings = {"bl1": [], "bl2": [], "flicker": []}
    for path, waveform, phase in recordings:
        group_name = _infer_flicker_group_name(path)
        grouped_recordings[group_name].append((path, waveform, phase))

    control_recordings = [
        grouped_recordings[group_name]
        for group_name in ("bl1", "bl2")
        if grouped_recordings[group_name]
    ]
    control_target_timepoints = None
    control_target_beats = None
    control_target_segments = None
    if control_recordings:
        control_target_timepoints = max(
            waveform.shape[0]
            for group in control_recordings
            for _, waveform, _ in group
        )
        control_target_beats = max(
            waveform.shape[1]
            for group in control_recordings
            for _, waveform, _ in group
        )
        control_target_segments = max(
            waveform.shape[2]
            for group in control_recordings
            for _, waveform, _ in group
        )

    results = {}
    for group_name, group_recordings in grouped_recordings.items():
        if not group_recordings:
            continue
        if group_name in {"bl1", "bl2"} and control_target_timepoints is not None:
            target_timepoints = control_target_timepoints
            target_beats = control_target_beats
            target_segments = control_target_segments
        else:
            target_timepoints = max(
                waveform.shape[0] for _, waveform, _ in group_recordings
            )
            target_beats = max(waveform.shape[1] for _, waveform, _ in group_recordings)
            target_segments = max(
                waveform.shape[2] for _, waveform, _ in group_recordings
            )

        a_list = []
        p_a_list = []
        valid_list = []
        phases = []
        names = []

        for path, waveform, phase in group_recordings:
            waveform = _normalize_time_axis(waveform, target_timepoints)
            waveform = _normalize_beats(waveform, target_beats)
            waveform = _normalize_segment_axis(waveform, target_segments)
            phase = _normalize_phase_axis(phase, target_timepoints)
            a, p_a = _make_svd_like_arrays(waveform)

            a_list.append(a[None, ...])
            p_a_list.append(p_a[None, ...])
            valid_list.append(
                np.ones((1, a.shape[1], a.shape[2], a.shape[3]), dtype=bool)
            )
            phases.append(np.asarray(phase, dtype=float))
            names.append(path.stem)

        phase_values = np.asarray(phases[0], dtype=float)
        if any(
            phase.shape != phase_values.shape or not np.allclose(phase, phase_values)
            for phase in phases[1:]
        ):
            raise ValueError(
                f"All recordings in {group_name} must share the same "
                "cardiac-phase vector"
            )

        results[group_name] = (
            np.concatenate(a_list, axis=0),
            np.concatenate(p_a_list, axis=0),
            np.concatenate(valid_list, axis=0),
            np.array(names, dtype="S"),
            phase_values,
        )

    return results


def convert_flicker_directory_to_svd_inputs(input_dir: Path, output_dir: Path):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped_results = _collect_flicker_recordings(input_dir)

    for group_name in ("bl1", "bl2", "flicker"):
        if group_name in grouped_results:
            _write_group_h5(output_dir, group_name, *grouped_results[group_name])

    if "flicker" in grouped_results:
        _write_group_h5(output_dir, "pressure", *grouped_results["flicker"])

    control_sources = [
        grouped_results[group_name]
        for group_name in ("bl1", "bl2")
        if group_name in grouped_results
    ]
    if control_sources:
        _write_group_h5(output_dir, "ctrl", *_combine_groups(control_sources))

    return output_dir


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert EF HDF5 files into the SVD-compatible HDF5 layout"
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--group", default=None)
    args = parser.parse_args()

    if args.group is not None:
        convert_ef_directory_to_svd_inputs(
            args.input_dir, args.output_dir, group=args.group
        )
    else:
        convert_acquisition_groups_to_svd_inputs(args.input_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
