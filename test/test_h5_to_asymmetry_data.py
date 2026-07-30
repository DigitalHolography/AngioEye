import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from postprocess.utils.h5_to_asymmetry_data import (
    convert_acquisition_groups_to_svd_inputs,
    convert_ef_directory_to_svd_inputs,
    convert_flicker_directory_to_svd_inputs,
)
from postprocess.utils.indentation_flicker_svd import (
    load_recording_group,
    main as run_indentation_flicker_svd,
)


def _write_source_h5(path: Path, signal: np.ndarray, include_meta_time: bool = True):
    with h5py.File(path, "w") as h:
        h.create_group("/Artery")
        h.create_group("/Artery/VelocityPerBeat")
        h.create_group("/Artery/VelocityPerBeat/Segments")
        h.create_dataset(
            "/Artery/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegment/value",
            data=signal,
        )
        h.create_dataset(
            "/Artery/VelocityPerBeat/beatPeriodSeconds/value",
            data=np.array([np.linspace(0.5, 0.7, signal.shape[1])], dtype=float),
        )
        if include_meta_time:
            h.create_dataset(
                "/Meta/Time/value",
                data=np.array([[0.0], [0.5], [1.0]], dtype=float),
            )


def test_convert_ef_directory_to_svd_inputs(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    signal = np.arange(24, dtype=float).reshape(2, 3, 2, 2)
    source_path = source_dir / "recording_a.h5"
    _write_source_h5(source_path, signal)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    convert_ef_directory_to_svd_inputs(source_dir, out_dir, group="flicker")

    asymmetry_path = out_dir / "flicker_asymmetry_A.h5"
    partition_path = out_dir / "flicker_partition_functions_pA.h5"
    assert asymmetry_path.exists()
    assert partition_path.exists()

    with h5py.File(asymmetry_path, "r") as h:
        a = np.asarray(h["Asymmetry/A/value"], dtype=float)
        valid = np.asarray(h["QualityControl/ParabolicFitValid/value"], dtype=bool)
        phase = np.asarray(h["Axes/CardiacPhase/value"], dtype=float)
        names = [
            name.decode("utf-8") if isinstance(name, bytes) else str(name)
            for name in h["Files/Names/value"][:]
        ]

    with h5py.File(partition_path, "r") as h:
        p_a = np.asarray(h["PartitionFunctions/p_A/value"], dtype=float)

    assert a.shape == (1, 2, 3, 2, 2)
    assert p_a.shape == a.shape
    assert valid.shape == (1, 3, 2, 2)
    assert phase.shape == (2,)
    assert names == ["recording_a"]

    expected_p_a = (
        signal[None, ...] - signal.mean(axis=0, keepdims=True)[None, ...]
    ) ** 2
    assert np.allclose(p_a[0], expected_p_a)


def test_convert_acquisition_groups_to_svd_inputs(tmp_path):
    root = tmp_path / "acquisitions"
    root.mkdir()

    bl1_dir = root / "BL1"
    bl2_dir = root / "BL2"
    pressure_dir = root / "F"
    bl1_dir.mkdir()
    bl2_dir.mkdir()
    pressure_dir.mkdir()

    signal = np.arange(24, dtype=float).reshape(2, 3, 2, 2)

    for acquisition_dir in [bl1_dir, bl2_dir, pressure_dir]:
        _write_source_h5(acquisition_dir / "recording_a.h5", signal)

    out_dir = tmp_path / "out"
    convert_acquisition_groups_to_svd_inputs(root, out_dir)

    bl1_asymmetry = out_dir / "bl1_asymmetry_A.h5"
    bl1_partition = out_dir / "bl1_partition_functions_pA.h5"
    bl2_asymmetry = out_dir / "bl2_asymmetry_A.h5"
    bl2_partition = out_dir / "bl2_partition_functions_pA.h5"
    flicker_asymmetry = out_dir / "flicker_asymmetry_A.h5"
    flicker_partition = out_dir / "flicker_partition_functions_pA.h5"
    ctrl_asymmetry = out_dir / "ctrl_asymmetry_A.h5"
    ctrl_partition = out_dir / "ctrl_partition_functions_pA.h5"
    pressure_asymmetry = out_dir / "pressure_asymmetry_A.h5"
    pressure_partition = out_dir / "pressure_partition_functions_pA.h5"

    for path in [
        bl1_asymmetry,
        bl1_partition,
        bl2_asymmetry,
        bl2_partition,
        flicker_asymmetry,
        flicker_partition,
        ctrl_asymmetry,
        ctrl_partition,
        pressure_asymmetry,
        pressure_partition,
    ]:
        assert path.exists()

    bl1_a, _, _, bl1_names = load_recording_group(out_dir, "bl1", "A")
    bl2_a, _, _, bl2_names = load_recording_group(out_dir, "bl2", "A")
    ctrl_a, _, _, ctrl_names = load_recording_group(out_dir, "ctrl", "A")
    flicker_a, _, _, flicker_names = load_recording_group(out_dir, "f", "A")
    pressure_a, _, _, pressure_names = load_recording_group(out_dir, "pressure", "A")

    assert bl1_a.shape == (1, 2, 3, 2, 2)
    assert bl2_a.shape == (1, 2, 3, 2, 2)
    assert ctrl_a.shape == (2, 2, 3, 2, 2)
    assert flicker_a.shape == (1, 2, 3, 2, 2)
    assert pressure_a.shape == flicker_a.shape
    assert bl1_names == ["BL1/recording_a"]
    assert bl2_names == ["BL2/recording_a"]
    assert ctrl_names == ["BL1/recording_a", "BL2/recording_a"]
    assert flicker_names == ["F/recording_a"]
    assert pressure_names == flicker_names


def test_convert_acquisition_groups_uses_synthetic_phase_when_meta_time_missing(
    tmp_path,
):
    root = tmp_path / "acquisitions"
    source_dir = root / "F"
    source_dir.mkdir(parents=True)

    signal = np.arange(24, dtype=float).reshape(2, 3, 2, 2)
    _write_source_h5(
        source_dir / "recording_a.h5",
        signal,
        include_meta_time=False,
    )

    out_dir = tmp_path / "out"
    convert_acquisition_groups_to_svd_inputs(root, out_dir)

    _, phase, _, _ = load_recording_group(out_dir, "flicker", "A")

    assert np.allclose(phase, np.array([0.0, 0.5]))


def test_convert_flicker_directory_to_svd_inputs(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    signal = np.arange(24, dtype=float).reshape(2, 3, 2, 2)
    source_path = source_dir / "recording_a.h5"
    _write_source_h5(source_path, signal)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    convert_flicker_directory_to_svd_inputs(source_dir, out_dir)

    asymmetry_path = out_dir / "flicker_asymmetry_A.h5"
    partition_path = out_dir / "flicker_partition_functions_pA.h5"
    assert asymmetry_path.exists()
    assert partition_path.exists()

    with h5py.File(asymmetry_path, "r") as h:
        a = np.asarray(h["Asymmetry/A/value"], dtype=float)
        valid = np.asarray(h["QualityControl/ParabolicFitValid/value"], dtype=bool)
        phase = np.asarray(h["Axes/CardiacPhase/value"], dtype=float)
        names = [
            name.decode("utf-8") if isinstance(name, bytes) else str(name)
            for name in h["Files/Names/value"][:]
        ]

    with h5py.File(partition_path, "r") as h:
        p_a = np.asarray(h["PartitionFunctions/p_A/value"], dtype=float)

    assert a.shape == (1, 2, 3, 2, 2)
    assert p_a.shape == a.shape
    assert valid.shape == (1, 3, 2, 2)
    assert phase.shape == (2,)
    assert names == ["recording_a"]

    expected_p_a = (
        signal[None, ...] - signal.mean(axis=0, keepdims=True)[None, ...]
    ) ** 2
    assert np.allclose(p_a[0], expected_p_a)


def test_convert_flicker_directory_to_svd_inputs_resamples_segment_axis(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    signal_a = np.arange(24, dtype=float).reshape(2, 3, 2, 2)
    signal_b = np.arange(36, dtype=float).reshape(2, 3, 3, 2)

    for name, signal in [("recording_a", signal_a), ("recording_b", signal_b)]:
        _write_source_h5(source_dir / f"{name}.h5", signal)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    convert_flicker_directory_to_svd_inputs(source_dir, out_dir)

    with h5py.File(out_dir / "flicker_asymmetry_A.h5", "r") as h:
        a = np.asarray(h["Asymmetry/A/value"], dtype=float)

    assert a.shape == (2, 2, 3, 3, 2)


def test_convert_flicker_directory_to_svd_inputs_resamples_beat_axis(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    signal_a = np.arange(24, dtype=float).reshape(2, 3, 2, 2)
    signal_b = np.arange(40, dtype=float).reshape(2, 5, 2, 2)

    for name, signal in [("recording_a", signal_a), ("recording_b", signal_b)]:
        _write_source_h5(source_dir / f"{name}.h5", signal)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    convert_flicker_directory_to_svd_inputs(source_dir, out_dir)

    with h5py.File(out_dir / "flicker_asymmetry_A.h5", "r") as h:
        a = np.asarray(h["Asymmetry/A/value"], dtype=float)

    assert a.shape == (2, 2, 5, 2, 2)


def test_convert_flicker_directory_to_svd_inputs_splits_bl1_bl2_and_flicker(
    tmp_path,
):
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    for index in [0, 7, 8, 19, 20, 27]:
        signal = np.arange(24, dtype=float).reshape(2, 3, 2, 2)
        _write_source_h5(source_dir / f"recording_{index}_EF.h5", signal)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    convert_flicker_directory_to_svd_inputs(source_dir, out_dir)

    for prefix in ["bl1", "flicker", "pressure", "bl2", "ctrl"]:
        assert (out_dir / f"{prefix}_asymmetry_A.h5").exists()
        assert (out_dir / f"{prefix}_partition_functions_pA.h5").exists()

    flicker_a, _, _, flicker_names = load_recording_group(out_dir, "f", "A")
    pressure_a, _, _, pressure_names = load_recording_group(out_dir, "pressure", "A")

    assert flicker_a.shape == pressure_a.shape
    assert flicker_names == pressure_names


def test_indentation_flicker_svd_runs_flicker_analysis_for_both_eyes(
    tmp_path,
    capsys,
):
    raw_root = tmp_path / "raw"
    export_root = tmp_path / "exported"
    signal = np.arange(24, dtype=float).reshape(2, 3, 2, 2)

    for eye in ("L", "R"):
        eye_raw = raw_root / eye
        for acquisition in ("BL1", "BL2", "F"):
            acquisition_dir = eye_raw / acquisition
            acquisition_dir.mkdir(parents=True)
            _write_source_h5(acquisition_dir / "recording_a.h5", signal)

        convert_acquisition_groups_to_svd_inputs(eye_raw, export_root / eye)

    exit_code = run_indentation_flicker_svd([str(export_root), "--mode", "flicker"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "=== left eye:" in output
    assert "=== right eye:" in output
    assert "ctrl vs flicker" in output
