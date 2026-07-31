"""EyeFlow HDF5 schema paths and backward-compatible dataset lookup.

EyeFlow 0.11.4 introduced the ``eyeflow_v2`` layout.  AngioEye treats those
paths as canonical while continuing to accept the immediately preceding
layout, which is useful for mixed cohorts and existing tests.
"""

from __future__ import annotations

from collections.abc import Iterable

import h5py

BEAT_PERIOD = "/Processing/VelocityPerBeat/BeatPeriodSeconds/value"

VELOCITY_PER_BEAT = {
    ("artery", "raw"): "/Processing/VelocityPerBeat/Artery/Raw/value",
    ("artery", "bandlimited"): (
        "/Processing/VelocityPerBeat/Artery/BandLimited/value"
    ),
    ("vein", "raw"): "/Processing/VelocityPerBeat/Vein/Raw/value",
    ("vein", "bandlimited"): "/Processing/VelocityPerBeat/Vein/BandLimited/value",
}

SEGMENT_VELOCITY_PER_BEAT = {
    ("artery", "raw"): (
        "/Processing/VelocityPerBeat/Artery/Segments/Raw/value"
    ),
    ("artery", "bandlimited"): (
        "/Processing/VelocityPerBeat/Artery/Segments/BandLimited/value"
    ),
    ("vein", "raw"): "/Processing/VelocityPerBeat/Vein/Segments/Raw/value",
    ("vein", "bandlimited"): (
        "/Processing/VelocityPerBeat/Vein/Segments/BandLimited/value"
    ),
}

_LEGACY_PATHS = {
    BEAT_PERIOD: ("/Artery/VelocityPerBeat/beatPeriodSeconds/value",),
    VELOCITY_PER_BEAT[("artery", "raw")]: (
        "/Artery/VelocityPerBeat/VelocitySignalPerBeat/value",
    ),
    VELOCITY_PER_BEAT[("artery", "bandlimited")]: (
        "/Artery/VelocityPerBeat/VelocitySignalPerBeatBandLimited/value",
    ),
    VELOCITY_PER_BEAT[("vein", "raw")]: (
        "/Vein/VelocityPerBeat/VelocitySignalPerBeat/value",
    ),
    VELOCITY_PER_BEAT[("vein", "bandlimited")]: (
        "/Vein/VelocityPerBeat/VelocitySignalPerBeatBandLimited/value",
    ),
    SEGMENT_VELOCITY_PER_BEAT[("artery", "raw")]: (
        "/Artery/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegment/value",
    ),
    SEGMENT_VELOCITY_PER_BEAT[("artery", "bandlimited")]: (
        "/Artery/VelocityPerBeat/Segments/"
        "VelocitySignalPerBeatPerSegmentBandLimited/value",
    ),
    SEGMENT_VELOCITY_PER_BEAT[("vein", "raw")]: (
        "/Vein/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegment/value",
    ),
    SEGMENT_VELOCITY_PER_BEAT[("vein", "bandlimited")]: (
        "/Vein/VelocityPerBeat/Segments/"
        "VelocitySignalPerBeatPerSegmentBandLimited/value",
    ),
    "/Segmentation/OpticDisc/CenterXY/value": (
        "/Topology/OpticDisc/CenterXY/value",
    ),
    "/Segmentation/OpticDisc/Mask/value": ("/Topology/OpticDisc/Mask/value",),
    "/Segmentation/Artery/BranchIds/value": (
        "/Topology/Artery/BranchIds/value",
    ),
    "/Segmentation/Artery/BranchLabelMap/value": (
        "/Topology/Artery/BranchLabelMap/value",
    ),
    "/Segmentation/Artery/SegmentCenterXY/value": (
        "/Topology/Artery/SegmentCenterXY/value",
    ),
    "/Segmentation/Vein/BranchIds/value": ("/Topology/Vein/BranchIds/value",),
    "/Segmentation/Vein/BranchLabelMap/value": (
        "/Topology/Vein/BranchLabelMap/value",
    ),
    "/Segmentation/Vein/SegmentCenterXY/value": (
        "/Topology/Vein/SegmentCenterXY/value",
    ),
    "/Segmentation/Artery/Mask/value": ("/Artery/Segmentation/Mask/value",),
    "/Segmentation/Vein/Mask/value": ("/Vein/Segmentation/Mask/value",),
    "/Processing/Velocity/Artery/Raw/value": (
        "/Artery/Velocity/VelocitySignal/value",
    ),
    "/Processing/Velocity/Vein/Raw/value": (
        "/Vein/Velocity/VelocitySignal/value",
    ),
    "/Processing/FrequencyMaps/fRMS_avg/value": ("/Maps/f_AVG_mean/value",),
    "/Processing/CrossSections/Artery/VelocityProfile/value": (
        "/Artery/CrossSections/VelocityProfilesSegInterpOneBeat/value",
    ),
}


def candidate_paths(canonical_path: str) -> tuple[str, ...]:
    """Return the canonical EyeFlow path followed by supported legacy aliases."""
    if canonical_path.startswith("/Processing/Metrics/waveform_shape_metrics/"):
        suffix = canonical_path.removeprefix(
            "/Processing/Metrics/waveform_shape_metrics/"
        )
        return (
            canonical_path,
            f"/Metrics/waveform_shape_metrics/{suffix}",
        )
    return (canonical_path, *_LEGACY_PATHS.get(canonical_path, ()))


def resolve_path(
    h5file: h5py.File | h5py.Group,
    canonical_path: str,
) -> str | None:
    """Resolve a canonical path in either the v2 or supported legacy layout."""
    return next(
        (path for path in candidate_paths(canonical_path) if path in h5file),
        None,
    )


def has_path(h5file: h5py.File | h5py.Group, canonical_path: str) -> bool:
    return resolve_path(h5file, canonical_path) is not None


def get_object(
    h5file: h5py.File | h5py.Group,
    canonical_path: str,
) -> h5py.Group | h5py.Dataset | None:
    path = resolve_path(h5file, canonical_path)
    return h5file.get(path) if path is not None else None


def require_dataset(
    h5file: h5py.File | h5py.Group,
    canonical_path: str,
) -> h5py.Dataset:
    obj = get_object(h5file, canonical_path)
    if not isinstance(obj, h5py.Dataset):
        candidates = ", ".join(candidate_paths(canonical_path))
        raise KeyError(f"Missing EyeFlow dataset (tried: {candidates})")
    return obj


def missing_paths(
    h5file: h5py.File | h5py.Group,
    canonical_paths: Iterable[str],
) -> tuple[str, ...]:
    return tuple(path for path in canonical_paths if not has_path(h5file, path))
