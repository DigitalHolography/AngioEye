"""Build independent, compact analysis files from EyeFlow outputs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import h5py

from app_settings import app_version

PROCESSING_BRANCHES = ("Processing", "ProcessingAlt")
_SIGNAL_PATHS = (
    "VelocityPerBeat/Artery/Raw/value",
    "VelocityPerBeat/Artery/BandLimited/value",
    "VelocityPerBeat/Vein/Raw/value",
    "VelocityPerBeat/Vein/BandLimited/value",
)
_SEGMENTATION_METADATA_PATHS = (
    "PixelPitch_m",
    "OpticDisc/Center",
    "OpticDisc/Width",
    "OpticDisc/Height",
    "Artery/DeltaRadius",
    "Artery/LumenDiameter",
    "Artery/SegmentMaskArea",
    "Vein/DeltaRadius",
    "Vein/LumenDiameter",
    "Vein/SegmentMaskArea",
)


def _copy_object(source: h5py.Group, destination: h5py.Group, path: str) -> None:
    obj = source.get(path)
    if not isinstance(obj, (h5py.Group, h5py.Dataset)):
        return
    parts = path.split("/")
    parent = destination
    for index, part in enumerate(parts[:-1], start=1):
        parent = parent.require_group(part)
        source_parent = source.get("/".join(parts[:index]))
        if isinstance(source_parent, h5py.Group):
            for key, value in source_parent.attrs.items():
                parent.attrs[key] = value
    name = parts[-1]
    source.copy(obj, parent, name=name, expand_external=True, expand_soft=True)


def stage_analysis_file(
    source: Path,
    destination: Path,
    *,
    processing_branch: str = "Processing",
) -> Path:
    """Copy one EyeFlow metric branch into an independent analysis HDF5.

    Metrics retain their dataset attributes and axes. The selected branch is
    always stored under ``/Processing`` so statistical readers use one schema.
    Only global beat signals, cardiac timing and small geometry metadata are
    retained alongside the metrics; images and segment waveform stacks stay
    in the original EyeFlow file.
    """
    if processing_branch not in PROCESSING_BRANCHES:
        raise ValueError(
            f"Unknown EyeFlow processing branch {processing_branch!r}; "
            f"expected one of {', '.join(PROCESSING_BRANCHES)}."
        )
    source = Path(source).expanduser()
    destination = Path(destination).expanduser()
    if source.resolve() == destination.resolve() or (
        destination.exists() and source.samefile(destination)
    ):
        raise ValueError("The analysis output must differ from the EyeFlow source.")

    with h5py.File(source, "r") as original:
        branch = original.get(processing_branch)
        if not isinstance(branch, h5py.Group) or not isinstance(
            branch.get("Metrics"), h5py.Group
        ):
            raise ValueError(
                f"EyeFlow input {source} has no /{processing_branch}/Metrics group."
            )

        versions = {}
        if isinstance(original.get("app_versions"), h5py.Dataset):
            versions = json.loads(original["app_versions"][()])
            if not isinstance(versions, dict):
                raise ValueError(
                    f"EyeFlow app_versions must be a JSON object: {source}"
                )
        versions["AE_version"] = app_version()

        destination.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp:
            temporary_path = Path(temp.name)
        try:
            with h5py.File(temporary_path, "w") as staged:
                for key, value in original.attrs.items():
                    staged.attrs[key] = value
                staged.attrs["angioeye_input_file"] = str(source.resolve())
                staged.attrs["angioeye_processing_branch"] = processing_branch
                processing = staged.create_group("Processing")
                for key, value in branch.attrs.items():
                    processing.attrs[key] = value
                _copy_object(branch, processing, "Metrics")
                _copy_object(branch, processing, "CardiacCycle")
                for path in _SIGNAL_PATHS:
                    _copy_object(branch, processing, path)
                _copy_object(original, staged, "Meta")
                segmentation = original.get("Segmentation")
                if isinstance(segmentation, h5py.Group):
                    target = staged.create_group("Segmentation")
                    for key, value in segmentation.attrs.items():
                        target.attrs[key] = value
                    for path in _SEGMENTATION_METADATA_PATHS:
                        _copy_object(segmentation, target, path)
                staged.create_dataset(
                    "app_versions",
                    data=json.dumps(versions),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )
            os.replace(temporary_path, destination)
        finally:
            temporary_path.unlink(missing_ok=True)
    return destination
