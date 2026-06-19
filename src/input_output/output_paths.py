from pathlib import Path

H5_OUTPUT_DIRNAME = "h5"
PNG_OUTPUT_DIRNAME = "png"


def h5_output_dir(output_root: str | Path) -> Path:
    """Return the standard directory for generated HDF5 outputs."""
    return Path(output_root) / H5_OUTPUT_DIRNAME


def h5_output_parent(
    output_root: str | Path,
    relative_parent: str | Path = Path("."),
) -> Path:
    """Return the standard parent directory for one generated HDF5 output."""
    return h5_output_dir(output_root) / Path(relative_parent)


def png_output_dir(output_root: str | Path) -> Path:
    """Return the standard directory for generated PNG companion outputs."""
    return Path(output_root) / PNG_OUTPUT_DIRNAME
