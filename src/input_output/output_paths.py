from collections.abc import Sequence
import re
from pathlib import Path

from .hdf5_schema import is_hdf5_path

H5_OUTPUT_DIRNAME = "h5"
PNG_OUTPUT_DIRNAME = "png"
PNGS_OUTPUT_DIRNAME = "pngs"
EPS_OUTPUT_DIRNAME = "eps"
HTML_OUTPUT_DIRNAME = "html"
COHORT_RESULTS_DIRNAME = "cohort-results"
# Folder-name mirrors used when deriving an EPS companion path from a PNG path.
# ``pngs`` (epoch layout) still maps to sibling ``eps``, not ``epss``.
_PNG_FOLDER_TO_EPS = {
    PNG_OUTPUT_DIRNAME: EPS_OUTPUT_DIRNAME,
    PNGS_OUTPUT_DIRNAME: EPS_OUTPUT_DIRNAME,
    "export_png": "export_eps",
    "export_png_html": "export_eps_html",
}
APP_SUFFIXES = ("HD", "DV", "EF", "AE")
_APP_STEM_SUFFIXES = tuple(f"_{suffix}" for suffix in APP_SUFFIXES)
# Epoch folders are ``1_label``, ``2_label``, … Patient-id ZIP wraps such as
# ``260803_Flicker_EF`` also match ``number_label`` but are not epochs.
_EPOCH_FOLDER_RE = re.compile(
    r"^(?P<index>\d+)_(?P<label>[A-Za-z0-9]+(?:[_-][A-Za-z0-9]+)*)$"
)
_MAX_EPOCH_INDEX = 99


def default_h5_output_dir(path: str | Path) -> Path:
    """Return the default output root for a selected HDF5 input."""
    try:
        return app_output_dir(path, "AE")
    except ValueError:
        return Path(path).expanduser().parent


def default_h5_output_filename(path: str | Path) -> str:
    """Return the standard AE artifact name for a single HDF5 input."""
    try:
        stem = dataset_stem_from_path(path)
    except ValueError:
        stem = Path(path).expanduser().stem
        if stem.casefold().endswith("_ef"):
            stem = stem[:-3]
    return f"{stem or 'output'}_AE.h5"


def default_output_filename_for_run(
    data_path: str | Path,
    inputs: Sequence[Path],
) -> str | None:
    """Return the single-file AE name, leaving batch naming to the engine."""
    data_path_obj = Path(data_path).expanduser()
    if (
        len(inputs) == 1
        and data_path_obj.is_file()
        and is_hdf5_path(data_path_obj)
    ):
        return default_h5_output_filename(inputs[0])
    return None


def h5_output_dir(output_root: str | Path) -> Path:
    """Return the standard directory for generated HDF5 outputs."""
    return Path(output_root) / H5_OUTPUT_DIRNAME


def h5_output_parent(
    output_root: str | Path,
    relative_parent: str | Path = Path("."),
) -> Path:
    """Return the standard parent directory for one generated HDF5 output."""
    return h5_output_dir(output_root) / Path(relative_parent)


def is_epoch_folder(name: str) -> bool:
    """True for numbered cohort epoch folders such as ``1_BL1`` / ``2_Flicker``."""
    match = _EPOCH_FOLDER_RE.fullmatch(str(name).strip())
    if match is None:
        return False
    return 1 <= int(match.group("index")) <= _MAX_EPOCH_INDEX


def strip_epoch_wrap(relative_parent: str | Path) -> Path:
    """Drop a ZIP wrap folder so epoch folders sit at the output root.

    ``260803_Flicker_EF/1_BL1`` → ``1_BL1``. ``1_BL1`` is unchanged.
    """
    relative = Path(relative_parent)
    parts = relative.parts
    if not parts:
        return Path(".")
    for index, part in enumerate(parts):
        if is_epoch_folder(part):
            return Path(*parts[index:])
    return relative


def ae_result_filename(source_h5: str | Path) -> str:
    """``260803_GOA_1_EF.h5`` → ``260803_GOA_1_AE.h5``."""
    stem = Path(source_h5).stem
    if stem.endswith("_pipelines_result"):
        stem = stem[: -len("_pipelines_result")]
    for suffix in _APP_STEM_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)] or stem
            break
    return f"{stem}_AE.h5"


def pipeline_result_parent(
    output_root: str | Path,
    relative_parent: str | Path = Path("."),
) -> Path:
    """Parent directory for one pipeline result H5.

    Epoch folders (``1_BL1``, …) keep H5s beside ``pngs/``. A flat relative
    parent still uses the canonical ``h5/`` product folder (``.holo`` runs).
    """
    relative = strip_epoch_wrap(relative_parent)
    if relative == Path("."):
        return h5_output_parent(output_root, relative)
    return Path(output_root) / relative


def cohort_results_dir(output_root: str | Path) -> Path:
    """Return the shared root for cohort-only ``h5`` and ``png`` products."""
    return Path(output_root) / COHORT_RESULTS_DIRNAME


def png_output_dir(output_root: str | Path) -> Path:
    """Return the standard directory for generated PNG companion outputs."""
    return Path(output_root) / PNG_OUTPUT_DIRNAME


def eps_output_dir(output_root: str | Path) -> Path:
    """Return the standard directory for generated EPS companion outputs."""
    return Path(output_root) / EPS_OUTPUT_DIRNAME


def eps_path_for_png(png_path: str | Path) -> Path:
    """Mirror a PNG path into the sibling EPS tree.

    Replaces the PNG product folder in the path (``png``, ``pngs``,
    ``export_png``, ``export_png_html``, or a ``*_png`` directory) with the
    matching EPS folder and swaps the suffix to ``.eps``. When no such folder
    is present, the EPS file is written beside the PNG (same parent).
    """
    png_path = Path(png_path)
    parts = list(png_path.parts)
    for index, part in enumerate(parts[:-1]):
        key = part.lower()
        replacement = _PNG_FOLDER_TO_EPS.get(key)
        if replacement is None and key.endswith("_png"):
            replacement = f"{part[:-4]}_eps"
        if replacement is None:
            continue
        mirrored_parts = list(parts)
        mirrored_parts[index] = replacement
        mirrored = Path(mirrored_parts[0])
        for piece in mirrored_parts[1:]:
            mirrored /= piece
        return mirrored.with_suffix(".eps")
    return png_path.with_suffix(".eps")


def html_output_dir(output_root: str | Path) -> Path:
    """Return the standard directory for generated HTML companion outputs."""
    return Path(output_root) / HTML_OUTPUT_DIRNAME


def normalize_app_suffix(app_suffix: str) -> str:
    suffix = app_suffix.strip().upper().removeprefix("_")
    if suffix not in APP_SUFFIXES:
        raise ValueError(
            f"Unknown app suffix: {app_suffix!r}. "
            f"Expected one of {', '.join(APP_SUFFIXES)}."
        )
    return suffix


def dataset_root_from_path(path: str | Path) -> Path:
    """Return the <stem> dataset folder for a HOLO/app companion path."""
    path_obj = Path(path).expanduser()
    if path_obj.suffix.lower() == ".holo":
        return path_obj.parent / path_obj.stem

    candidates = (path_obj, *path_obj.parents)
    for candidate in candidates:
        folder_name = candidate.name
        for suffix in APP_SUFFIXES:
            marker = f"_{suffix}"
            if folder_name.endswith(marker) and len(folder_name) > len(marker):
                return candidate.parent

    raise ValueError(f"Could not resolve dataset root from path: {path_obj}")


def dataset_stem_from_path(path: str | Path) -> str:
    """Return the dataset stem for a HOLO/app companion path."""
    return dataset_root_from_path(path).name


def app_output_dir(path: str | Path, app_suffix: str) -> Path:
    """Return <stem>/<stem>_<APP_SUFFIX> for any path in the dataset tree."""
    suffix = normalize_app_suffix(app_suffix)
    dataset_root = dataset_root_from_path(path)
    return dataset_root / f"{dataset_root.name}_{suffix}"


def companion_output_dir(
    path: str | Path,
    *,
    app_suffix: str,
    query_type: str,
) -> Path:
    """Return an app companion folder such as <stem>_AE/html or <stem>_DV/png."""
    app_dir = app_output_dir(path, app_suffix)
    query = query_type.strip().lower()
    if query == H5_OUTPUT_DIRNAME:
        return h5_output_dir(app_dir)
    if query == PNG_OUTPUT_DIRNAME:
        return png_output_dir(app_dir)
    if query == EPS_OUTPUT_DIRNAME:
        return eps_output_dir(app_dir)
    if query == HTML_OUTPUT_DIRNAME:
        return html_output_dir(app_dir)
    return app_dir / query


def companion_file_path(
    path: str | Path,
    *,
    app_suffix: str,
    query_type: str,
    filename: str | Path,
) -> Path:
    """Return an exact modern app companion file path."""
    return companion_output_dir(
        path,
        app_suffix=app_suffix,
        query_type=query_type,
    ) / Path(filename)


def legacy_companion_dirs(path: str | Path, *, query_type: str) -> tuple[Path, ...]:
    """Return legacy companion folders near an H5 path, ordered by preference."""
    path_obj = Path(path).expanduser()
    query = query_type.strip().lower()
    candidates: list[Path] = []

    if path_obj.parent.name.lower() == H5_OUTPUT_DIRNAME:
        candidates.append(path_obj.parent.parent / query)
    candidates.append(path_obj.parent / query)
    if path_obj.parent.parent != path_obj.parent:
        candidates.append(path_obj.parent.parent / query)

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return tuple(unique)


def find_companion_file(
    path: str | Path,
    *,
    app_suffix: str,
    query_type: str,
    filename: str | Path,
    legacy_filenames: tuple[str | Path, ...] = (),
) -> Path | None:
    """Find an exact companion file, trying modern app paths before legacy folders."""
    filenames = (filename, *legacy_filenames)
    try:
        modern_path = companion_file_path(
            path,
            app_suffix=app_suffix,
            query_type=query_type,
            filename=filename,
        )
    except ValueError:
        modern_path = None
    if modern_path is not None and modern_path.is_file():
        return modern_path

    for legacy_dir in legacy_companion_dirs(path, query_type=query_type):
        for legacy_filename in filenames:
            legacy_path = legacy_dir / Path(legacy_filename)
            if legacy_path.is_file():
                return legacy_path
    return None
