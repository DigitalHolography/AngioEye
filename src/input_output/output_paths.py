from pathlib import Path

H5_OUTPUT_DIRNAME = "h5"
PNG_OUTPUT_DIRNAME = "png"
HTML_OUTPUT_DIRNAME = "html"
APP_SUFFIXES = ("HD", "DV", "EF", "AE")


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
