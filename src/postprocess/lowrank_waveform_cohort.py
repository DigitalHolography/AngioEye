"""Cohort low-rank Figs 4--8 and ``lowrank_cohort.h5`` from EyeFlow H5s.

Joint-SVD Figs 4--8 keep the article acquisition scalars; a parallel
``*_pb.png`` set uses ``median_b`` of packed per-beat SVD endpoints (ρ is
``median_b(R)/median_b(R0)``). ``T``, ``μ``, R0, and MPR stay joint.
Fig. 4 joint and Fig. 5 ratio plots use packed joint singular values. Stats /
confounds live in the same ``lowrank_cohort.h5``.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from scipy.stats import kruskal, linregress, mannwhitneyu, spearmanr

from input_output import (
    COHORT_RESULTS_DIRNAME,
    H5_OUTPUT_DIRNAME,
    cohort_results_dir,
    find_hdf5_inputs,
    h5_output_dir,
    png_output_dir,
)
from input_output.archive_io import extracted_zip_tree
from input_output.hdf5_io import (
    UTF8_STRING_DTYPE,
    open_h5,
    set_attr_safe,
    write_value_dataset,
)
from input_output.hdf5_schema import ANGIOEYE_POSTPROCESS_ROOT
from pipelines.lowrank_waveform_decomposition import (
    SPECTRUM_N_MODES,
    SVD_METHODS,
    aggregate_beatwise,
    aggregate_rho,
    coerce_beat_spectra,
    enabled_vessels,
    find_eyeflow_lowrank_group,
    finite_std,
    is_usable_beat_spectra,
    load_acquisition_from_result_h5,
    mean_pm_std,
)

from .core.base import (
    BatchPostprocess,
    PostprocessContext,
    PostprocessResult,
    registerPostprocess,
)

# =====================================================================
# Group / epoch identity
# =====================================================================

EPOCH_ORDER = ("baseline1", "flicker", "baseline2")
EPOCH_SHORT = {
    "baseline1": "B1",
    "flicker": "Flicker",
    "baseline2": "B2",
}
EPOCH_SHORT_TO_KEY = {short: key for key, short in EPOCH_SHORT.items()}
EPOCH_ALIASES = {
    "bl1": "baseline1",
    "b1": "baseline1",
    "baseline1": "baseline1",
    "f": "flicker",
    "flicker": "flicker",
    "bl2": "baseline2",
    "b2": "baseline2",
    "baseline2": "baseline2",
}
_LEADING_INDEX_RE = re.compile(r"^\d+[\s_\-]*")
_NUMBERED_GROUP_RE = re.compile(
    r"^(?P<index>\d+)_(?P<label>[A-Za-z0-9]+(?:[_-][A-Za-z0-9]+)*)$"
)
_PATIENT_ID_RE = re.compile(r"^(\d{6})")
_FIG_LABEL_SIZE = 14
_FIG_TICK_SIZE = 12
_RNG = np.random.default_rng(0)


def _alnum_token_key(text: str) -> tuple:
    """Natural alphanumeric sort key (``2_`` before ``10_``)."""
    tokens: list[tuple[int, int | str]] = []
    for part in re.split(r"(\d+)", text):
        if part == "":
            continue
        tokens.append((0, int(part)) if part.isdigit() else (1, part.lower()))
    return tuple(tokens)


def epoch_key_from_folder(name: str) -> str | None:
    """Map a folder name to baseline1/flicker/baseline2 when recognizable."""
    lower = name.strip().lower()
    for cand in (lower, _LEADING_INDEX_RE.sub("", lower)):
        if cand and cand in EPOCH_ALIASES:
            return EPOCH_ALIASES[cand]
    return None


def group_display_label(group: str) -> str:
    """Visible label with an indexed folder's numeric prefix removed.

    The suffix is preserved exactly, including case: ``1_BL1`` becomes
    ``BL1`` and ``2_Flicker`` becomes ``Flicker``.
    """
    match = _NUMBERED_GROUP_RE.fullmatch(group.strip())
    if match is not None:
        return match.group("label")
    key = epoch_key_from_folder(group)
    return EPOCH_SHORT[key] if key is not None else group


def group_analysis_label(group: str) -> str:
    """Canonical flicker identity used by statistics, else the display label."""
    key = epoch_key_from_folder(group)
    return EPOCH_SHORT[key] if key is not None else group_display_label(group)


def _group_sort_key(group: str | None) -> tuple:
    """Alphanumeric folder order; ungrouped (None) last."""
    return (1, ()) if group is None else (0, _alnum_token_key(group))


def _row_sort_key(epoch_label: str) -> tuple:
    """Sort points/beats rows: B1 → Flicker → B2, then other labels alphanumerically."""
    short_order = tuple(EPOCH_SHORT.values())
    if epoch_label in short_order:
        return (0, short_order.index(epoch_label))
    return (1, _alnum_token_key(epoch_label))


def ordered_groups(groups: Iterable[str | None]) -> list[str]:
    """Unique named groups in figure x-axis order.

    Bare flicker folders (no leading index) keep B1→Flicker→B2. Indexed names
    such as ``1_baseline1`` follow alphanumeric order of the folder strings.
    """
    ordered = sorted({g for g in groups if g is not None}, key=_group_sort_key)
    if set(ordered) == set(EPOCH_ORDER):
        return list(EPOCH_ORDER)
    keys = [epoch_key_from_folder(g) for g in ordered]
    if set(keys) == set(EPOCH_ORDER) and all(
        key is not None and not _LEADING_INDEX_RE.match(g.strip())
        for g, key in zip(ordered, keys, strict=True)
    ):
        by_key = {key: g for g, key in zip(ordered, keys, strict=True)}
        return [by_key[epoch] for epoch in EPOCH_ORDER]
    return ordered


def is_flicker_triad(groups: Iterable[str | None]) -> bool:
    """True when folders encode baseline1 + flicker + baseline2."""
    keys = {epoch_key_from_folder(g) for g in groups if g is not None}
    keys.discard(None)
    return set(EPOCH_ORDER).issubset(keys)


def extract_patient_id(input_path: Path | str) -> str | None:
    """6-digit patient ID prefixing the ZIP/folder name, else None."""
    match = _PATIENT_ID_RE.match(Path(input_path).name)
    return match.group(1) if match else None


def prefixed_filename(filename: str, patient_id: str | None) -> str:
    """Prepend ``{patient_id}_`` when a patient ID is known."""
    return f"{patient_id}_{filename}" if patient_id else filename


def classify_cohort(
    h5_paths: Iterable[Path], input_root: Path
) -> tuple[list[tuple[str | None, Path]], list[str]]:
    """Group each H5 by its top-level folder under ``input_root``.

    Returns ``(records, group_order)``. Files sitting directly under the root
    get ``group=None``. Files outside the root are skipped.
    """
    records: list[tuple[str | None, Path]] = []
    input_root = Path(input_root)
    resolved_input_root = input_root.resolve()
    for h5_path in h5_paths:
        h5_path = Path(h5_path)
        try:
            # On Windows, ``resolve`` can turn a mapped drive (for example Z:)
            # into its UNC path. Normalize both sides before comparing so paths
            # that name the same network file are not incorrectly skipped.
            rel = h5_path.resolve().relative_to(resolved_input_root)
        except ValueError:
            continue
        group = rel.parts[0] if len(rel.parts) > 1 else None
        records.append((group, h5_path))
    records.sort(key=lambda r: (_group_sort_key(r[0]), r[1].name))
    return records, ordered_groups(group for group, _ in records)


def resolve_cohort_root(path: Path | str) -> Path:
    """Directory that directly contains the split folders.

    Unwraps a single child folder (typical after extracting a ZIP).
    """
    path = Path(path)
    if not path.is_dir():
        raise NotADirectoryError(f"Cohort root is not a directory: {path}")
    children = sorted(
        child
        for child in path.iterdir()
        if child.is_dir()
        and not child.name.startswith(".")
        and child.name != "__MACOSX"
    )
    if len(children) != 1:
        return path
    # A lone numbered child is the cohort's only group, not a ZIP wrapper.
    if _NUMBERED_GROUP_RE.fullmatch(children[0].name):
        return path
    return children[0]


class CohortLayoutError(ValueError):
    """The selected input does not use the numbered EyeFlow cohort layout."""


@dataclass(frozen=True)
class CohortInputLayout:
    """Validated EyeFlow inputs grouped by consecutively numbered folders."""

    root: Path
    h5_paths: tuple[Path, ...]
    group_order: tuple[str, ...]


@dataclass(frozen=True)
class CohortGroup:
    """One numbered cohort folder and its inferred analysis role."""

    folder: str
    index: int
    label: str
    role: str


@dataclass(frozen=True)
class CohortProtocol:
    """General protocol inferred from numbered folder labels.

    The folder suffix is always the public label. Roles are deliberately
    conservative: reference/intervention contrasts are enabled only when at
    least one suffix is an unambiguous baseline/control alias.
    """

    groups: tuple[CohortGroup, ...]
    kind: str
    contrast_available: bool

    @property
    def group_order(self) -> tuple[str, ...]:
        return tuple(group.folder for group in self.groups)

    @property
    def group_labels(self) -> tuple[str, ...]:
        return tuple(group.label for group in self.groups)

    @property
    def group_roles(self) -> tuple[str, ...]:
        return tuple(group.role for group in self.groups)

    def role_for(self, folder: str) -> str:
        for group in self.groups:
            if group.folder == folder:
                return group.role
        return "unclassified"


_REFERENCE_TOKEN_RE = re.compile(
    r"^(?:bl|b|base|baseline|ctrl|ctl|control|ref|reference)(?:group)?(?P<ordinal>[12])?$"
)
_NUMBER_WORDS = {"one": "1", "first": "1", "two": "2", "second": "2"}


def _normalized_role_token(label: str) -> str:
    """Case/separator/camel-insensitive token used only for role inference."""
    ascii_text = (
        unicodedata.normalize("NFKD", str(label))
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    # The compact form handles BL_1, baseline-1, BaselineOne and BASELINE1.
    compact = re.sub(r"[^a-z0-9]+", "", ascii_text.casefold())
    for word, digit in _NUMBER_WORDS.items():
        if compact.endswith(word):
            compact = compact[: -len(word)] + digit
            break
    return compact


def _reference_ordinal(label: str) -> int | None:
    """Return 1/2 for numbered aliases, 0 for plain aliases, else None."""
    match = _REFERENCE_TOKEN_RE.fullmatch(_normalized_role_token(label))
    if match is None:
        return None
    ordinal = match.group("ordinal")
    return int(ordinal) if ordinal else 0


def infer_cohort_protocol(groups: Iterable[str]) -> CohortProtocol:
    """Infer safe roles while supporting any positive number of groups.

    Examples include BL/CTRL aliases in any case, with separators or number
    words (``BL1``, ``baseline_one``, ``ControlTwo``). Unrecognized layouts
    remain valid and receive generic roles; their descriptive, omnibus and
    pairwise analyses still run without inventing a reference contrast.
    """
    parsed: list[tuple[int, str, str, int | None]] = []
    for fallback_index, folder in enumerate(groups, start=1):
        match = _NUMBERED_GROUP_RE.fullmatch(str(folder).strip())
        index = int(match.group("index")) if match is not None else fallback_index
        label = (
            match.group("label") if match is not None else group_display_label(folder)
        )
        parsed.append((index, str(folder), label, _reference_ordinal(label)))
    parsed.sort(key=lambda item: item[0])
    n_groups = len(parsed)
    if n_groups == 0:
        return CohortProtocol((), "empty", False)

    reference_positions = [
        position for position, item in enumerate(parsed) if item[3] is not None
    ]
    intervention_positions = [
        position for position, item in enumerate(parsed) if item[3] is None
    ]
    contrast_available = bool(reference_positions and intervention_positions)

    roles = [f"group_{position + 1}" for position in range(n_groups)]
    if n_groups == 1:
        roles[0] = "reference" if reference_positions else "single_group"
        kind = "single_group"
        contrast_available = False
    elif contrast_available:
        for sequence, position in enumerate(reference_positions, start=1):
            ordinal = parsed[position][3]
            if ordinal in (1, 2):
                roles[position] = f"reference_{ordinal}"
            elif len(reference_positions) == 1:
                roles[position] = "reference"
            else:
                roles[position] = f"reference_{sequence}"
        for sequence, position in enumerate(intervention_positions, start=1):
            roles[position] = (
                "intervention"
                if len(intervention_positions) == 1
                else f"intervention_{sequence}"
            )
        if n_groups == 2:
            kind = "reference_vs_intervention"
        elif n_groups == 3 and len(reference_positions) == 2:
            kind = "reference_intervention_reference"
        else:
            kind = "multi_group_reference_comparison"
    else:
        kind = f"unclassified_{n_groups}_group"

    return CohortProtocol(
        tuple(
            CohortGroup(folder=folder, index=index, label=label, role=role)
            for (index, folder, label, _marker), role in zip(parsed, roles, strict=True)
        ),
        kind,
        contrast_available,
    )


def _is_generated_output_h5(path: Path, root: Path) -> bool:
    """Exclude standard output trees when validating source EyeFlow H5s."""
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    generated_roots = {
        COHORT_RESULTS_DIRNAME.lower(),
        H5_OUTPUT_DIRNAME.lower(),
    }
    return bool(relative.parts and relative.parts[0].lower() in generated_roots)


def find_cohort_input_h5s(root: Path | str) -> list[Path]:
    """Return candidate source H5s, excluding earlier cohort products."""
    root = Path(root)
    return [
        path
        for path in find_hdf5_inputs(root)
        if not _is_generated_output_h5(path, root)
    ]


def eyeflow_h5_has_lowrank(path: Path | str) -> bool:
    """True only for an EyeFlow H5 containing packed low-rank metrics."""
    try:
        with h5py.File(path, "r") as handle:
            return find_eyeflow_lowrank_group(handle) is not None
    except OSError:
        return False


def validate_cohort_layout(
    h5_paths: Iterable[Path],
    input_root: Path | str,
) -> CohortInputLayout:
    """Validate the explicit ``1_label``, ``2_label`` EyeFlow folder contract.

    A valid cohort has one or more top-level group folders, numbered
    consecutively from 1. Every selected HDF5 file must sit below one of those
    folders and expose EyeFlow's low-rank metrics group. Acquisition/Holo trees,
    files directly in the root, and AngioEye result H5s are therefore rejected.
    """
    input_root = Path(input_root)
    unique_paths: list[Path] = []
    seen: set[str] = set()
    for path in h5_paths:
        candidate = Path(path)
        key = os.path.normcase(str(candidate.resolve()))
        if key not in seen:
            unique_paths.append(candidate)
            seen.add(key)

    if not unique_paths:
        raise CohortLayoutError(
            "no EyeFlow H5 files were found in numbered cohort folders"
        )

    incompatible = [path for path in unique_paths if not eyeflow_h5_has_lowrank(path)]
    if incompatible:
        names = ", ".join(path.name for path in incompatible[:3])
        remainder = len(incompatible) - 3
        if remainder > 0:
            names += f", and {remainder} more"
        raise CohortLayoutError(
            "all cohort inputs must be EyeFlow H5 files containing packed "
            f"low-rank metrics; incompatible file(s): {names}"
        )

    records, group_order = classify_cohort(unique_paths, input_root)
    if len(records) != len(unique_paths):
        raise CohortLayoutError("every EyeFlow H5 must be inside the selected folder")
    if any(group is None for group, _path in records):
        raise CohortLayoutError(
            "EyeFlow H5 files must be inside numbered group folders, not directly "
            "in the selected folder"
        )
    indexed_groups: list[tuple[int, str]] = []
    invalid_groups: list[str] = []
    for group in group_order:
        match = _NUMBERED_GROUP_RE.fullmatch(group)
        if match is None:
            invalid_groups.append(group)
            continue
        indexed_groups.append((int(match.group("index")), group))
    if invalid_groups:
        raise CohortLayoutError(
            "group folders must use the '<number>_<alphanumeric label>' format; "
            f"invalid folder(s): {', '.join(invalid_groups)}"
        )

    indexed_groups.sort(key=lambda item: item[0])
    indices = [index for index, _group in indexed_groups]
    expected = list(range(1, len(indexed_groups) + 1))
    if indices != expected:
        raise CohortLayoutError(
            "group folders must be uniquely and consecutively numbered from 1 "
            f"(found {indices})"
        )

    ordered_names = tuple(group for _index, group in indexed_groups)
    return CohortInputLayout(
        root=input_root,
        h5_paths=tuple(path for _group, path in records),
        group_order=ordered_names,
    )


def acqs_by_canonical_epochs(
    acqs_by_group: dict[str, list[dict]],
    protocol: CohortProtocol | None = None,
) -> dict[str, list[dict]]:
    """Map inferred reference/intervention roles onto the legacy triad schema."""
    out: dict[str, list[dict]] = {epoch: [] for epoch in EPOCH_ORDER}
    for group, acqs in acqs_by_group.items():
        role = protocol.role_for(group) if protocol is not None else ""
        if role == "reference_2":
            key = "baseline2"
        elif role.startswith("reference"):
            key = "baseline1"
        elif role.startswith("intervention"):
            key = "flicker"
        else:
            key = epoch_key_from_folder(group)
        if key in out:
            out[key].extend(acqs)
    return out


def _group_mask(df: pd.DataFrame, group: CohortGroup) -> pd.Series:
    """Select one group from new tables or older epoch-only frames."""
    if "group" in df.columns:
        return df["group"].astype(str) == group.folder
    if "epoch" not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    candidates = {
        group.folder,
        group.label,
        group_analysis_label(group.folder),
    }
    return df["epoch"].astype(str).isin(candidates)


def _legacy_epoch_for_role(role: str) -> str | None:
    """Translate a generalized role to the compatibility stats schema."""
    if role == "reference_2":
        return EPOCH_SHORT["baseline2"]
    if role.startswith("reference"):
        return EPOCH_SHORT["baseline1"]
    if role.startswith("intervention"):
        return EPOCH_SHORT["flicker"]
    return None


def _legacy_acquisitions_frame(
    acquisitions_df: pd.DataFrame,
    protocol: CohortProtocol,
) -> pd.DataFrame:
    """Return a role-labelled copy for the existing stats/confound columns."""
    frames: list[pd.DataFrame] = []
    for group in protocol.groups:
        epoch = _legacy_epoch_for_role(group.role)
        if epoch is None:
            key = epoch_key_from_folder(group.folder)
            epoch = EPOCH_SHORT[key] if key is not None else None
        if epoch is None:
            continue
        selected = acquisitions_df.loc[_group_mask(acquisitions_df, group)].copy()
        selected["epoch"] = epoch
        frames.append(selected)
    if not frames:
        return acquisitions_df.iloc[0:0].copy()
    return pd.concat(frames, ignore_index=True)


# =====================================================================
# Cohort H5: stats / confounds
# =====================================================================

COHORT_H5_ROOT = f"{ANGIOEYE_POSTPROCESS_ROOT}/lowrank_waveform_cohort"
COHORT_H5_BASENAME = "lowrank_cohort.h5"

# Internal acquisitions-table column, published symbol.
CANONICAL_ENDPOINTS = (
    ("A1", "A1"),
    ("A2", "A2"),
    ("R0", "R0"),
    ("R1", "R1"),
    ("R2", "R2"),
    ("rho1", "rho1"),
    ("rho2", "rho2"),
    ("MPR", "MPR"),
    ("Reff", "Reff"),
    ("PR", "PR"),
)
ENDPOINT_BY_METRIC = dict(CANONICAL_ENDPOINTS)

CONFUND_EIGHT = ("A1", "A2", "R1", "R2", "rho1", "rho2", "R0", "MPR")
CONFUND_SIX = ("Reff", "PR")

COHORT_DICTIONARY = {
    "columns": {
        "endpoint": "Published symbol for the endpoint.",
        "metric": "Internal acquisitions-table column name.",
        "group": "Exact numbered source-folder name.",
        "group_index": "Leading integer from the source-folder name.",
        "label": "Visible group label: the exact suffix after the first underscore.",
        "role": "Conservatively inferred reference/intervention role, or a generic role.",
        "n": "Finite observations contributing to a descriptive row.",
        "mean": "Arithmetic mean within one group.",
        "std": "Sample standard deviation within one group; NaN for fewer than two values.",
        "median": "Median within one group.",
        "q1": "25th percentile within one group.",
        "q3": "75th percentile within one group.",
        "iqr": "Interquartile range (q3 - q1) within one group.",
        "min": "Minimum within one group.",
        "max": "Maximum within one group.",
        "status": "Whether an omnibus test ran or the row is descriptive/insufficient only.",
        "kw_p_holm": "Kruskal-Wallis p, Holm-adjusted across the endpoint family.",
        "p_holm": "Pairwise p, Holm-adjusted across group pairs for this endpoint.",
        "cliffs_delta_b_vs_a": "Cliff's delta for group B relative to group A.",
        "B1": "Baseline1 median [IQR].",
        "Flicker": "Flicker median [IQR].",
        "B2": "Baseline2 median [IQR].",
        "n_B1": "Finite Baseline1 acquisition count.",
        "n_Flicker": "Finite Flicker acquisition count.",
        "n_B2": "Finite Baseline2 acquisition count.",
        "kw_H": "Kruskal-Wallis H across B1 / Flicker / B2.",
        "kw_p": "Kruskal-Wallis p across B1 / Flicker / B2.",
        "B1_vs_F": "Pairwise MWU raw p / Holm p (Holm among the 3 epoch pairs of this endpoint).",
        "F_vs_B2": "Pairwise MWU raw p / Holm p (Holm among the 3 epoch pairs of this endpoint).",
        "B1_vs_B2": "Pairwise MWU raw p / Holm p (Holm among the 3 epoch pairs of this endpoint).",
        "pooled_p": "Mann-Whitney p for Flicker vs pooled Baseline (B1+B2).",
        "pooled_p_holm": "Pooled Baseline-vs-Flicker p, Holm-adjusted across the 10-endpoint family.",
        "pooled_delta": "Cliff's delta for Flicker vs pooled Baseline.",
        "n_combinations": "Designed size of this endpoint's confound grid, not a missing-data count.",
        "n_significant": "Confound combinations with pooled p < 0.05.",
        "retained": "True when every confound combination has pooled p < 0.05 and one Cliff's delta sign.",
        "aggregation_retained": "Native values only; median vs mean; both SVD methods.",
        "beat_period_retained": "Default aggregation (median, or n/a); native vs residualized; both SVD methods.",
        "spearman_rho": (
            "Spearman correlation of the headline metric vs beat period T, "
            "on B1∪B2 only."
        ),
        "spearman_p": ("Spearman p-value vs beat period T, on B1∪B2 only."),
        "r_squared": (
            "OLS R² of the B1∪B2 fit of endpoint vs beat period T "
            "(the line used for residualization)."
        ),
        "r_squared_flicker": (
            "R² of the baseline-fitted line α + βT evaluated on flicker points "
            "only (1 − SS_res/SS_tot). NaN if the baseline fit is missing or "
            "there are fewer than two finite flicker pairs."
        ),
        "slope": "OLS slope of endpoint vs beat period T, fitted on B1∪B2 only.",
        "intercept": (
            "OLS intercept of endpoint vs beat period T, fitted on B1∪B2 only."
        ),
        "n_regression": (
            "Number of B1∪B2 acquisitions used in the OLS / Spearman fit."
        ),
        "svd_method": "Which SVD representation this confound row used.",
        "beat_aggregation": "How beats were collapsed to one acquisition value.",
        "beat_period_control": "Whether the metric was residualized against beat period.",
        "cliffs_delta": "Cliff's delta for Flicker vs pooled Baseline in this combination.",
    },
    "values": {
        "svd_method": {
            "joint": "Acquisition-level joint SVD.",
            "per_beat": "Per-beat SVD, then aggregated across beats.",
            "n/a": "This metric has no SVD axis in this row.",
        },
        "beat_aggregation": {
            "median": "Acquisition value is the median across beats.",
            "mean": "Acquisition value is the mean across beats.",
            "n/a": "No beat-aggregation choice (acquisition scalar).",
        },
        "beat_period_control": {
            "native": "Metric as computed, no period regression.",
            "residualized": (
                "Residual after subtracting α + βT, with α, β fitted on B1∪B2 only."
            ),
        },
        "endpoint": {
            "A1": "Mode-1 amplitude.",
            "A2": "Mode-2 amplitude.",
            "R0": "Total pulsatile RMS.",
            "R1": "Mode-1 residual RMS.",
            "R2": "Mode-2 residual RMS.",
            "rho1": "Mode-1 residual ratio R1/R0.",
            "rho2": "Mode-2 residual ratio R2/R0.",
            "MPR": "Mean-to-pulsatile ratio.",
            "Reff": "Effective rank.",
            "PR": "Participation ratio.",
        },
    },
}

_INT_COLUMNS = {
    "n_B1",
    "n_Flicker",
    "n_B2",
    "n_combinations",
    "n_significant",
    "n_regression",
    "acquisition",
    "beat_index",
    "n_valid_columns",
    "n_total_columns",
    "group_index",
    "n",
    "n_groups",
    "n_groups_with_data",
    "n_total",
    "n_a",
    "n_b",
}
_BOOL_COLUMNS = {
    "aggregation_retained",
    "beat_period_retained",
    "retained",
}
_FLOAT_COLUMNS = {
    "mean",
    "std",
    "median",
    "q1",
    "q3",
    "iqr",
    "min",
    "max",
    "kw_H",
    "kw_p",
    "kw_p_holm",
    "mannwhitney_U",
    "p",
    "p_holm",
    "cliffs_delta_b_vs_a",
}


class LowRankWaveformStatistics:
    """Nonparametric tests and the ten-row ``stats`` table."""

    CANONICAL_ENDPOINTS = CANONICAL_ENDPOINTS

    @staticmethod
    def clean(x) -> np.ndarray:
        """Drop non-finite entries so tests never see NaN/inf."""
        arr = np.asarray(x, dtype=float)
        return arr[np.isfinite(arr)]

    @staticmethod
    def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
        """Cliff's delta: Pr(X>Y) - Pr(X<Y). Positive means ``x`` tends larger."""
        x = LowRankWaveformStatistics.clean(x)
        y = LowRankWaveformStatistics.clean(y)
        if x.size == 0 or y.size == 0:
            return float("nan")
        diff = x[:, None] - y[None, :]
        return float(np.mean(diff > 0) - np.mean(diff < 0))

    @staticmethod
    def holm_adjust(p_values: list[float]) -> list[float]:
        """Holm-Bonferroni adjusted p-values, same order as the input.

        NaN entries stay NaN and are excluded from the family size.
        """
        p = np.asarray(p_values, dtype=float)
        n_valid = int(np.sum(np.isfinite(p)))
        order = np.argsort(np.where(np.isfinite(p), p, np.inf))
        adjusted = np.full(p.size, np.nan)
        running_max = 0.0
        rank = 0
        for idx in order:
            if not np.isfinite(p[idx]):
                continue
            running_max = max(running_max, (n_valid - rank) * p[idx])
            adjusted[idx] = min(running_max, 1.0)
            rank += 1
        return adjusted.tolist()

    @staticmethod
    def epoch_group_tests(epoch_values: dict[str, np.ndarray]) -> dict:
        """Kruskal-Wallis, pairwise MWU+Holm, and pooled Baseline-vs-Flicker tests."""
        b1 = LowRankWaveformStatistics.clean(epoch_values["baseline1"])
        fl = LowRankWaveformStatistics.clean(epoch_values["flicker"])
        b2 = LowRankWaveformStatistics.clean(epoch_values["baseline2"])
        pooled_baseline = np.concatenate([b1, b2])

        kw_H = np.nan
        kw_p = np.nan
        if b1.size >= 1 and fl.size >= 1 and b2.size >= 1:
            try:
                kw = kruskal(b1, fl, b2)
                kw_H = float(kw.statistic)
                kw_p = float(kw.pvalue)
            except ValueError:
                kw_H = np.nan
                kw_p = np.nan

        pairs = [
            ("baseline1", "flicker", b1, fl),
            ("flicker", "baseline2", fl, b2),
            ("baseline1", "baseline2", b1, b2),
        ]
        raw_p: list[float] = []
        raw_U: list[float] = []
        for _, _, x, y in pairs:
            if x.size >= 2 and y.size >= 2:
                result = mannwhitneyu(x, y, alternative="two-sided")
                raw_p.append(float(result.pvalue))
                raw_U.append(float(result.statistic))
            else:
                raw_p.append(float("nan"))
                raw_U.append(float("nan"))
        holm_p = LowRankWaveformStatistics.holm_adjust(raw_p)
        deltas = [LowRankWaveformStatistics.cliffs_delta(x, y) for _, _, x, y in pairs]

        pooled_p = np.nan
        pooled_delta = np.nan
        pooled_U = np.nan
        if pooled_baseline.size >= 2 and fl.size >= 2:
            pooled = mannwhitneyu(fl, pooled_baseline, alternative="two-sided")
            pooled_p = float(pooled.pvalue)
            pooled_U = float(pooled.statistic)
            pooled_delta = LowRankWaveformStatistics.cliffs_delta(fl, pooled_baseline)

        return {
            "kw_H": kw_H,
            "kw_p": kw_p,
            "pairwise": [
                {
                    "pair": f"{a} vs {b}",
                    "p": p,
                    "p_holm": ph,
                    "U": u,
                    "cliffs_delta": d,
                }
                for (a, b, _, _), p, ph, u, d in zip(
                    pairs, raw_p, holm_p, raw_U, deltas, strict=True
                )
            ],
            "pooled_p": pooled_p,
            "pooled_U": pooled_U,
            "pooled_delta": pooled_delta,
            "n": {"baseline1": b1.size, "flicker": fl.size, "baseline2": b2.size},
        }

    @staticmethod
    def format_median_iqr(vals: np.ndarray) -> str:
        """``median [IQR]`` with 3 significant figures."""
        vals = LowRankWaveformStatistics.clean(vals)
        if vals.size == 0:
            return "--"
        med = float(np.median(vals))
        iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
        return f"{med:.3g} [{iqr:.3g}]"

    @staticmethod
    def format_p_scientific(p: float) -> str:
        """Scientific / fixed formatting for a p-value."""
        if not np.isfinite(p):
            return "--"
        if p >= 1e-2:
            return f"{p:.3f}"
        exp = int(np.floor(np.log10(p)))
        mant = p / (10**exp)
        return f"{mant:.2f}e{exp}"

    @classmethod
    def format_raw_holm(cls, raw: float, holm: float) -> str:
        """``raw / Holm`` p-value pair."""
        return f"{cls.format_p_scientific(raw)} / {cls.format_p_scientific(holm)}"

    @classmethod
    def beat_period_regression(cls, acquisitions_df: pd.DataFrame) -> dict[str, dict]:
        """OLS / Spearman of each headline metric vs beat period, fitted on B1∪B2."""
        out: dict[str, dict] = {}
        for metric, _endpoint in CANONICAL_ENDPOINTS:
            empty = {
                "spearman_rho": float("nan"),
                "spearman_p": float("nan"),
                "r_squared": float("nan"),
                "r_squared_flicker": float("nan"),
                "slope": float("nan"),
                "intercept": float("nan"),
                "n_regression": 0,
            }
            if metric not in acquisitions_df.columns:
                out[metric] = empty
                continue
            values = {
                epoch: acquisitions_df.loc[
                    acquisitions_df["epoch"] == short, metric
                ].to_numpy(dtype=float)
                for epoch, short in EPOCH_SHORT.items()
            }
            periods = {
                epoch: acquisitions_df.loc[
                    acquisitions_df["epoch"] == short, "beat_period"
                ].to_numpy(dtype=float)
                for epoch, short in EPOCH_SHORT.items()
            }
            _, fit = LowRankWaveformConfounds.residualize_against_beat_period(
                values, periods
            )
            n = int(fit["n"])
            rho = p_rho = float("nan")
            if n >= 3:
                y = np.concatenate([values["baseline1"], values["baseline2"]])
                x = np.concatenate([periods["baseline1"], periods["baseline2"]])
                mask = np.isfinite(x) & np.isfinite(y)
                try:
                    spearman = spearmanr(x[mask], y[mask])
                    rho = float(spearman.statistic)
                    p_rho = float(spearman.pvalue)
                except ValueError:
                    pass
            out[metric] = {
                "spearman_rho": rho,
                "spearman_p": p_rho,
                "r_squared": float(fit["r_squared"]),
                "r_squared_flicker": float(fit["r_squared_flicker"]),
                "slope": float(fit["slope"]),
                "intercept": float(fit["intercept"]),
                "n_regression": n,
            }
        return out

    @classmethod
    def build_stats_table(
        cls,
        acquisitions_df: pd.DataFrame,
        confounds_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """One row per canonical endpoint: medians, tests, retain flags, regression."""
        retain = LowRankWaveformConfounds.retain_flags_by_metric(confounds_df)
        regression = cls.beat_period_regression(acquisitions_df)

        rows: list[dict] = []
        pooled_raw: list[float] = []
        for metric, endpoint in CANONICAL_ENDPOINTS:
            if metric not in acquisitions_df.columns:
                b1 = fl = b2 = np.asarray([], dtype=float)
            else:
                b1 = acquisitions_df.loc[
                    acquisitions_df["epoch"] == EPOCH_SHORT["baseline1"], metric
                ].to_numpy(dtype=float)
                fl = acquisitions_df.loc[
                    acquisitions_df["epoch"] == EPOCH_SHORT["flicker"], metric
                ].to_numpy(dtype=float)
                b2 = acquisitions_df.loc[
                    acquisitions_df["epoch"] == EPOCH_SHORT["baseline2"], metric
                ].to_numpy(dtype=float)
            tests = cls.epoch_group_tests(
                {"baseline1": b1, "flicker": fl, "baseline2": b2}
            )
            pair = {p["pair"]: p for p in tests["pairwise"]}
            b1_f = pair["baseline1 vs flicker"]
            f_b2 = pair["flicker vs baseline2"]
            b1_b2 = pair["baseline1 vs baseline2"]
            flags = retain.get(metric, {})
            fit = regression.get(metric, {})
            pooled_raw.append(tests["pooled_p"])
            rows.append(
                {
                    "endpoint": endpoint,
                    "metric": metric,
                    "B1": cls.format_median_iqr(b1),
                    "Flicker": cls.format_median_iqr(fl),
                    "B2": cls.format_median_iqr(b2),
                    "n_B1": int(tests["n"]["baseline1"]),
                    "n_Flicker": int(tests["n"]["flicker"]),
                    "n_B2": int(tests["n"]["baseline2"]),
                    "kw_H": tests["kw_H"],
                    "kw_p": tests["kw_p"],
                    "B1_vs_F": cls.format_raw_holm(b1_f["p"], b1_f["p_holm"]),
                    "F_vs_B2": cls.format_raw_holm(f_b2["p"], f_b2["p_holm"]),
                    "B1_vs_B2": cls.format_raw_holm(b1_b2["p"], b1_b2["p_holm"]),
                    "B1_vs_F_p": b1_f["p"],
                    "B1_vs_F_p_holm": b1_f["p_holm"],
                    "B1_vs_F_U": b1_f["U"],
                    "F_vs_B2_p": f_b2["p"],
                    "F_vs_B2_p_holm": f_b2["p_holm"],
                    "F_vs_B2_U": f_b2["U"],
                    "B1_vs_B2_p": b1_b2["p"],
                    "B1_vs_B2_p_holm": b1_b2["p_holm"],
                    "B1_vs_B2_U": b1_b2["U"],
                    "pooled_p": tests["pooled_p"],
                    "pooled_U": tests["pooled_U"],
                    "pooled_delta": tests["pooled_delta"],
                    "delta_F_B1": cls.cliffs_delta(cls.clean(fl), cls.clean(b1)),
                    "delta_F_B2": cls.cliffs_delta(cls.clean(fl), cls.clean(b2)),
                    "delta_F_pooled": tests["pooled_delta"],
                    "n_combinations": int(flags.get("n_combinations", 0)),
                    "n_significant": int(flags.get("n_significant", 0)),
                    "aggregation_retained": bool(
                        flags.get("aggregation_retained", False)
                    ),
                    "beat_period_retained": bool(
                        flags.get("beat_period_retained", False)
                    ),
                    "retained": bool(flags.get("retained", False)),
                    "spearman_rho": fit.get("spearman_rho", float("nan")),
                    "spearman_p": fit.get("spearman_p", float("nan")),
                    "r_squared": fit.get("r_squared", float("nan")),
                    "r_squared_flicker": fit.get("r_squared_flicker", float("nan")),
                    "slope": fit.get("slope", float("nan")),
                    "intercept": fit.get("intercept", float("nan")),
                    "n_regression": int(fit.get("n_regression", 0)),
                }
            )

        for row, holm in zip(rows, cls.holm_adjust(pooled_raw), strict=True):
            row["pooled_p_holm"] = holm

        column_order = [
            "endpoint",
            "metric",
            "B1",
            "Flicker",
            "B2",
            "n_B1",
            "n_Flicker",
            "n_B2",
            "kw_H",
            "kw_p",
            "B1_vs_F",
            "F_vs_B2",
            "B1_vs_B2",
            "B1_vs_F_p",
            "B1_vs_F_p_holm",
            "B1_vs_F_U",
            "F_vs_B2_p",
            "F_vs_B2_p_holm",
            "F_vs_B2_U",
            "B1_vs_B2_p",
            "B1_vs_B2_p_holm",
            "B1_vs_B2_U",
            "pooled_p",
            "pooled_p_holm",
            "pooled_U",
            "pooled_delta",
            "delta_F_B1",
            "delta_F_B2",
            "delta_F_pooled",
            "n_combinations",
            "n_significant",
            "aggregation_retained",
            "beat_period_retained",
            "retained",
            "spearman_rho",
            "spearman_p",
            "r_squared",
            "r_squared_flicker",
            "slope",
            "intercept",
            "n_regression",
        ]
        return pd.DataFrame(rows)[column_order]

    @classmethod
    def build_general_tables(
        cls,
        acquisitions_df: pd.DataFrame,
        protocol: CohortProtocol,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Build analyses that are well-defined for any number of groups.

        One-group cohorts receive descriptive rows and an explicit
        ``descriptive_only`` omnibus status. Two or more groups additionally
        receive Kruskal-Wallis and all-pairs Mann-Whitney/Cliff's-delta rows.
        """
        descriptive_columns = [
            "endpoint",
            "metric",
            "group_index",
            "group",
            "label",
            "role",
            "n",
            "mean",
            "std",
            "median",
            "q1",
            "q3",
            "iqr",
            "min",
            "max",
        ]
        omnibus_columns = [
            "endpoint",
            "metric",
            "n_groups",
            "n_groups_with_data",
            "n_total",
            "kw_H",
            "kw_p",
            "kw_p_holm",
            "status",
        ]
        pairwise_columns = [
            "endpoint",
            "metric",
            "group_a",
            "label_a",
            "role_a",
            "n_a",
            "group_b",
            "label_b",
            "role_b",
            "n_b",
            "mannwhitney_U",
            "p",
            "p_holm",
            "cliffs_delta_b_vs_a",
        ]

        descriptive_rows: list[dict] = []
        omnibus_rows: list[dict] = []
        pairwise_rows: list[dict] = []
        for metric, endpoint in CANONICAL_ENDPOINTS:
            group_values: list[tuple[CohortGroup, np.ndarray]] = []
            for group in protocol.groups:
                values = (
                    cls.clean(
                        acquisitions_df.loc[_group_mask(acquisitions_df, group), metric]
                    )
                    if metric in acquisitions_df.columns
                    else np.asarray([], dtype=float)
                )
                group_values.append((group, values))
                descriptive_rows.append(
                    {
                        "endpoint": endpoint,
                        "metric": metric,
                        "group_index": group.index,
                        "group": group.folder,
                        "label": group.label,
                        "role": group.role,
                        "n": int(values.size),
                        "mean": float(np.mean(values)) if values.size else np.nan,
                        "std": finite_std(values),
                        "median": float(np.median(values)) if values.size else np.nan,
                        "q1": float(np.percentile(values, 25))
                        if values.size
                        else np.nan,
                        "q3": float(np.percentile(values, 75))
                        if values.size
                        else np.nan,
                        "iqr": (
                            float(np.percentile(values, 75) - np.percentile(values, 25))
                            if values.size
                            else np.nan
                        ),
                        "min": float(np.min(values)) if values.size else np.nan,
                        "max": float(np.max(values)) if values.size else np.nan,
                    }
                )

            nonempty = [values for _group, values in group_values if values.size]
            kw_H = kw_p = np.nan
            if len(protocol.groups) == 1:
                status = "descriptive_only"
            elif len(nonempty) < 2:
                status = "insufficient_groups_with_data"
            else:
                status = "ok"
                concatenated = np.concatenate(nonempty)
                if concatenated.size and np.ptp(concatenated) == 0:
                    kw_H, kw_p = 0.0, 1.0
                else:
                    try:
                        result = kruskal(*nonempty)
                        kw_H, kw_p = float(result.statistic), float(result.pvalue)
                    except ValueError:
                        status = "test_not_defined"
            omnibus_rows.append(
                {
                    "endpoint": endpoint,
                    "metric": metric,
                    "n_groups": len(protocol.groups),
                    "n_groups_with_data": len(nonempty),
                    "n_total": int(sum(values.size for values in nonempty)),
                    "kw_H": kw_H,
                    "kw_p": kw_p,
                    "kw_p_holm": np.nan,
                    "status": status,
                }
            )

            endpoint_pair_rows: list[dict] = []
            raw_p: list[float] = []
            for (group_a, values_a), (group_b, values_b) in combinations(
                group_values, 2
            ):
                U = p = np.nan
                if values_a.size and values_b.size:
                    result = mannwhitneyu(values_a, values_b, alternative="two-sided")
                    U, p = float(result.statistic), float(result.pvalue)
                raw_p.append(p)
                endpoint_pair_rows.append(
                    {
                        "endpoint": endpoint,
                        "metric": metric,
                        "group_a": group_a.folder,
                        "label_a": group_a.label,
                        "role_a": group_a.role,
                        "n_a": int(values_a.size),
                        "group_b": group_b.folder,
                        "label_b": group_b.label,
                        "role_b": group_b.role,
                        "n_b": int(values_b.size),
                        "mannwhitney_U": U,
                        "p": p,
                        "p_holm": np.nan,
                        "cliffs_delta_b_vs_a": cls.cliffs_delta(values_b, values_a),
                    }
                )
            for row, adjusted in zip(
                endpoint_pair_rows, cls.holm_adjust(raw_p), strict=True
            ):
                row["p_holm"] = adjusted
            pairwise_rows.extend(endpoint_pair_rows)

        omnibus_adjusted = cls.holm_adjust([row["kw_p"] for row in omnibus_rows])
        for row, adjusted in zip(omnibus_rows, omnibus_adjusted, strict=True):
            row["kw_p_holm"] = adjusted

        return (
            pd.DataFrame(descriptive_rows, columns=descriptive_columns),
            pd.DataFrame(omnibus_rows, columns=omnibus_columns),
            pd.DataFrame(pairwise_rows, columns=pairwise_columns),
        )

    @classmethod
    def build_stats_tables(
        cls,
        collection: CohortCollection,
        confounds: LowRankWaveformConfounds,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        """Build ``stats`` / ``confounds`` frames for every cohort vessel.

        Non-triad cohorts retain the complete table schemas with unavailable
        epoch comparisons represented by zero counts and NaN values.
        """
        protocol = infer_cohort_protocol(collection.group_order)
        stats_by_vessel: dict[str, pd.DataFrame] = {}
        confounds_by_vessel: dict[str, pd.DataFrame] = {}

        for vessel, acquisitions_df in collection.points_by_vessel.items():
            grid = confounds.build_grid(
                acqs_by_canonical_epochs(collection.acqs_by_vessel[vessel], protocol)
            )
            legacy_df = _legacy_acquisitions_frame(acquisitions_df, protocol)
            stats_by_vessel[vessel] = cls.build_stats_table(legacy_df, grid)
            confounds_by_vessel[vessel] = grid
        return stats_by_vessel, confounds_by_vessel


class LowRankWaveformConfounds:
    """Confound-control grid: analysis-choice sweep plus retain flags."""

    @staticmethod
    def _first_finite_array(mapping: dict, *keys: str) -> np.ndarray:
        """First array among ``keys`` that contains a finite value."""
        for key in keys:
            if key not in mapping:
                continue
            arr = np.asarray(mapping[key], dtype=float)
            if np.isfinite(arr).any():
                return arr
        return np.asarray([], dtype=float)

    @staticmethod
    def residualize_against_beat_period(
        epoch_values: dict[str, np.ndarray],
        epoch_beat_periods: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """Fit ``d = α + βT`` on B1∪B2, then residualize every epoch with that line.

        Returns residuals and the baseline-only fit (slope, intercept, R², R² on
        flicker, N). Needs at least 3 finite baseline (value, period) pairs;
        otherwise returns the inputs and NaN diagnostics.
        """
        baseline_epochs = ("baseline1", "baseline2")
        base_values = np.concatenate(
            [
                np.asarray(epoch_values.get(epoch, []), dtype=float)
                for epoch in baseline_epochs
            ]
        )
        base_periods = np.concatenate(
            [
                np.asarray(epoch_beat_periods.get(epoch, []), dtype=float)
                for epoch in baseline_epochs
            ]
        )
        mask = np.isfinite(base_values) & np.isfinite(base_periods)
        n = int(np.sum(mask))
        empty_fit = {
            "slope": float("nan"),
            "intercept": float("nan"),
            "r_squared": float("nan"),
            "r_squared_flicker": float("nan"),
            "n": n,
        }
        native = {k: np.asarray(v, dtype=float) for k, v in epoch_values.items()}
        if n < 3:
            return native, empty_fit

        fit = linregress(base_periods[mask], base_values[mask])
        slope = float(fit.slope)
        intercept = float(fit.intercept)
        r_squared = float(fit.rvalue**2)

        flicker_values = np.asarray(epoch_values.get("flicker", []), dtype=float)
        flicker_periods = np.asarray(epoch_beat_periods.get("flicker", []), dtype=float)
        fl_mask = np.isfinite(flicker_values) & np.isfinite(flicker_periods)
        r_squared_flicker = float("nan")
        if int(np.sum(fl_mask)) >= 2:
            y = flicker_values[fl_mask]
            yhat = intercept + slope * flicker_periods[fl_mask]
            ss_res = float(np.sum((y - yhat) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            if ss_tot > 0:
                r_squared_flicker = 1.0 - ss_res / ss_tot

        out = {}
        for epoch, values in epoch_values.items():
            values = np.asarray(values, dtype=float)
            periods = np.asarray(epoch_beat_periods[epoch], dtype=float)
            out[epoch] = values - (slope * periods + intercept)
        return out, {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
            "r_squared_flicker": r_squared_flicker,
            "n": n,
        }

    def acq_dots(
        self,
        acqs: list[dict],
        metric: str,
        svd_method: str | None,
        stat: str,
    ) -> np.ndarray:
        """One acquisition-level value per acquisition for ``metric``."""
        out = []
        for a in acqs:
            beatwise = a.get("beatwise") or {}
            per_beat = a.get("per_beat_svd") or {}
            acq = a.get("acq") or {}
            if metric == "R0":
                if svd_method == "per_beat":
                    arr = self._first_finite_array(per_beat, "R0")
                else:
                    arr = self._first_finite_array(beatwise, "R0_b")
                out.append(aggregate_beatwise(arr, stat if stat != "n/a" else "median"))
            elif metric == "MPR":
                if svd_method == "per_beat":
                    arr = self._first_finite_array(per_beat, "MPR")
                else:
                    arr = self._first_finite_array(beatwise, "MPR_b")
                out.append(aggregate_beatwise(arr, stat if stat != "n/a" else "median"))
            elif metric in ("A1", "A2", "R1", "R2"):
                if svd_method == "per_beat":
                    arr = self._first_finite_array(per_beat, metric)
                else:
                    arr = self._first_finite_array(beatwise, f"{metric}_b")
                out.append(aggregate_beatwise(arr, stat if stat != "n/a" else "median"))
            elif metric in ("rho1", "rho2"):
                m = metric[-1]
                if svd_method == "per_beat":
                    R_b = self._first_finite_array(per_beat, f"R{m}")
                    tpr_b = self._first_finite_array(per_beat, "R0")
                else:
                    R_b = self._first_finite_array(beatwise, f"R{m}_b")
                    tpr_b = self._first_finite_array(beatwise, "R0_b")
                out.append(
                    aggregate_rho(R_b, tpr_b, stat if stat != "n/a" else "median")
                )
            elif metric in ("Reff", "PR"):
                if svd_method == "per_beat":
                    arr = self._first_finite_array(per_beat, metric)
                    out.append(
                        aggregate_beatwise(arr, stat if stat != "n/a" else "median")
                    )
                else:
                    out.append(float(acq.get(metric, np.nan)))
            else:
                raise ValueError(f"Unknown metric: {metric}")
        return np.asarray(out, dtype=float)

    def epoch_dots(
        self,
        acqs_by_epoch: dict[str, list[dict]],
        metric: str,
        svd_method: str | None,
        stat: str,
    ) -> dict[str, np.ndarray]:
        """``acq_dots`` per epoch, keyed by baseline1 / flicker / baseline2."""
        return {
            epoch: self.acq_dots(acqs_by_epoch[epoch], metric, svd_method, stat)
            for epoch in EPOCH_ORDER
        }

    @staticmethod
    def epoch_beat_periods(
        acqs_by_epoch: dict[str, list[dict]],
    ) -> dict[str, np.ndarray]:
        """Mean beat period per acquisition, the covariate for residualization."""
        return {
            epoch: np.array(
                [a["beat_period_mean"] for a in acqs_by_epoch[epoch]], dtype=float
            )
            for epoch in EPOCH_ORDER
        }

    def build_grid(
        self,
        acqs_by_epoch: dict[str, list[dict]],
    ) -> pd.DataFrame:
        """Sweep metric × SVD × aggregation × period-control into one table."""
        periods = self.epoch_beat_periods(acqs_by_epoch)
        grid_rows: list[dict] = []

        def run_combo(
            metric: str, svd_method: str | None, stat: str, regressed: bool
        ) -> dict:
            """Epoch-group tests for one grid cell, residualizing when asked."""
            values = self.epoch_dots(acqs_by_epoch, metric, svd_method, stat)
            if regressed:
                values, _fit = self.residualize_against_beat_period(values, periods)
            return LowRankWaveformStatistics.epoch_group_tests(values)

        def add_row(
            metric: str, svd_method: str | None, stat: str, regressed: bool
        ) -> None:
            """Append one flattened grid row."""
            tests = run_combo(metric, svd_method, stat, regressed)
            grid_rows.append(
                {
                    "endpoint": ENDPOINT_BY_METRIC[metric],
                    "metric": metric,
                    "svd_method": svd_method if svd_method is not None else "n/a",
                    "beat_aggregation": stat,
                    "beat_period_control": "residualized" if regressed else "native",
                    "kw_p": tests["kw_p"],
                    "pooled_p": tests["pooled_p"],
                    "cliffs_delta": tests["pooled_delta"],
                    "n_B1": tests["n"]["baseline1"],
                    "n_Flicker": tests["n"]["flicker"],
                    "n_B2": tests["n"]["baseline2"],
                }
            )

        for metric in CONFUND_EIGHT:
            for svd_method in SVD_METHODS:
                for stat in ("median", "mean"):
                    for regressed in (False, True):
                        add_row(metric, svd_method, stat, regressed)

        for metric in CONFUND_SIX:
            for regressed in (False, True):
                add_row(metric, "joint", "n/a", regressed)
            for stat in ("median", "mean"):
                for regressed in (False, True):
                    add_row(metric, "per_beat", stat, regressed)

        counts = pd.Series([r["metric"] for r in grid_rows]).value_counts()
        for row in grid_rows:
            row["n_combinations"] = int(counts[row["metric"]])

        column_order = [
            "endpoint",
            "metric",
            "svd_method",
            "beat_aggregation",
            "beat_period_control",
            "n_combinations",
            "kw_p",
            "pooled_p",
            "cliffs_delta",
            "n_B1",
            "n_Flicker",
            "n_B2",
        ]
        return pd.DataFrame(grid_rows)[column_order]

    @staticmethod
    def _combination_retained(rows: list[dict]) -> bool:
        """True when every row has pooled p < 0.05 and one Cliff's-delta sign."""
        if not rows:
            return False
        ps = [float(r["pooled_p"]) for r in rows]
        deltas = [float(r["cliffs_delta"]) for r in rows]
        n_significant = sum(1 for p in ps if np.isfinite(p) and p < 0.05)
        signed = [d for d in deltas if np.isfinite(d)]
        consistent = bool(signed) and (
            all(d > 0 for d in signed) or all(d < 0 for d in signed)
        )
        return bool(n_significant == len(rows) and consistent)

    @classmethod
    def retain_flags_by_metric(cls, confounds_df: pd.DataFrame) -> dict[str, dict]:
        """Per-metric combination counts and aggregation / period / overall retain."""
        out: dict[str, dict] = {}
        if confounds_df is None or confounds_df.empty:
            return out
        for metric, group in confounds_df.groupby("metric", sort=False):
            rows = group.to_dict("records")
            agg_rows = [
                r
                for r in rows
                if r["beat_period_control"] == "native"
                and r["beat_aggregation"] in ("median", "mean")
            ]
            period_rows = [
                r for r in rows if r["beat_aggregation"] in ("median", "n/a")
            ]
            ps = [float(r["pooled_p"]) for r in rows]
            out[str(metric)] = {
                "n_combinations": len(rows),
                "n_significant": int(sum(1 for p in ps if np.isfinite(p) and p < 0.05)),
                "aggregation_retained": cls._combination_retained(agg_rows),
                "beat_period_retained": cls._combination_retained(period_rows),
                "retained": cls._combination_retained(rows),
            }
        return out


# =====================================================================
# Figure annotations (p / Cliff's δ on Figs 6--8)
# =====================================================================


def _pooled_test(df: pd.DataFrame, metric: str) -> tuple[float, float]:
    """Mann-Whitney p and Cliff's δ for Flicker vs pooled Baseline (B1∪B2)."""
    baseline = LowRankWaveformStatistics.clean(
        df.loc[df["epoch"].isin(["B1", "B2"]), metric]
    )
    flicker = LowRankWaveformStatistics.clean(df.loc[df["epoch"] == "Flicker", metric])
    if baseline.size < 2 or flicker.size < 2:
        return float("nan"), float("nan")
    p = float(mannwhitneyu(baseline, flicker, alternative="two-sided").pvalue)
    return p, LowRankWaveformStatistics.cliffs_delta(flicker, baseline)


def _protocol_contrast_test(
    df: pd.DataFrame,
    metric: str,
    protocol: CohortProtocol,
) -> tuple[float, float]:
    """Pooled inferred intervention-vs-reference test for any group count."""
    reference_parts = [
        df.loc[_group_mask(df, group), metric].to_numpy(dtype=float)
        for group in protocol.groups
        if group.role.startswith("reference") and metric in df.columns
    ]
    intervention_parts = [
        df.loc[_group_mask(df, group), metric].to_numpy(dtype=float)
        for group in protocol.groups
        if group.role.startswith("intervention") and metric in df.columns
    ]
    reference = LowRankWaveformStatistics.clean(
        np.concatenate(reference_parts) if reference_parts else []
    )
    intervention = LowRankWaveformStatistics.clean(
        np.concatenate(intervention_parts) if intervention_parts else []
    )
    if reference.size < 2 or intervention.size < 2:
        return float("nan"), float("nan")
    result = mannwhitneyu(reference, intervention, alternative="two-sided")
    return (
        float(result.pvalue),
        LowRankWaveformStatistics.cliffs_delta(intervention, reference),
    )


def _format_p(p: float) -> str:
    """Figure annotation for a p-value."""
    if not np.isfinite(p):
        return "p=n/a"
    return f"p={p:.1e}" if p < 1e-3 else f"p={p:.3f}"


def _format_delta(delta: float) -> str:
    """Figure annotation for Cliff's δ."""
    if not np.isfinite(delta):
        return r"$\delta$=n/a"
    return rf"$\delta$={delta:+.2f}"


# =====================================================================
# Figs 4--8
# =====================================================================


class LowRankWaveformCohortFigures:
    """Article Figs. 4--8, written once per cohort (joint and ``_pb``)."""

    PANEL_SIZE = 2.5
    FLICKER_SHADE = "#add8e6"
    SPECTRUM_BASELINE = ("black", "-", "o")
    SPECTRUM_FLICKER = ("#555555", "--", "s")

    FIG5_PANELS = (
        ("beat_period", r"Beat period" "\n" r"$T$"),
        ("mu", r"Baseline level" "\n" r"$\mu$"),
        ("R0", r"Total Pulsatile RMS" "\n" r"$R_0$"),
        ("MPR", r"Mean-to-pulsatile ratio" "\n" r"MPR"),
    )
    FIG6_PANELS = (
        ("A1", r"Mode-1 amplitude" "\n" r"$A_1$"),
        ("A2", r"Mode-2 amplitude" "\n" r"$A_2$"),
        ("R1", r"Residual RMS" "\n" r"$R_1$"),
        ("R2", r"Residual RMS" "\n" r"$R_2$"),
    )
    FIG7_PANELS = (
        ("rho1", r"Residual ratio" "\n" r"$\rho_1$"),
        ("rho2", r"Residual ratio" "\n" r"$\rho_2$"),
        ("Reff", r"Effective rank" "\n" r"$R_{\mathrm{eff}}$"),
        ("PR", r"Participation ratio" "\n" r"PR"),
    )

    @classmethod
    def plot_all(
        cls,
        points_by_vessel: dict[str, pd.DataFrame],
        out_dir: Path,
        group_order: list[str],
        *,
        patient_id: str | None = None,
        beats_by_vessel: dict[str, pd.DataFrame] | None = None,
        points_by_vessel_per_beat: dict[str, pd.DataFrame] | None = None,
    ) -> list[Path]:
        """Write Figs. 4--8 (joint and ``_pb``) for the artery cohort."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        protocol = infer_cohort_protocol(group_order)
        grids = (
            ("fig6_nonsvd_endpoints.png", cls.FIG5_PANELS),
            ("fig7_lowrank_endpoints.png", cls.FIG6_PANELS),
            ("fig8_residual_spectrum_endpoints.png", cls.FIG7_PANELS),
        )
        written = [
            cls._plot_endpoint_grid(
                points_by_vessel,
                panels,
                out_dir / prefixed_filename(name, patient_id),
                group_order,
            )
            for name, panels in grids
        ]
        pb_points = points_by_vessel_per_beat or {}
        if _has_per_beat_endpoint_dots(pb_points):
            pb_grids = (
                ("fig6_nonsvd_endpoints_pb.png", cls.FIG5_PANELS),
                ("fig7_lowrank_endpoints_pb.png", cls.FIG6_PANELS),
                ("fig8_residual_spectrum_endpoints_pb.png", cls.FIG7_PANELS),
            )
            written.extend(
                cls._plot_endpoint_grid(
                    pb_points,
                    panels,
                    out_dir / prefixed_filename(name, patient_id),
                    group_order,
                )
                for name, panels in pb_grids
            )
        if _has_spectrum_modes(points_by_vessel.get("artery", pd.DataFrame())):
            written.append(
                cls._save_spectrum(
                    out_dir
                    / prefixed_filename("fig4_variance_fraction.png", patient_id),
                    points_by_vessel,
                    cumulative=False,
                    group_order=group_order,
                )
            )
            written.append(
                cls._save_spectrum(
                    out_dir
                    / prefixed_filename(
                        "fig4_variance_fraction_cumulative.png", patient_id
                    ),
                    points_by_vessel,
                    cumulative=True,
                    group_order=group_order,
                )
            )
            if protocol.contrast_available:
                written.append(
                    cls._save_spectrum_ratio(
                        out_dir
                        / prefixed_filename("fig5_spectrum_ratio.png", patient_id),
                        points_by_vessel,
                        cumulative=False,
                        group_order=group_order,
                    )
                )
                written.append(
                    cls._save_spectrum_ratio(
                        out_dir
                        / prefixed_filename(
                            "fig5_spectrum_ratio_cumulative.png", patient_id
                        ),
                        points_by_vessel,
                        cumulative=True,
                        group_order=group_order,
                    )
                )
        beats = beats_by_vessel or {}
        pb_spectrum_source = (
            pb_points
            if _has_spectrum_modes(pb_points.get("artery", pd.DataFrame()))
            else beats
        )
        if _has_spectrum_modes(pb_spectrum_source.get("artery", pd.DataFrame())):
            written.append(
                cls._save_spectrum(
                    out_dir
                    / prefixed_filename("fig4_variance_fraction_pb.png", patient_id),
                    pb_spectrum_source,
                    cumulative=False,
                    group_order=group_order,
                )
            )
            written.append(
                cls._save_spectrum(
                    out_dir
                    / prefixed_filename(
                        "fig4_variance_fraction_cumulative_pb.png", patient_id
                    ),
                    pb_spectrum_source,
                    cumulative=True,
                    group_order=group_order,
                )
            )
        return sorted(written, key=lambda path: path.name)

    @staticmethod
    def _style_axes(ax, *, tick_size: int = 9) -> None:
        """Hide the grid and set tick-label size."""
        ax.grid(False)
        ax.tick_params(axis="both", labelsize=tick_size)

    @classmethod
    def _draw_epoch_panel(
        cls, ax, df: pd.DataFrame, metric: str, group_order: list[str]
    ) -> None:
        """One endpoint panel: one box-and-whisker plus dots per group."""
        protocol = infer_cohort_protocol(group_order)
        display_labels = list(protocol.group_labels)
        if protocol.contrast_available:
            for idx, group in enumerate(protocol.groups):
                if group.role.startswith("intervention"):
                    ax.axvspan(idx - 0.5, idx + 0.5, color=cls.FLICKER_SHADE, zorder=0)
                    ax.axvline(
                        idx - 0.5, color="black", linestyle=":", linewidth=1.5, zorder=1
                    )
                    ax.axvline(
                        idx + 0.5, color="black", linestyle=":", linewidth=1.5, zorder=1
                    )

        for x0, group in enumerate(protocol.groups):
            vals = (
                df.loc[_group_mask(df, group), metric].dropna().to_numpy(dtype=float)
                if metric in df.columns
                else np.asarray([], dtype=float)
            )
            if vals.size == 0:
                continue
            jitter = (_RNG.random(vals.size) - 0.5) * 0.16
            med = float(np.nanmedian(vals))
            ax.boxplot(
                [vals],
                positions=[x0],
                widths=0.46,
                whis=1.5,
                showfliers=False,
                patch_artist=True,
                boxprops={
                    "facecolor": "white",
                    "edgecolor": "black",
                    "linewidth": 1.25,
                },
                whiskerprops={"color": "black", "linewidth": 1.25},
                capprops={"color": "black", "linewidth": 1.25},
                medianprops={"color": "black", "linewidth": 1.5},
                zorder=3,
            )
            ax.scatter(
                x0 + jitter, vals, s=20, color="black", edgecolors="none", zorder=5
            )
            ax.scatter(
                [x0],
                [med],
                s=49,
                facecolors="white",
                edgecolors="black",
                linewidths=1.4,
                zorder=6,
            )

        ax.set_xticks(range(len(display_labels)))
        ax.set_xticklabels(
            display_labels,
            fontsize=9,
            rotation=35 if len(display_labels) > 4 else 0,
            ha="right" if len(display_labels) > 4 else "center",
        )
        ax.set_xlim(-0.5, max(len(display_labels) - 0.5, 0.5))
        cls._style_axes(ax)

        if metric in df.columns and protocol.contrast_available:
            all_vals = df[metric].to_numpy(dtype=float)
            if np.isfinite(all_vals).any():
                p, delta = _protocol_contrast_test(df, metric, protocol)
                y_min = float(np.nanmin(all_vals))
                y_max = float(np.nanmax(all_vals))
                pad = 0.08 * (y_max - y_min if y_max > y_min else 1.0)
                ax.text(
                    0.03,
                    0.97,
                    f"{_format_p(p)}\n{_format_delta(delta)}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8.5,
                    color="black",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.78,
                        "pad": 1.2,
                    },
                )
                ax.set_ylim(y_min - pad, y_max + 2.2 * pad)

    @classmethod
    def _plot_endpoint_grid(
        cls,
        points_by_vessel: dict[str, pd.DataFrame],
        panels: tuple[tuple[str, str], ...],
        out_path: Path,
        group_order: list[str],
    ) -> Path:
        """Write a 1×N PNG of arterial endpoint panels."""
        out_path = Path(out_path)
        df = points_by_vessel.get("artery", pd.DataFrame())
        n_cols = len(panels)
        panel_width = max(cls.PANEL_SIZE, 0.55 * len(group_order))
        fig, axes = plt.subplots(
            1,
            n_cols,
            figsize=(panel_width * n_cols, cls.PANEL_SIZE),
            squeeze=False,
        )
        for col_idx, (metric, title) in enumerate(panels):
            ax = axes[0, col_idx]
            cls._draw_epoch_panel(ax, df, metric, group_order)
            ax.set_box_aspect(1)
            ax.set_title(title, fontsize=11)
            if col_idx == 0:
                ax.set_ylabel("Value", fontsize=11)
        fig.tight_layout(w_pad=1.0, h_pad=1.0)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    @classmethod
    def _draw_spectrum(
        cls,
        ax,
        df: pd.DataFrame,
        mode_cols: list[str],
        *,
        cumulative: bool,
        group_order: list[str] | None = None,
    ) -> None:
        """Draw a mean±SD singular-value curve for every numbered group."""
        if not mode_cols or df.empty or not ({"epoch", "group"} & set(df.columns)):
            return
        plot_modes = np.arange(1, len(mode_cols) + 1)
        identity = df["group"] if "group" in df.columns else df["epoch"]
        groups = group_order or ordered_groups(identity)
        protocol = infer_cohort_protocol(groups)
        color_map = plt.get_cmap("tab10")
        markers = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h")
        for position, group in enumerate(protocol.groups):
            mask = _group_mask(df, group)
            color = color_map(position % 10)
            marker = markers[position % len(markers)]
            vals = df.loc[mask, mode_cols].to_numpy(dtype=float)
            if vals.size == 0:
                continue
            if cumulative:
                vals = np.nancumsum(vals, axis=1)
            mean, lo, hi = mean_pm_std(vals, axis=0)
            if not cumulative:
                lo = np.maximum(lo, np.where(mean > 0, mean * 1e-6, 1e-12))
            ax.plot(
                plot_modes,
                mean,
                color=color,
                linestyle="-" if position % 2 == 0 else "--",
                marker=marker,
                label=group.label,
                linewidth=1.5,
                markersize=5,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.2,
            )
            ax.fill_between(plot_modes, lo, hi, color=color, alpha=0.12, linewidth=0)
        if len(protocol.groups) > 1:
            ax.legend(
                frameon=False,
                fontsize=8,
                ncol=max(1, min(3, (len(protocol.groups) + 5) // 6)),
            )

    @classmethod
    def _save_spectrum(
        cls,
        out_path: Path,
        beats_by_vessel: dict[str, pd.DataFrame] | None,
        *,
        cumulative: bool,
        group_order: list[str] | None = None,
    ) -> Path:
        """Write the per-mode or cumulative λ spectrum PNG."""
        out_path = Path(out_path)
        frames = beats_by_vessel or {}
        beats = frames.get("artery", pd.DataFrame())
        mode_cols = _spectrum_mode_cols(beats)
        df = beats if (not beats.empty and mode_cols) else pd.DataFrame()

        fig_h = 3.0
        fig, axes = plt.subplots(1, 1, figsize=(2.0 * fig_h, fig_h), squeeze=False)
        ax = axes[0, 0]
        cls._draw_spectrum(
            ax,
            df,
            mode_cols,
            cumulative=cumulative,
            group_order=group_order,
        )

        modes = np.arange(1, SPECTRUM_N_MODES + 1)
        ax.set_xticks(modes)
        ax.set_xticklabels([str(m) for m in modes])
        ax.set_xlim(0.5, SPECTRUM_N_MODES + 0.5)
        if not cumulative:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(
                LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0), numticks=8)
            )
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda y, _pos: f"{y:g}" if y > 0 else "0")
            )
            ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(r"$m$", fontsize=_FIG_LABEL_SIZE)
        ax.set_ylabel(
            r"$\sum_{i=1}^{m}\lambda_i$" if cumulative else r"$\lambda_m$",
            fontsize=_FIG_LABEL_SIZE,
        )
        ax.set_box_aspect(0.5)
        cls._style_axes(ax, tick_size=_FIG_TICK_SIZE)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    @staticmethod
    def _spectrum_ratio_values(
        df: pd.DataFrame,
        mode_cols: list[str],
        *,
        cumulative: bool,
        group_order: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return modes and percent flicker/baseline spectrum change."""
        modes = np.arange(1, len(mode_cols) + 1)
        if df.empty or not mode_cols or not ({"epoch", "group"} & set(df.columns)):
            return modes, np.full(len(mode_cols), np.nan, dtype=float)
        identity = df["group"] if "group" in df.columns else df["epoch"]
        groups = group_order or ordered_groups(identity)
        protocol = infer_cohort_protocol(groups)
        reference_mask = pd.Series(False, index=df.index, dtype=bool)
        intervention_mask = pd.Series(False, index=df.index, dtype=bool)
        for group in protocol.groups:
            if group.role.startswith("reference"):
                reference_mask |= _group_mask(df, group)
            elif group.role.startswith("intervention"):
                intervention_mask |= _group_mask(df, group)
        baseline = df.loc[reference_mask, mode_cols].to_numpy(dtype=float)
        flicker = df.loc[intervention_mask, mode_cols].to_numpy(dtype=float)
        if baseline.size == 0 or flicker.size == 0:
            return modes, np.full(len(mode_cols), np.nan, dtype=float)
        with np.errstate(all="ignore"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                baseline_mean = np.nanmean(baseline, axis=0)
                flicker_mean = np.nanmean(flicker, axis=0)
        if cumulative:
            baseline_mean = np.nancumsum(baseline_mean)
            flicker_mean = np.nancumsum(flicker_mean)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = flicker_mean / baseline_mean
        percent_change = 100.0 * (ratio - 1.0)
        percent_change = np.where(np.isfinite(percent_change), percent_change, np.nan)
        return modes, percent_change

    @classmethod
    def _save_spectrum_ratio(
        cls,
        out_path: Path,
        points_by_vessel: dict[str, pd.DataFrame] | None,
        *,
        cumulative: bool,
        group_order: list[str] | None = None,
    ) -> Path:
        """Write Fig. 5 spectrum ratio PNG from acquisition-level singular values."""
        out_path = Path(out_path)
        frames = points_by_vessel or {}
        df = frames.get("artery", pd.DataFrame())
        mode_cols = _spectrum_mode_cols(df)
        modes, percent_change = cls._spectrum_ratio_values(
            df,
            mode_cols,
            cumulative=cumulative,
            group_order=group_order,
        )

        fig_h = 3.0
        fig, axes = plt.subplots(1, 1, figsize=(2.0 * fig_h, fig_h), squeeze=False)
        ax = axes[0, 0]
        ax.axhline(0.0, color="black", linestyle=":", linewidth=1.1)
        ax.plot(
            modes,
            percent_change,
            color="black",
            linestyle="-",
            marker="o",
            linewidth=1.5,
            markersize=5,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.2,
        )

        finite = percent_change[np.isfinite(percent_change)]
        if finite.size:
            extent = max(5.0, float(np.nanmax(np.abs(finite))) * 1.2)
            ax.set_ylim(-extent, extent)
        ax.set_xticks(np.arange(1, SPECTRUM_N_MODES + 1))
        ax.set_xticklabels([str(m) for m in range(1, SPECTRUM_N_MODES + 1)])
        ax.set_xlim(0.5, SPECTRUM_N_MODES + 0.5)
        ax.set_xlabel(r"$m$", fontsize=_FIG_LABEL_SIZE)
        ax.set_ylabel(
            (r"$100(Q^{\mathrm{cum}}_m-1)$ (%)" if cumulative else r"$100(Q_m-1)$ (%)"),
            fontsize=_FIG_LABEL_SIZE,
        )
        ax.set_box_aspect(0.5)
        cls._style_axes(ax, tick_size=_FIG_TICK_SIZE)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path


# =====================================================================
# Collection from packed result H5s
# =====================================================================

PER_BEAT_POINT_METRICS = (
    "A1",
    "A2",
    "R1",
    "R2",
    "rho1",
    "rho2",
    "Reff",
    "PR",
)


def _spectrum_mode_cols(df: pd.DataFrame) -> list[str]:
    """``mode1``…``modeM`` columns present on ``df``."""
    if df is None or df.empty:
        return []
    return [
        f"mode{i}" for i in range(1, SPECTRUM_N_MODES + 1) if f"mode{i}" in df.columns
    ]


def _has_spectrum_modes(df: pd.DataFrame) -> bool:
    """True if ``df`` has a finite packed singular-value entry."""
    cols = _spectrum_mode_cols(df)
    if not cols:
        return False
    return bool(np.isfinite(df[cols].to_numpy(dtype=float)).any())


def _has_per_beat_endpoint_dots(points_by_vessel: dict[str, pd.DataFrame]) -> bool:
    """True if any vessel has a finite per-beat SVD acquisition endpoint."""
    for df in points_by_vessel.values():
        if df is None or df.empty:
            continue
        for metric in PER_BEAT_POINT_METRICS:
            if metric not in df.columns:
                continue
            if np.isfinite(df[metric].to_numpy(dtype=float)).any():
                return True
    return False


@dataclass
class CohortCollection:
    """Packed-H5 endpoints grouped for figures or ``lowrank_cohort.h5``."""

    points_by_vessel: dict[str, pd.DataFrame]
    beats_by_vessel: dict[str, pd.DataFrame]
    acqs_by_vessel: dict[str, dict[str, list[dict]]]
    group_order: list[str]
    patient_id: str | None
    flicker_protocol: bool
    source_files: list[str]
    points_by_vessel_per_beat: dict[str, pd.DataFrame]


def _finite_at(arr, index: int) -> float:
    """Finite value at ``index``, else NaN."""
    arr = np.asarray(arr, dtype=float)
    if index < arr.size and np.isfinite(arr[index]):
        return float(arr[index])
    return float("nan")


def _beat_get(mapping: dict, key: str, index: int) -> float:
    """Look up beat ``index`` of ``key``, else NaN."""
    return _finite_at(mapping[key], index) if key in mapping else float("nan")


def _beat_rows(
    vessel: str,
    h5_path: Path,
    sequence: int,
    epoch_short: str,
    vessel_data: dict,
) -> list[dict]:
    """One long-format row per beat (joint-SVD and per-beat-SVD columns)."""
    beatwise = vessel_data["beatwise"]
    per_beat_svd = vessel_data["per_beat_svd"]
    n_beats = 0
    for mapping, key in (
        (beatwise, "R0_b"),
        (beatwise, "A1_b"),
        (per_beat_svd, "R0"),
        (per_beat_svd, "A1"),
    ):
        if key in mapping:
            n_beats = len(np.asarray(mapping[key]))
            break
    vfb = vessel_data["valid_fraction_per_beat"]
    period_b = vessel_data["beat_period_b"]
    singular_b = coerce_beat_spectra(per_beat_svd.get("singular_values_b", []))
    emit_modes = is_usable_beat_spectra(singular_b)

    rows = []
    for b in range(n_beats):
        row = {
            "vessel": vessel,
            "acquisition": sequence,
            "file": h5_path.name,
            "epoch": epoch_short,
            "beat_index": b,
            "beat_period": _finite_at(period_b, b),
            "valid_fraction": _finite_at(vfb, b),
            "mu": _beat_get(beatwise, "mu_b", b),
            "R0": _beat_get(beatwise, "R0_b", b),
            "MPR": _beat_get(beatwise, "MPR_b", b),
            "A1": _beat_get(beatwise, "A1_b", b),
            "R1": _beat_get(beatwise, "R1_b", b),
            "rho1": _beat_get(beatwise, "rho1_b", b),
            "A2": _beat_get(beatwise, "A2_b", b),
            "R2": _beat_get(beatwise, "R2_b", b),
            "rho2": _beat_get(beatwise, "rho2_b", b),
            "A1_pb": _beat_get(per_beat_svd, "A1", b),
            "R1_pb": _beat_get(per_beat_svd, "R1", b),
            "A2_pb": _beat_get(per_beat_svd, "A2", b),
            "R2_pb": _beat_get(per_beat_svd, "R2", b),
            "rho1_pb": _beat_get(per_beat_svd, "rho1", b),
            "rho2_pb": _beat_get(per_beat_svd, "rho2", b),
            "R0_pb": _beat_get(per_beat_svd, "R0", b),
            "MPR_pb": _beat_get(per_beat_svd, "MPR", b),
            "Reff_pb": _beat_get(per_beat_svd, "Reff", b),
            "PR_pb": _beat_get(per_beat_svd, "PR", b),
        }
        if emit_modes:
            for m in range(1, SPECTRUM_N_MODES + 1):
                if b < singular_b.shape[0] and m <= singular_b.shape[1]:
                    val = float(singular_b[b, m - 1])
                    row[f"mode{m}"] = val if np.isfinite(val) else float("nan")
                else:
                    row[f"mode{m}"] = float("nan")
        rows.append(row)
    return rows


def _points_row(
    vessel: str,
    h5_path: Path,
    sequence: int,
    epoch_short: str,
    vessel_data: dict,
) -> dict:
    """One acquisition-level row (a single dot per endpoint)."""
    acq = vessel_data["acq"]
    vfb = vessel_data["valid_fraction_per_beat"]

    def scalar(key: str) -> float:
        return float(acq.get(key, np.nan))

    energy = np.asarray(vessel_data.get("energy_fraction", []), dtype=float)
    singular = np.asarray(vessel_data.get("singular_values", []), dtype=float)
    spectrum = singular if singular.size else energy
    row = {
        "vessel": vessel,
        "acquisition": sequence,
        "file": h5_path.name,
        "epoch": epoch_short,
        "A1": scalar("A1"),
        "A1_sd": scalar("sigma_A1_beat"),
        "R1": scalar("R1"),
        "rho1": scalar("rho1"),
        "rho1_sd": scalar("sigma_rho1_beat"),
        "A2": scalar("A2"),
        "A2_sd": scalar("sigma_A2_beat"),
        "R2": scalar("R2"),
        "rho2": scalar("rho2"),
        "rho2_sd": scalar("sigma_rho2_beat"),
        "R0": scalar("R0"),
        "R0_sd": scalar("sigma_R0_beat"),
        "MPR": scalar("MPR"),
        "MPR_sd": scalar("sigma_MPR_beat"),
        "Reff": scalar("Reff"),
        "PR": scalar("PR"),
        "beat_period": vessel_data["beat_period_mean"],
        "beat_period_sd": vessel_data["beat_period_sd"],
        "mu": scalar("mu_acq"),
        "mu_sd": scalar("sigma_mu_beat"),
        "valid_fraction": float(np.nanmean(vfb)) if vfb.size else float("nan"),
        "valid_fraction_sd": (
            float(np.nanstd(vfb, ddof=1)) if vfb.size > 1 else float("nan")
        ),
        "n_valid_columns": vessel_data["n_valid_columns"],
        "n_total_columns": vessel_data["n_total_columns"],
    }
    for m in range(1, SPECTRUM_N_MODES + 1):
        row[f"mode{m}"] = float(spectrum[m - 1]) if spectrum.size >= m else float("nan")
    return row


def _first_usable_array(mapping: dict, *keys: str) -> np.ndarray:
    """First per-beat array among ``keys`` that has a finite value."""
    for key in keys:
        if key not in mapping:
            continue
        arr = np.asarray(mapping[key], dtype=float)
        if np.isfinite(arr).any():
            return arr
    return np.asarray([], dtype=float)


def _median_per_beat(mapping: dict, *keys: str) -> float:
    """Median of the first usable per-beat array among ``keys``."""
    return aggregate_beatwise(_first_usable_array(mapping, *keys), "median")


def _std_per_beat(mapping: dict, *keys: str) -> float:
    """Sample SD of the first usable per-beat array among ``keys``."""
    arr = _first_usable_array(mapping, *keys)
    if not np.isfinite(arr).any():
        return float("nan")
    return finite_std(arr)


def _rho_from_per_beat(mapping: dict, mode: str) -> float:
    """Article ρ = median_b(R) / median_b(R0) from packed per-beat arrays."""
    return aggregate_rho(
        _first_usable_array(mapping, f"R{mode}"),
        _first_usable_array(mapping, "R0"),
        "median",
    )


def _points_row_per_beat(
    vessel: str,
    h5_path: Path,
    sequence: int,
    epoch_short: str,
    vessel_data: dict,
) -> dict:
    """Acquisition row from packed per-beat SVD endpoints.

    ``T``, ``μ``, R0, and MPR are not SVD-derived and stay the joint
    values. Mode amplitudes and residuals use ``median_b``; ρ uses
    ``median_b(R)/median_b(R0)``. Beat SDs are taken from the same
    per-beat arrays.
    """
    row = _points_row(vessel, h5_path, sequence, epoch_short, vessel_data)
    pb = vessel_data.get("per_beat_svd") or {}
    row["A1"] = _median_per_beat(pb, "A1")
    row["A2"] = _median_per_beat(pb, "A2")
    row["R1"] = _median_per_beat(pb, "R1")
    row["R2"] = _median_per_beat(pb, "R2")
    row["rho1"] = _rho_from_per_beat(pb, "1")
    row["rho2"] = _rho_from_per_beat(pb, "2")
    row["Reff"] = _median_per_beat(pb, "Reff")
    row["PR"] = _median_per_beat(pb, "PR")
    row["A1_sd"] = _std_per_beat(pb, "A1")
    row["A2_sd"] = _std_per_beat(pb, "A2")
    row["rho1_sd"] = _std_per_beat(pb, "rho1")
    row["rho2_sd"] = _std_per_beat(pb, "rho2")
    spectrum = np.asarray(vessel_data.get("per_beat_spectrum", []), dtype=float)
    if spectrum.size:
        for m in range(1, SPECTRUM_N_MODES + 1):
            row[f"mode{m}"] = (
                float(spectrum[m - 1]) if spectrum.size >= m else float("nan")
            )
    return row


def collect_payload(
    input_h5_paths: Iterable[Path],
    input_root: Path,
    *,
    veins: bool = True,
) -> CohortCollection:
    """Load packed result H5s into points/beats frames for figures or stats."""
    input_root = Path(input_root)
    records, group_order = classify_cohort(input_h5_paths, input_root)
    if not records:
        raise ValueError(f"No acquisition HDF5 files found under {input_root}.")

    protocol = infer_cohort_protocol(group_order)
    groups_by_folder = {group.folder: group for group in protocol.groups}
    group_keys = list(dict.fromkeys([*group_order, *EPOCH_ORDER]))
    vessels = enabled_vessels(bool(veins))
    acqs_by_vessel = {v: {g: [] for g in group_keys} for v in vessels}
    points_rows = {v: [] for v in vessels}
    points_rows_per_beat = {v: [] for v in vessels}
    beat_rows = {v: [] for v in vessels}
    sequence = {v: {g: 0 for g in group_keys} for v in vessels}
    flat_sequence = {v: 0 for v in vessels}

    for group, h5_path in records:
        data = load_acquisition_from_result_h5(h5_path, veins_flag=bool(veins))
        if data is None:
            continue
        for vessel in vessels:
            vessel_data = data.get(vessel)
            if vessel_data is None:
                continue
            if group is None:
                seq = flat_sequence[vessel]
                flat_sequence[vessel] += 1
                label = "ungrouped"
                group_index = 0
                role = "unclassified"
            else:
                acqs_by_vessel[vessel].setdefault(group, [])
                sequence[vessel].setdefault(group, 0)
                acqs_by_vessel[vessel][group].append(vessel_data)
                seq = sequence[vessel][group]
                sequence[vessel][group] += 1
                group_spec = groups_by_folder[group]
                label = group_spec.label
                group_index = group_spec.index
                role = group_spec.role
            row_identity = {
                "group": group or "ungrouped",
                "group_index": group_index,
                "group_label": label,
                "group_role": role,
            }
            point_row = _points_row(vessel, h5_path, seq, label, vessel_data)
            point_row.update(row_identity)
            points_rows[vessel].append(point_row)
            point_pb_row = _points_row_per_beat(
                vessel, h5_path, seq, label, vessel_data
            )
            point_pb_row.update(row_identity)
            points_rows_per_beat[vessel].append(point_pb_row)
            acquisition_beats = _beat_rows(vessel, h5_path, seq, label, vessel_data)
            for beat_row in acquisition_beats:
                beat_row.update(row_identity)
            beat_rows[vessel].extend(acquisition_beats)

    points_by_vessel: dict[str, pd.DataFrame] = {}
    points_by_vessel_per_beat: dict[str, pd.DataFrame] = {}
    beats_by_vessel: dict[str, pd.DataFrame] = {}
    kept_acqs: dict[str, dict[str, list[dict]]] = {}
    for vessel in vessels:
        rows = points_rows[vessel]
        if not rows:
            continue
        rows.sort(key=lambda r: (r["group_index"], r["acquisition"]))
        pb_rows = points_rows_per_beat[vessel]
        pb_rows.sort(key=lambda r: (r["group_index"], r["acquisition"]))
        beats = beat_rows[vessel]
        beats.sort(
            key=lambda r: (
                r["group_index"],
                r["acquisition"],
                r["beat_index"],
            )
        )
        points_by_vessel[vessel] = pd.DataFrame(rows)
        points_by_vessel_per_beat[vessel] = pd.DataFrame(pb_rows)
        beats_by_vessel[vessel] = pd.DataFrame(beats)
        kept_acqs[vessel] = acqs_by_vessel[vessel]

    if not points_by_vessel:
        raise ValueError("No vessel had any valid acquisitions; nothing to write.")

    return CohortCollection(
        points_by_vessel=points_by_vessel,
        beats_by_vessel=beats_by_vessel,
        acqs_by_vessel=kept_acqs,
        group_order=group_order,
        patient_id=extract_patient_id(input_root),
        flicker_protocol=is_flicker_triad(group_order),
        source_files=[str(path) for _, path in records],
        points_by_vessel_per_beat=points_by_vessel_per_beat,
    )


def _cohort_root_from_paths(
    h5_paths: list[Path], preferred_root: Path | None = None
) -> Path:
    """Pick a directory that exposes group folders for ``h5_paths``."""
    if preferred_root is not None and preferred_root.is_dir():
        return resolve_cohort_root(preferred_root)

    try:
        common = Path(os.path.commonpath([str(Path(p).resolve()) for p in h5_paths]))
    except ValueError:
        common = Path(h5_paths[0]).resolve().parent
    if common.is_file():
        common = common.parent

    probe = common
    for _ in range(5):
        records, _order = classify_cohort(h5_paths, probe)
        if any(group is not None for group, _ in records):
            return probe
        if probe.parent == probe:
            break
        probe = probe.parent
    return common


# =====================================================================
# lowrank_cohort.h5 writer
# =====================================================================


def _column_dtype(name: str, series: pd.Series):
    """HDF5-friendly dtype for one DataFrame column."""
    if name in _BOOL_COLUMNS or pd.api.types.is_bool_dtype(series):
        return np.bool_
    if name in _INT_COLUMNS or (
        pd.api.types.is_integer_dtype(series) and not pd.api.types.is_bool_dtype(series)
    ):
        return np.int64
    if name in _FLOAT_COLUMNS or pd.api.types.is_float_dtype(series):
        return np.float64
    return UTF8_STRING_DTYPE


def dataframe_to_compound(df: pd.DataFrame) -> np.ndarray:
    """Convert a DataFrame to a numpy structured array for one HDF5 dataset."""
    dtype = [(str(col), _column_dtype(str(col), df[col])) for col in df.columns]
    rec = np.empty(len(df), dtype=dtype)
    for col, field_dtype in dtype:
        series = df[col]
        kind = np.dtype(field_dtype).kind
        if kind == "b":
            rec[col] = series.fillna(False).astype(bool).to_numpy()
        elif kind in "iu":
            rec[col] = (
                pd.to_numeric(series, errors="coerce")
                .fillna(0)
                .astype(np.int64)
                .to_numpy()
            )
        elif kind == "f":
            rec[col] = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
        else:
            rec[col] = series.fillna("").astype(str).to_numpy()
    return rec


def write_compound_dataset(group: h5py.Group, name: str, df: pd.DataFrame) -> None:
    """Write ``df`` as a single compound dataset ``name`` (not a column group)."""
    if name in group:
        del group[name]
    rec = dataframe_to_compound(df)
    if rec.size == 0:
        if not rec.dtype.names:
            group.create_dataset(name, shape=(0,), dtype=np.uint8)
            return
        # HDF5 rejects a zero-length compound dataset containing variable-length
        # strings on some h5py/HDF5 builds. No strings are stored in an empty
        # table, so a one-byte fixed string preserves the schema safely.
        empty_dtype = np.dtype(
            [
                (field, "S1" if rec.dtype[field].kind == "O" else rec.dtype[field])
                for field in rec.dtype.names or ()
            ]
        )
        group.create_dataset(name, shape=(0,), dtype=empty_dtype)
    else:
        group.create_dataset(name, data=rec)


def dictionary_json() -> str:
    """Pretty JSON for the root ``dictionary`` dataset; must round-trip."""
    text = json.dumps(COHORT_DICTIONARY, indent=2, ensure_ascii=False)
    parsed = json.loads(text)
    if parsed != COHORT_DICTIONARY:
        raise ValueError("cohort dictionary JSON did not round-trip")
    return text


def validate_stats_confounds(
    stats_df: pd.DataFrame, confounds_df: pd.DataFrame
) -> None:
    """Refuse internally inconsistent stats / confounds tables."""
    values = COHORT_DICTIONARY["values"]
    coded = {
        "svd_method": confounds_df["svd_method"],
        "beat_aggregation": confounds_df["beat_aggregation"],
        "beat_period_control": confounds_df["beat_period_control"],
        "endpoint": pd.concat(
            [stats_df["endpoint"], confounds_df["endpoint"]], ignore_index=True
        ),
    }
    for field, series in coded.items():
        allowed = values[field]
        for value in series.astype(str).unique():
            if value not in allowed:
                raise ValueError(
                    f"coded value {field}={value!r} is missing from dictionary"
                )

    stats_endpoints = set(stats_df["endpoint"].astype(str))
    conf_endpoints = set(confounds_df["endpoint"].astype(str))
    missing = stats_endpoints - conf_endpoints
    if missing:
        raise ValueError(f"stats endpoints missing from confounds: {sorted(missing)}")
    for _, row in stats_df.iterrows():
        metric = str(row["metric"])
        n_declared = int(row["n_combinations"])
        n_rows = int((confounds_df["metric"] == metric).sum())
        if n_declared != n_rows:
            raise ValueError(
                f"{row['endpoint']}: n_combinations={n_declared} but "
                f"confounds has {n_rows} rows"
            )


def _epoch_counts(acquisitions_df: pd.DataFrame) -> dict[str, int]:
    """Finite-row counts per flicker epoch from the acquisitions table."""
    counts = {"n_B1": 0, "n_Flicker": 0, "n_B2": 0}
    if (
        acquisitions_df is None
        or acquisitions_df.empty
        or "epoch" not in acquisitions_df
    ):
        return counts
    value_counts = acquisitions_df["epoch"].astype(str).value_counts()
    counts["n_B1"] = int(value_counts.get(EPOCH_SHORT["baseline1"], 0))
    counts["n_Flicker"] = int(value_counts.get(EPOCH_SHORT["flicker"], 0))
    counts["n_B2"] = int(value_counts.get(EPOCH_SHORT["baseline2"], 0))
    return counts


def write_cohort_h5(
    output_path: Path,
    *,
    acquisitions_by_vessel: dict[str, pd.DataFrame],
    beats_by_vessel: dict[str, pd.DataFrame],
    group_order: list[str],
    patient_id: str | None = None,
    flicker_protocol: bool = False,
    source_files: list[str] | None = None,
    stats_by_vessel: dict[str, pd.DataFrame] | None = None,
    confounds_by_vessel: dict[str, pd.DataFrame] | None = None,
    group_statistics_by_vessel: dict[str, pd.DataFrame] | None = None,
    omnibus_statistics_by_vessel: dict[str, pd.DataFrame] | None = None,
    pairwise_statistics_by_vessel: dict[str, pd.DataFrame] | None = None,
) -> Path:
    """Write acquisitions/beats/stats/confounds as compound datasets.

    Layout under ``/AngioEye/Postprocessing/lowrank_waveform_cohort/``:

    * root attributes — provenance
    * ``dictionary`` — UTF-8 JSON of column / value meanings
    * ``{vessel}/acquisitions``, ``{vessel}/beats`` — compound tables
    * ``{vessel}/stats``, ``{vessel}/confounds`` — compatibility contrast tables
    * ``{vessel}/group_statistics`` — per-group descriptive statistics
    * ``{vessel}/omnibus_statistics`` — Kruskal-Wallis results (N >= 2)
    * ``{vessel}/pairwise_statistics`` — all-pairs MWU/Holm/effect sizes
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats_by_vessel = stats_by_vessel or {}
    confounds_by_vessel = confounds_by_vessel or {}
    group_statistics_by_vessel = group_statistics_by_vessel or {}
    omnibus_statistics_by_vessel = omnibus_statistics_by_vessel or {}
    pairwise_statistics_by_vessel = pairwise_statistics_by_vessel or {}
    protocol = infer_cohort_protocol(group_order)
    dict_text = dictionary_json()
    for vessel, stats_df in stats_by_vessel.items():
        if vessel not in confounds_by_vessel:
            raise ValueError(f"{vessel}: stats present but confounds missing")
        validate_stats_confounds(stats_df, confounds_by_vessel[vessel])

    if output_path.exists():
        output_path.unlink()

    n_acquisitions = (
        len(source_files)
        if source_files is not None
        else int(sum(len(df) for df in acquisitions_by_vessel.values()))
    )
    primary = None
    if "artery" in acquisitions_by_vessel:
        primary = acquisitions_by_vessel["artery"]
    elif acquisitions_by_vessel:
        primary = next(iter(acquisitions_by_vessel.values()))
    legacy_primary = _legacy_acquisitions_frame(
        primary if primary is not None else pd.DataFrame(), protocol
    )
    root_counts = _epoch_counts(legacy_primary)

    with open_h5(output_path, "w") as handle:
        root = handle.require_group(COHORT_H5_ROOT)
        set_attr_safe(root, "subject", "UNSET" if not patient_id else str(patient_id))
        set_attr_safe(
            root,
            "protocol",
            ("Baseline1 / Flicker / Baseline2" if flicker_protocol else protocol.kind),
        )
        set_attr_safe(root, "flicker_protocol", bool(flicker_protocol))
        set_attr_safe(root, "protocol_kind", protocol.kind)
        set_attr_safe(root, "contrast_available", protocol.contrast_available)
        set_attr_safe(root, "n_groups", len(protocol.groups))
        set_attr_safe(root, "n_acquisitions", int(n_acquisitions))
        set_attr_safe(root, "n_baseline1", int(root_counts["n_B1"]))
        set_attr_safe(root, "n_flicker", int(root_counts["n_Flicker"]))
        set_attr_safe(root, "n_baseline2", int(root_counts["n_B2"]))
        set_attr_safe(root, "svd_methods", list(SVD_METHODS))
        set_attr_safe(root, "beat_aggregations", ["median", "mean"])
        set_attr_safe(root, "group_order", list(group_order))
        set_attr_safe(root, "group_labels", list(protocol.group_labels))
        set_attr_safe(root, "group_roles", list(protocol.group_roles))
        set_attr_safe(root, "source_files", list(source_files or []))
        set_attr_safe(
            root,
            "generated",
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        write_value_dataset(root, "dictionary", dict_text)

        for vessel, acquisitions_df in acquisitions_by_vessel.items():
            vessel_group = root.require_group(str(vessel))
            counts = _epoch_counts(
                _legacy_acquisitions_frame(acquisitions_df, protocol)
            )
            set_attr_safe(vessel_group, "n_B1", counts["n_B1"])
            set_attr_safe(vessel_group, "n_Flicker", counts["n_Flicker"])
            set_attr_safe(vessel_group, "n_B2", counts["n_B2"])
            write_compound_dataset(vessel_group, "acquisitions", acquisitions_df)
            write_compound_dataset(
                vessel_group, "beats", beats_by_vessel.get(vessel, pd.DataFrame())
            )
            stats_df = stats_by_vessel.get(vessel)
            if stats_df is not None and not stats_df.empty:
                write_compound_dataset(vessel_group, "stats", stats_df)
            confounds_df = confounds_by_vessel.get(vessel)
            if confounds_df is not None and not confounds_df.empty:
                write_compound_dataset(vessel_group, "confounds", confounds_df)
            group_statistics_df = group_statistics_by_vessel.get(vessel)
            if group_statistics_df is not None:
                write_compound_dataset(
                    vessel_group, "group_statistics", group_statistics_df
                )
            omnibus_statistics_df = omnibus_statistics_by_vessel.get(vessel)
            if omnibus_statistics_df is not None:
                write_compound_dataset(
                    vessel_group, "omnibus_statistics", omnibus_statistics_df
                )
            pairwise_statistics_df = pairwise_statistics_by_vessel.get(vessel)
            if pairwise_statistics_df is not None:
                write_compound_dataset(
                    vessel_group, "pairwise_statistics", pairwise_statistics_df
                )

    return output_path


def write_stats_h5(
    collection: CohortCollection,
    confounds: LowRankWaveformConfounds,
    output_dir: Path,
) -> Path:
    """Pack ``collection`` into ``{patient_id_}lowrank_cohort.h5`` under output_dir."""
    stats_by_vessel, confounds_by_vessel = LowRankWaveformStatistics.build_stats_tables(
        collection, confounds
    )
    protocol = infer_cohort_protocol(collection.group_order)
    group_statistics_by_vessel: dict[str, pd.DataFrame] = {}
    omnibus_statistics_by_vessel: dict[str, pd.DataFrame] = {}
    pairwise_statistics_by_vessel: dict[str, pd.DataFrame] = {}
    for vessel, acquisitions_df in collection.points_by_vessel.items():
        group_table, omnibus_table, pairwise_table = (
            LowRankWaveformStatistics.build_general_tables(acquisitions_df, protocol)
        )
        group_statistics_by_vessel[vessel] = group_table
        omnibus_statistics_by_vessel[vessel] = omnibus_table
        pairwise_statistics_by_vessel[vessel] = pairwise_table
    return write_cohort_h5(
        Path(output_dir) / prefixed_filename(COHORT_H5_BASENAME, collection.patient_id),
        acquisitions_by_vessel=collection.points_by_vessel,
        beats_by_vessel=collection.beats_by_vessel,
        group_order=collection.group_order,
        patient_id=collection.patient_id,
        flicker_protocol=collection.flicker_protocol,
        source_files=collection.source_files,
        stats_by_vessel=stats_by_vessel,
        confounds_by_vessel=confounds_by_vessel,
        group_statistics_by_vessel=group_statistics_by_vessel,
        omnibus_statistics_by_vessel=omnibus_statistics_by_vessel,
        pairwise_statistics_by_vessel=pairwise_statistics_by_vessel,
    )


def _run_on_paths(
    h5_paths: list[Path],
    cohort_root: Path,
    output_dir: Path,
    *,
    veins: bool,
    patient_id: str | None = None,
) -> tuple[str, list[Path]]:
    """Validate and collect EyeFlow H5s, then write the cohort products."""
    layout = validate_cohort_layout(h5_paths, cohort_root)
    collection = collect_payload(
        layout.h5_paths,
        layout.root,
        veins=veins,
    )
    if collection.patient_id is None and patient_id is not None:
        collection.patient_id = patient_id
    generated_paths: list[Path] = []
    generated_paths.append(
        write_stats_h5(
            collection,
            LowRankWaveformConfounds(),
            h5_output_dir(output_dir),
        )
    )
    if collection.points_by_vessel:
        generated_paths.extend(
            LowRankWaveformCohortFigures.plot_all(
                collection.points_by_vessel,
                png_output_dir(output_dir),
                collection.group_order,
                patient_id=collection.patient_id,
                beats_by_vessel=collection.beats_by_vessel,
                points_by_vessel_per_beat=collection.points_by_vessel_per_beat,
            )
        )

    vessel_summaries = []
    for vessel, points_df in collection.points_by_vessel.items():
        acqs_by_group = collection.acqs_by_vessel[vessel]
        n_acq = len(points_df)
        group_counts = (
            ", ".join(
                f"{group_display_label(g)}={len(acqs_by_group.get(g, []))}"
                for g in collection.group_order
            )
            or f"ungrouped={n_acq}"
        )
        vessel_summaries.append(f"{vessel}={n_acq} ({group_counts})")
    split_note = f"cohort groups={collection.group_order}"
    n_rows = sum(len(df) for df in collection.points_by_vessel.values())
    h5_note = generated_paths[0].name if generated_paths else "no stats h5"
    summary = (
        f"Low-rank waveform run: {n_rows} "
        f"acquisition-vessel row(s) ({', '.join(vessel_summaries)}); "
        f"{split_note}; wrote {h5_note}."
    )
    return summary, generated_paths


def run(
    input_path: Path | str,
    output_dir: Path | str,
    *,
    veins: bool = True,
    result_h5_paths: Iterable[Path] | None = None,
) -> tuple[str, list[Path]]:
    """Build ``lowrank_cohort.h5`` and Figs 4--8 from grouped EyeFlow H5s.

    The folder (or extracted ZIP) must contain consecutively numbered group
    folders such as ``1_baseline1`` / ``2_flicker`` / ``3_baseline2`` or
    ``1_ctrl`` / ``2_path``. Any other input or layout raises
    :class:`CohortLayoutError` before an output directory is created.
    ``result_h5_paths`` remains available for callers that already discovered
    the EyeFlow files, but it is subject to the same validation.
    """
    input_path = Path(input_path).expanduser()
    output_dir = Path(output_dir).expanduser()

    if result_h5_paths is not None:
        h5_paths = [Path(path) for path in result_h5_paths]
        preferred = input_path if input_path.is_dir() else None
        if not h5_paths:
            raise CohortLayoutError(
                "no EyeFlow H5 files were provided for cohort postprocessing"
            )
        return _run_on_paths(
            h5_paths,
            _cohort_root_from_paths(h5_paths, preferred),
            output_dir,
            veins=veins,
        )

    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        with extracted_zip_tree(input_path) as extracted_root:
            cohort_root = resolve_cohort_root(extracted_root)
            return _run_on_paths(
                find_cohort_input_h5s(cohort_root),
                cohort_root,
                output_dir,
                veins=veins,
                patient_id=extract_patient_id(input_path),
            )

    if input_path.is_dir():
        cohort_root = resolve_cohort_root(input_path)
        return _run_on_paths(
            find_cohort_input_h5s(cohort_root),
            cohort_root,
            output_dir,
            veins=veins,
        )

    input_kind = (
        f"{input_path.suffix.lower()} file" if input_path.is_file() else "missing path"
    )
    raise CohortLayoutError(
        "low-rank cohort postprocessing accepts only a folder or .zip archive; "
        f"received {input_kind}: {input_path}"
    )


@registerPostprocess(
    name="Low-rank waveform cohort figures",
    description=(
        "From EyeFlow H5s arranged in consecutively numbered group folders, "
        "write ``lowrank_cohort.h5`` (stats, confounds, acquisitions, beats) "
        "under ``cohort-results/h5`` and cohort Figs. 4--8 (joint and _pb) "
        "under ``cohort-results/png``. This manual postprocess requires one or "
        "more folders such as ``1_ctrl`` or ``1_ctrl`` / ``2_path``. Any "
        "positive number of groups is supported."
    ),
    required_deps=[
        "numpy>=1.24",
        "pandas>=2.1",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "h5py>=3.8",
    ],
    accepted_input_modes=["folder", "zip"],
)
class LowRankWaveformCohortPostprocess(BatchPostprocess):
    veins_flag = False

    def run(self, context: PostprocessContext) -> PostprocessResult:
        """Registered entry: read the explicitly selected cohort folder/ZIP."""
        input_path = Path(context.input_path).expanduser()
        output_dir = cohort_results_dir(context.output_dir)

        summary, generated_paths = run(
            input_path,
            output_dir,
            veins=bool(self.veins_flag),
        )
        return PostprocessResult(
            summary=summary,
            generated_paths=[str(path) for path in generated_paths],
            metadata={
                "input_path": str(input_path),
                "output_dir": str(output_dir),
                "n_generated": len(generated_paths),
                "n_context_h5": len(context.input_h5_paths),
            },
        )
