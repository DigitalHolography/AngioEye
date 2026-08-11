"""AngioEye low-rank waveform decomposition: H5 adapter, stats, figures, cohort.

SVD math and metric packing live in EyeFlow (``calculator.py`` /
``outputs.py``). This module loads that packer (avoiding ``pipelines.*``
name collisions), then owns AngioEye's H5 I/O, group classification,
statistics, figures, confound sweeps, and cohort ``run()``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from input_output.archive_io import extracted_zip_tree
from input_output.hdf5_io import (
    create_h5_file,
    find_first_existing_path,
    write_metrics_trees_to_h5,
)
from input_output.hdf5_schema import ANGIOEYE_PROCESSING_ROOT
from input_output.inputs import find_hdf5_inputs, relative_hdf5_parent
from input_output.output_paths import h5_output_parent

from .core.base import (
    ProcessPipeline,
    ProcessResult,
    process_result_to_metrics_tree,
    registerPipeline,
)

import re
from collections.abc import Iterable

from scipy.stats import kruskal, mannwhitneyu
import matplotlib.pyplot as plt


T_INPUT = "Processing/VelocityPerBeat/BeatPeriodSeconds/value"
V_RAW_SEGMENT_INPUT_ARTERY = "Processing/VelocityPerBeat/Artery/Segments/Raw/value"
V_BAND_SEGMENT_INPUT_ARTERY = "Processing/VelocityPerBeat/Artery/Segments/BandLimited/value"
V_RAW_SEGMENT_INPUT_VEIN = "Processing/VelocityPerBeat/Vein/Segments/Raw/value"
V_BAND_SEGMENT_INPUT_VEIN = "Processing/VelocityPerBeat/Vein/Segments/BandLimited/value"
VESSEL_TYPES = ("artery", "vein")
# Figures are arterial-only; vein endpoints may still be computed for tables.
FIGURE_VESSELS = ("artery",)

# =====================================================================
# EyeFlow bootstrap (package-name collision with AngioEye ``pipelines``)
# =====================================================================

_EYEFLOW_LOWRANK_CACHE: dict | None = None


def _eyeflow_src() -> Path:
    env = os.environ.get("EYEFLOW_SRC")
    if env:
        return Path(env).expanduser().resolve()
    # AngioEye/src/pipelines/... -> Developer/EyeFlow/src
    developer = Path(__file__).resolve().parents[3]
    for name in ("EyeFlow", "Eyeflow"):
        candidate = (developer / name / "src").resolve()
        if (
            candidate
            / "pipelines"
            / "lowrank_waveform_decomposition"
            / "outputs.py"
        ).is_file():
            return candidate
    return (developer / "EyeFlow" / "src").resolve()


def _eyeflow_conflicting(name: str) -> bool:
    for prefix in (
        "pipelines",
        "input_output",
        "pipeline_engine",
        "app_settings",
        "dependency_utils",
    ):
        if name == prefix or name.startswith(prefix + "."):
            return True
    return False


def _load_eyeflow_module(qualname: str, path: Path):
    """Load a single EyeFlow file as ``qualname`` without package ``__init__``."""
    import importlib.util
    import types

    if path.is_dir():
        module = types.ModuleType(qualname)
        module.__file__ = str(path / "__init__.py")
        module.__path__ = [str(path)]
        module.__package__ = qualname
        sys.modules[qualname] = module
        return module

    spec = importlib.util.spec_from_file_location(qualname, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load EyeFlow module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualname] = module
    spec.loader.exec_module(module)
    return module


def _load_eyeflow_lowrank() -> dict:
    """Import EyeFlow packer + calculator, restoring AngioEye modules after.

    Loads calculator/outputs by file so we never execute EyeFlow's
    ``pipelines/__init__.py`` (which needs EyeFlow ``app_settings``) or the
    heavy ``input_output`` writers stack (skimage, etc.).
    """
    global _EYEFLOW_LOWRANK_CACHE
    if _EYEFLOW_LOWRANK_CACHE is not None:
        return _EYEFLOW_LOWRANK_CACHE

    eyeflow_src = _eyeflow_src()
    lr_dir = eyeflow_src / "pipelines" / "lowrank_waveform_decomposition"
    outputs_path = lr_dir / "outputs.py"
    calculator_path = lr_dir / "calculator.py"
    if not outputs_path.is_file() or not calculator_path.is_file():
        raise ImportError(
            "EyeFlow low-rank outputs not found under "
            f"{eyeflow_src}. Set EYEFLOW_SRC to EyeFlow's src directory."
        )

    saved = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if _eyeflow_conflicting(name)
    }
    src = str(eyeflow_src.resolve())
    inserted = False
    if src not in sys.path:
        sys.path.insert(0, src)
        inserted = True

    loaded_names: list[str] = []
    try:
        def _take(qualname: str, path: Path):
            module = _load_eyeflow_module(qualname, path)
            loaded_names.append(qualname)
            return module

        _take("dependency_utils", eyeflow_src / "dependency_utils.py")

        pe = _take("pipeline_engine", eyeflow_src / "pipeline_engine")
        pe_base = _take("pipeline_engine.base", eyeflow_src / "pipeline_engine" / "base.py")
        pe.DatasetValue = pe_base.DatasetValue
        pe.with_attrs = pe_base.with_attrs

        _take("input_output", eyeflow_src / "input_output")
        schema_pkg = _take(
            "input_output.schema",
            eyeflow_src / "input_output" / "schema",
        )
        eyeflow_output = _take(
            "input_output.schema.eyeflow_output",
            eyeflow_src / "input_output" / "schema" / "eyeflow_output.py",
        )
        schema_pkg.EyeFlowOutputPaths = eyeflow_output.EyeFlowOutputPaths

        _take("pipelines", eyeflow_src / "pipelines")
        _take("pipelines.lowrank_waveform_decomposition", lr_dir)
        calculator = _take(
            "pipelines.lowrank_waveform_decomposition.calculator",
            calculator_path,
        )
        outputs = _take(
            "pipelines.lowrank_waveform_decomposition.outputs",
            outputs_path,
        )

        output_root = (
            eyeflow_output.EyeFlowOutputPaths.active().lowrank_waveform_decomposition_root
        )
        _EYEFLOW_LOWRANK_CACHE = {
            "pack": outputs.pack_lowrank_waveform_decomposition_outputs,
            "Calculator": calculator.LowRankWaveformDecompositionCalculator,
            "normalize_periods": calculator.normalize_periods,
            "ensure_segment_shape": calculator.ensure_segment_shape,
            "output_root": str(output_root),
        }
    finally:
        for name in reversed(loaded_names):
            sys.modules.pop(name, None)
        for name in list(sys.modules):
            if not _eyeflow_conflicting(name):
                continue
            mod = sys.modules.get(name)
            file = (getattr(mod, "__file__", "") or "").replace("\\", "/")
            paths = [
                str(p).replace("\\", "/")
                for p in (getattr(mod, "__path__", None) or ())
            ]
            if src in file or any(src in p for p in paths):
                sys.modules.pop(name, None)
        sys.modules.update(saved)
        if inserted:
            try:
                sys.path.remove(src)
            except ValueError:
                pass

    return _EYEFLOW_LOWRANK_CACHE

# =====================================================================
# Group / epoch identity
# =====================================================================
# Known flicker-provocation aliases still normalize to the canonical triad
# (baseline1/flicker/baseline2) with short labels B1/Flicker/B2. Any other
# top-level subfolder name (e.g. ctrl, path) is kept as its own group key so
# cohort figures work for arbitrary splits, not only the flicker protocol.
EPOCHS = (("baseline1", "B1"), ("flicker", "Flicker"), ("baseline2", "B2"))
EPOCH_ORDER = tuple(key for key, _short in EPOCHS)
EPOCH_SHORT = dict(EPOCHS)
EPOCH_SHORT_TO_KEY = {short: key for key, short in EPOCHS}
EPOCH_SHORT_ORDER = tuple(short for _key, short in EPOCHS)
EPOCH_LABELS = {"B1": "B1", "Flicker": "F", "B2": "B2"}
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


def canonicalize_group_name(name: str) -> str:
    """Map a folder name to its canonical group key. Known flicker aliases
    become baseline1/flicker/baseline2; every other name is kept as-is
    (lowercased only when it matched an alias)."""
    return EPOCH_ALIASES.get(name.lower(), name)


def group_display_label(group: str) -> str:
    """Short axis/table label for a group key (B1/Flicker/B2 for the
    flicker triad; otherwise the folder name itself)."""
    return EPOCH_SHORT.get(group, group)


def _group_sort_key(group: str | None) -> tuple[int, int, str]:
    """Sort key: known flicker triad in canonical order, then other named
    groups alphabetically, then ungrouped (None) last."""
    if group is None:
        return (2, 0, "")
    if group in EPOCH_ORDER:
        return (0, EPOCH_ORDER.index(group), group)
    return (1, 0, group.lower())


def _epoch_rank(epoch_short: str) -> int:
    """Deprecated sort helper kept for call sites; prefer
    ``_row_group_sort_key``. Known short labels use the flicker triad
    order; arbitrary labels sort after those alphabetically via a
    secondary string key when paired with ``_row_group_sort_key``."""
    if epoch_short in EPOCH_SHORT_ORDER:
        return EPOCH_SHORT_ORDER.index(epoch_short)
    return len(EPOCH_SHORT_ORDER)


def _row_group_sort_key(epoch_label: str) -> tuple[int, int, str]:
    """Sort key for points/beats rows keyed by display label."""
    group = EPOCH_SHORT_TO_KEY.get(epoch_label, epoch_label)
    return _group_sort_key(group)


def ordered_groups(groups: Iterable[str | None]) -> list[str]:
    """Unique named groups (None dropped), sorted via ``_group_sort_key``."""
    named = {g for g in groups if g is not None}
    return sorted(named, key=_group_sort_key)


def is_flicker_triad(groups: Iterable[str | None]) -> bool:
    """True when the classic baseline1/flicker/baseline2 split is present
    (required for Sec. V.A/V.B confound tables that assume that protocol)."""
    return set(EPOCH_ORDER).issubset({g for g in groups if g is not None})


# =====================================================================
# Patient identification
# =====================================================================
PATIENT_ID_RE = re.compile(r"^(\d{6})")


def extract_patient_id(input_path: Path | str) -> str | None:
    """Returns the 6-digit patient ID prefixing the input ZIP/folder name,
    or None if the name isn't ID-prefixed."""
    match = PATIENT_ID_RE.match(Path(input_path).name)
    return match.group(1) if match else None


def prefixed_filename(filename: str, patient_id: str | None) -> str:
    """Prepends ``{patient_id}_`` to filename when a patient ID is known,
    leaving it unchanged otherwise."""
    return f"{patient_id}_{filename}" if patient_id else filename


# =====================================================================
# Vessel/schema resolution
# =====================================================================


def _enabled_vessels(veins_flag: bool) -> tuple[str, ...]:
    return ("artery", "vein") if veins_flag else ("artery",)


def _filter_vessel_candidates(
    candidates: dict[str, str], veins_flag: bool
) -> dict[str, str]:
    if veins_flag:
        return candidates
    return {k: v for k, v in candidates.items() if not k.startswith("vein/")}


def _resolve_vessel_sources(h5file, veins_flag: bool) -> tuple[dict[str, str], str] | None:
    if find_first_existing_path(h5file, [T_INPUT]) is None:
        return None
    candidates = {
        "artery/raw": V_RAW_SEGMENT_INPUT_ARTERY,
        "artery/bandlimited": V_BAND_SEGMENT_INPUT_ARTERY,
        "vein/raw": V_RAW_SEGMENT_INPUT_VEIN,
        "vein/bandlimited": V_BAND_SEGMENT_INPUT_VEIN,
    }
    return _filter_vessel_candidates(candidates, veins_flag), T_INPUT


# =====================================================================
# Acquisition/group classification -- the first path component under the
# cohort root is the group (baseline1, flicker, ctrl, path, ...). Flat
# acquisitions (h5 directly under the root) get group=None: they still
# receive acquisition-level outputs, but do not participate in cohort
# figures. Known flicker folder aliases are canonicalized via
# EPOCH_ALIASES; every other folder name is kept as its own group.
# =====================================================================


def classify_group(h5_path: Path, input_root: Path) -> str | None:
    """Return the group key for one acquisition, or None when the file sits
    directly under ``input_root`` (no split folder)."""
    h5_path = Path(h5_path)
    try:
        rel = h5_path.relative_to(input_root)
    except ValueError:
        return None
    parent_parts = rel.parts[:-1]
    if not parent_parts:
        return None
    return canonicalize_group_name(parent_parts[0])


def classify_epoch(h5_path: Path, input_root: Path) -> str | None:
    """Backward-compatible alias for :func:`classify_group`."""
    return classify_group(h5_path, input_root)


def classify_cohort(
    h5_paths: Iterable[Path], input_root: Path
) -> tuple[list[tuple[str | None, Path]], list[str]]:
    """Classify every acquisition by its top-level split folder under
    ``input_root``. Returns ``(records, group_order)`` where ``records`` is
    ``(group | None, h5_path)`` sorted by group then file name, and
    ``group_order`` lists the named groups present (length >= 2 means a
    cohort split that should get cohort-level figures). Flat files are
    kept with ``group=None`` rather than skipped."""
    records: list[tuple[str | None, Path]] = []
    for h5_path in h5_paths:
        h5_path = Path(h5_path)
        try:
            h5_path.relative_to(input_root)
        except ValueError:
            continue
        records.append((classify_group(h5_path, input_root), h5_path))

    records.sort(key=lambda r: (_group_sort_key(r[0]), r[1].name))
    group_order = ordered_groups(group for group, _ in records)
    return records, group_order


def resolve_cohort_root(path: Path | str) -> Path:
    """Resolve the cohort root that directly contains the split folders
    (e.g. baseline1/flicker/baseline2, or ctrl/path, or any other set of
    group subfolders). If ``path`` already has two or more child
    directories, it is returned as-is; if it has a single child directory (the
    usual layout after extracting ``260803_Flicker_EF.zip`` into a temp
    tree that wraps ``260803_Flicker_EF/``), that child is returned."""
    path = Path(path)
    if not path.is_dir():
        raise NotADirectoryError(f"Cohort root is not a directory: {path}")

    def _keep_child(child: Path) -> bool:
        name = child.name
        if not child.is_dir():
            return False
        # Ignore Finder metadata trees that macOS often injects into ZIPs.
        if name.startswith(".") or name == "__MACOSX":
            return False
        return True

    children = sorted(child for child in path.iterdir() if _keep_child(child))
    if len(children) >= 2:
        return path
    if len(children) == 1:
        return children[0]
    return path


# =====================================================================
# Thin AngioEye adapter around EyeFlow outputs.py packing
# =====================================================================


def _unwrap_packed_value(value):
    """Strip EyeFlow DatasetValue / (data, attrs) wrappers to raw arrays."""
    if isinstance(value, str):
        return value
    if hasattr(value, "data") and hasattr(value, "attrs"):
        return np.asarray(value.data)
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], dict):
        return np.asarray(value[0])
    return value if isinstance(value, (str, bytes)) else np.asarray(value)


def _angio_metrics_from_eyeflow_pack(
    packed: dict, output_root: str
) -> tuple[dict, list[str]]:
    """Map EyeFlow packed paths to AngioEye relative metrics keys."""
    root = output_root.rstrip("/")
    prefix = root + "/"
    metrics: dict = {}
    for key, value in packed.items():
        if not key.startswith(prefix):
            continue
        rel = key[len(prefix) :]
        metrics[rel] = _unwrap_packed_value(value)

    resolved: list[str] = []
    for source_name in (
        "artery/raw",
        "artery/bandlimited",
        "vein/raw",
        "vein/bandlimited",
    ):
        flag = metrics.get(f"{source_name}/qc/input_available")
        if flag is None:
            continue
        if int(np.asarray(flag).reshape(-1)[0]) == 1:
            resolved.append(source_name)
    return metrics, resolved


@registerPipeline(name="lowrank_waveform_decomposition")
class LowRankWaveformDecomposition(ProcessPipeline):
    """
    Low-rank SVD decomposition for beat-aligned arterial (and optionally
    venous) segment waveforms.

    SVD math and metric packing live in EyeFlow
    (``calculator.py`` / ``outputs.py``). This AngioEye class is a thin H5
    adapter plus cohort entry points: it feeds per-beat waveforms into
    EyeFlow's packer and writes the result under AngioEye's processing root.

    Vein processing is opt-in via ``veins_flag`` (default False).
    For full cohort regeneration use the module-level :func:`run`.
    """

    description = (
        "Joint low-rank waveform decomposition from beat-aligned arterial "
        "(and optionally venous) segment waveforms, reporting A1, rho1, A2, "
        "rho2, and TPR per acquisition and per beat, for raw and bandlimited "
        "signals."
    )

    veins_flag = False

    def __init__(self) -> None:
        super().__init__()
        ef = _load_eyeflow_lowrank()
        self._ef = ef
        self._pack = ef["pack"]
        self._calculator = ef["Calculator"]()
        self._output_root = ef["output_root"]

    @property
    def exported_modes(self) -> int:
        return int(self._calculator.exported_modes)

    @exported_modes.setter
    def exported_modes(self, value: int) -> None:
        self._calculator.exported_modes = int(value)

    @property
    def min_valid_samples_fraction(self) -> float:
        return float(self._calculator.min_valid_samples_fraction)

    @min_valid_samples_fraction.setter
    def min_valid_samples_fraction(self, value: float) -> None:
        self._calculator.min_valid_samples_fraction = float(value)

    @property
    def min_valid_columns(self) -> int:
        return int(self._calculator.min_valid_columns)

    @min_valid_columns.setter
    def min_valid_columns(self, value: int) -> None:
        self._calculator.min_valid_columns = int(value)

    @property
    def eps(self) -> float:
        return float(self._calculator.eps)

    def aggregate_beatwise(self, values_per_beat, stat: str) -> float:
        return self._calculator.aggregate_beatwise(values_per_beat, stat)

    def aggregate_rho(self, R_b, TPR_b, stat: str) -> float:
        return self._calculator.aggregate_rho(R_b, TPR_b, stat)

    def _normalize_T(self, T):
        return self._ef["normalize_periods"](T)

    def _ensure_segment_shape(self, v_block, T=None):
        return self._ef["ensure_segment_shape"](v_block, T)

    def _compute_representation(self, v_block, T):
        return self._calculator.compute(v_block, T)

    def per_beat_svd_panels(self, v_block, T):
        return self._calculator.per_beat_svd_panels(v_block, T)

    def _compute_per_beat_endpoints(self, panels):
        return self._calculator._compute_per_beat_endpoints(panels)

    def _build_attrs(self, representations: list[str], input_beat_period_path: str) -> dict:
        return {
            "pipeline_family": "low_rank_waveform_decomposition",
            "svd_method": "joint (t,bkr) SVD + per-beat (t,kr) SVD robustness variant",
            "aggregation": "median over (k,r), then median over b",
            "mode_panel_max": int(self.exported_modes),
            "vessels": list(_enabled_vessels(self.veins_flag)),
            "veins_flag": bool(self.veins_flag),
            "representations": representations,
            "primary_endpoints": ["A1", "rho1", "A2", "rho2"],
            "context_endpoint": "TPR",
            "input_beat_period_path": input_beat_period_path,
        }

    def _velocity_outputs_from_h5(
        self, h5file, candidates: dict[str, str], t_path: str
    ) -> dict[str, object]:
        """Build the shared per-beat dict EyeFlow's packer expects."""
        velocity_outputs: dict[str, object] = {
            t_path: np.asarray(h5file[t_path], dtype=float),
        }
        for dataset_path in candidates.values():
            if dataset_path in h5file:
                velocity_outputs[dataset_path] = np.asarray(
                    h5file[dataset_path], dtype=float
                )
        return velocity_outputs

    def _compute_source_metrics(
        self, h5file, candidates: dict[str, str], t_path: str
    ) -> tuple[dict, list[str]]:
        """Pack EyeFlow low-rank metrics for every candidate present in h5."""
        velocity_outputs = self._velocity_outputs_from_h5(
            h5file, candidates, t_path
        )
        packed = self._pack(
            velocity_outputs,
            vein_flag=bool(self.veins_flag),
        )
        return _angio_metrics_from_eyeflow_pack(packed, self._output_root)

    def run(self, h5file) -> ProcessResult:
        """Framework entry: pack EyeFlow low-rank metrics from an open h5."""
        legacy_candidates = _filter_vessel_candidates(
            {
                "artery/raw": V_RAW_SEGMENT_INPUT_ARTERY,
                "artery/bandlimited": V_BAND_SEGMENT_INPUT_ARTERY,
                "vein/raw": V_RAW_SEGMENT_INPUT_VEIN,
                "vein/bandlimited": V_BAND_SEGMENT_INPUT_VEIN,
            },
            self.veins_flag,
        )
        schema = _resolve_vessel_sources(h5file, self.veins_flag)
        if schema is None:
            metrics = {}
            for source_name, dataset_path in legacy_candidates.items():
                metrics[f"{source_name}/qc/input_available"] = np.asarray(
                    0, dtype=np.uint8
                )
                metrics[f"{source_name}/qc/missing_dataset_path"] = dataset_path
            attrs = self._build_attrs([], T_INPUT)
            return ProcessResult(metrics=metrics, attrs=attrs)

        candidates, t_path = schema
        metrics, resolved = self._compute_source_metrics(
            h5file, candidates, t_path
        )
        attrs = self._build_attrs(resolved, t_path)
        return ProcessResult(metrics=metrics, attrs=attrs)

    def compute_acquisition_endpoints(self, h5_path) -> dict | None:
        """In-memory endpoints for confounds/figures (EyeFlow calculator)."""
        with h5py.File(h5_path, "r") as h5file:
            schema = _resolve_vessel_sources(h5file, self.veins_flag)
            if schema is None:
                return None
            candidates, t_path = schema
            T = np.asarray(h5file[t_path], dtype=float)

            vessel_blocks: dict[str, np.ndarray | None] = {}
            for vessel in _enabled_vessels(self.veins_flag):
                raw_path = candidates.get(f"{vessel}/raw")
                if raw_path is not None and raw_path in h5file:
                    vessel_blocks[vessel] = np.asarray(h5file[raw_path], dtype=float)
                else:
                    vessel_blocks[vessel] = None

        if all(v_block is None for v_block in vessel_blocks.values()):
            return None

        result: dict[str, dict | None] = {}
        for vessel, v_block in vessel_blocks.items():
            if v_block is None:
                result[vessel] = None
                continue

            bundle = self._calculator.compute_acquisition(v_block, T)
            if (
                bundle is None
                or not bundle.get("representation", {}).get("svd_available", False)
            ):
                result[vessel] = None
                continue

            result[vessel] = {
                "acq": bundle["acq"],
                "beatwise": bundle["beatwise"],
                "per_beat_svd": bundle["per_beat_svd"],
                "mu": bundle["mu"],
                "energy_fraction": bundle["energy_fraction"],
                "beat_period_mean": bundle["beat_period_mean"],
                "beat_period_sd": bundle["beat_period_sd"],
                "beat_period_b": bundle["beat_period_b"],
                "valid_fraction_per_beat": bundle["valid_fraction_per_beat"],
                "n_valid_columns": bundle["n_valid_columns"],
                "n_total_columns": bundle["n_total_columns"],
            }

        return result

    def write_acquisition_h5(self, h5_path, out_path: Path | str) -> bool:
        """Standalone counterpart to run(): write packed metrics to out_path."""
        with h5py.File(h5_path, "r") as h5file:
            schema = _resolve_vessel_sources(h5file, self.veins_flag)
            if schema is None:
                return False
            candidates, t_path = schema
            metrics, resolved = self._compute_source_metrics(
                h5file, candidates, t_path
            )

        if not resolved:
            return False

        attrs = self._build_attrs(resolved, t_path)

        create_h5_file(out_path, source_file=str(h5_path), trim_source=True)
        metric_tree = process_result_to_metrics_tree(
            self.name, ProcessResult(metrics=metrics, attrs=attrs)
        )
        write_metrics_trees_to_h5(
            out_path, ANGIOEYE_PROCESSING_ROOT, [metric_tree], overwrite=False
        )
        return True


class LowRankWaveformStatistics:
    """Sec. V.A nonparametric test bundle and Sec. V.C endpoint tables."""

    TABLE_METRICS = ["rho2", "A2", "rho1", "A1", "TPR", "mpr", "mpr_prime", "alpha", "G1"]

    # Article Table I: eleven predefined endpoints (Holm family) plus exploratory G1.
    # ``metric`` is the points-table column; ``label`` is the published symbol.
    TABLE1_METRICS = (
        ("A1", "A1"),
        ("A2", "A2"),
        ("TPR", "R0"),
        ("R1", "R1"),
        ("R2", "R2"),
        ("rho1", "rho1"),
        ("rho2", "rho2"),
        ("mpr", "MPR"),
        ("effective_rank", "Reff"),
        ("participation_ratio", "PR"),
        ("alpha", "alpha"),
    )
    TABLE1_EXPLORATORY_METRICS = (("G1", "G1"),)

    @staticmethod
    def clean(x) -> np.ndarray:
        """Return ``x`` as a 1-D float array with non-finite (NaN/inf) entries
        dropped, so downstream tests never see missing values."""
        arr = np.asarray(x, dtype=float)
        return arr[np.isfinite(arr)]

    @staticmethod
    def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
        """Cliff's delta effect size: Pr(X>Y) - Pr(X<Y), in [-1, 1]. A positive
        delta means values in ``x`` tend to exceed those in ``y``. Returns NaN if
        either sample is empty after cleaning."""
        x = LowRankWaveformStatistics.clean(x)
        y = LowRankWaveformStatistics.clean(y)
        if x.size == 0 or y.size == 0:
            return float("nan")
        diff = x[:, None] - y[None, :]
        return float(np.mean(diff > 0) - np.mean(diff < 0))

    @staticmethod
    def holm_adjust(p_values: list[float]) -> list[float]:
        """Holm-Bonferroni step-down correction, returning adjusted p-values in the
        same order as the input. NaN entries stay NaN and are excluded from the
        family size, so missing tests don't inflate the correction applied to the
        rest; adjusted values are monotone non-decreasing and capped at 1.0."""
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
        """Sec. V.A nonparametric test bundle for one metric across the three
        epochs (baseline1/flicker/baseline2 arrays of acquisition-level dots).

        Returns a dict with: the Kruskal-Wallis p-value across all three epochs;
        a ``pairwise`` list of the three two-sided Mann-Whitney U tests (raw p,
        Holm-adjusted p, Cliff's delta, in baseline-first order); the pooled
        baseline (B1+B2) vs flicker p-value and delta; and per-epoch sample sizes.
        Tests that lack enough data are reported as NaN rather than raising."""
        b1 = LowRankWaveformStatistics.clean(epoch_values["baseline1"])
        fl = LowRankWaveformStatistics.clean(epoch_values["flicker"])
        b2 = LowRankWaveformStatistics.clean(epoch_values["baseline2"])
        pooled_baseline = np.concatenate([b1, b2])

        kw_p = np.nan
        if b1.size >= 1 and fl.size >= 1 and b2.size >= 1:
            try:
                kw_p = float(kruskal(b1, fl, b2).pvalue)
            except ValueError:
                kw_p = np.nan

        pairs = [
            ("baseline1", "flicker", b1, fl),
            ("flicker", "baseline2", fl, b2),
            ("baseline1", "baseline2", b1, b2),
        ]
        raw_p = [
            float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
            if x.size >= 2 and y.size >= 2
            else np.nan
            for _, _, x, y in pairs
        ]
        holm_p = LowRankWaveformStatistics.holm_adjust(raw_p)
        deltas = [LowRankWaveformStatistics.cliffs_delta(x, y) for _, _, x, y in pairs]

        pooled_p = np.nan
        pooled_delta = np.nan
        if pooled_baseline.size >= 2 and fl.size >= 2:
            pooled_p = float(
                mannwhitneyu(fl, pooled_baseline, alternative="two-sided").pvalue
            )
            pooled_delta = LowRankWaveformStatistics.cliffs_delta(fl, pooled_baseline)

        return {
            "kruskal_wallis_p": kw_p,
            "pairwise": [
                {"pair": f"{a} vs {b}", "p": p, "p_holm": ph, "cliffs_delta": d}
                for (a, b, _, _), p, ph, d in zip(pairs, raw_p, holm_p, deltas)
            ],
            "pooled_baseline_vs_flicker_p": pooled_p,
            "pooled_baseline_vs_flicker_delta": pooled_delta,
            "n": {"baseline1": b1.size, "flicker": fl.size, "baseline2": b2.size},
        }

    @classmethod
    def build_endpoint_table(cls, vessel: str, points_df: pd.DataFrame) -> pd.DataFrame:
        """Build the Table I/II summary (one row per metric in TABLE_METRICS) from
        a vessel's acquisition-level points frame: per-epoch median/SD/n, the
        Kruskal-Wallis and Holm-adjusted pairwise p-values, the pooled
        baseline-vs-flicker p-value, and flicker-vs-baseline Cliff's deltas."""
        rows = []
        for metric in cls.TABLE_METRICS:
            b1 = points_df.loc[points_df["epoch"] == "B1", metric].to_numpy(dtype=float)
            fl = points_df.loc[points_df["epoch"] == "Flicker", metric].to_numpy(
                dtype=float
            )
            b2 = points_df.loc[points_df["epoch"] == "B2", metric].to_numpy(dtype=float)
            tests = cls.epoch_group_tests(
                {"baseline1": b1, "flicker": fl, "baseline2": b2}
            )
            pair_lookup = {p["pair"]: p for p in tests["pairwise"]}

            b1c, flc, b2c = cls.clean(b1), cls.clean(fl), cls.clean(b2)
            pooled_baseline = np.concatenate([b1c, b2c])

            rows.append(
                {
                    "vessel": vessel,
                    "metric": metric,
                    "B1_median": np.nanmedian(b1) if b1.size else np.nan,
                    "B1_sd": np.nanstd(b1, ddof=1) if b1c.size > 1 else np.nan,
                    "n_B1": int(b1c.size),
                    "Flicker_median": np.nanmedian(fl) if fl.size else np.nan,
                    "Flicker_sd": np.nanstd(fl, ddof=1) if flc.size > 1 else np.nan,
                    "n_Flicker": int(flc.size),
                    "B2_median": np.nanmedian(b2) if b2.size else np.nan,
                    "B2_sd": np.nanstd(b2, ddof=1) if b2c.size > 1 else np.nan,
                    "n_B2": int(b2c.size),
                    "kruskal_wallis_p": tests["kruskal_wallis_p"],
                    "B1_vs_F_p_holm": pair_lookup["baseline1 vs flicker"]["p_holm"],
                    "F_vs_B2_p_holm": pair_lookup["flicker vs baseline2"]["p_holm"],
                    "B1_vs_B2_p_holm": pair_lookup["baseline1 vs baseline2"]["p_holm"],
                    "pooled_baseline_vs_flicker_p": tests["pooled_baseline_vs_flicker_p"],
                    # Flicker-first convention (matches paper's "Cliff's delta (F vs ...)"
                    # columns), NOT the epoch_group_tests pairwise list's (baseline,
                    # flicker) order, whose first-pair delta sign is flipped relative
                    # to this.
                    "cliffs_delta_F_vs_B1": cls.cliffs_delta(flc, b1c),
                    "cliffs_delta_F_vs_B2": cls.cliffs_delta(flc, b2c),
                    "cliffs_delta_F_vs_pooled_baseline": cls.cliffs_delta(
                        flc, pooled_baseline
                    ),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def format_median_iqr(vals: np.ndarray) -> str:
        """``median [IQR]`` with 3 significant figures (paper Table I style)."""
        vals = LowRankWaveformStatistics.clean(vals)
        if vals.size == 0:
            return "--"
        med = float(np.median(vals))
        iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
        return f"{med:.3g} [{iqr:.3g}]"

    @staticmethod
    def format_p_scientific(p: float) -> str:
        """Scientific / fixed formatting matching the paper's raw-p / p_H cells."""
        if not np.isfinite(p):
            return "--"
        if p >= 1e-2:
            return f"{p:.3f}"
        exp = int(np.floor(np.log10(p)))
        mant = p / (10 ** exp)
        return f"{mant:.2f}e{exp}"

    @classmethod
    def build_table1_pooled_comparison(
        cls, vessel: str, points_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Article Table I: pooled Baseline (B1+B2) vs Flicker for one vessel.

        Rows follow the eleven predefined endpoints (Holm-adjusted as a family)
        then exploratory ``G1`` (raw p only; ``p_holm`` left NaN). Columns mirror
        the published table and are written as CSV by the cohort runner.
        """
        family = list(cls.TABLE1_METRICS)
        exploratory = list(cls.TABLE1_EXPLORATORY_METRICS)

        raw_ps: list[float] = []
        family_rows: list[dict] = []
        for metric, label in family:
            if metric not in points_df.columns:
                base = np.asarray([], dtype=float)
                fl = np.asarray([], dtype=float)
                p, delta = float("nan"), float("nan")
            else:
                base = points_df.loc[
                    points_df["epoch"].isin(["B1", "B2"]), metric
                ].to_numpy(dtype=float)
                fl = points_df.loc[
                    points_df["epoch"] == "Flicker", metric
                ].to_numpy(dtype=float)
                p, delta = cls.pooled_test(points_df, metric)
            raw_ps.append(p)
            base_c, fl_c = cls.clean(base), cls.clean(fl)
            family_rows.append(
                {
                    "vessel": vessel,
                    "endpoint": label,
                    "metric": metric,
                    "is_exploratory": False,
                    "baseline_median": float(np.median(base_c)) if base_c.size else np.nan,
                    "baseline_iqr": (
                        float(np.percentile(base_c, 75) - np.percentile(base_c, 25))
                        if base_c.size
                        else np.nan
                    ),
                    "n_baseline": int(base_c.size),
                    "flicker_median": float(np.median(fl_c)) if fl_c.size else np.nan,
                    "flicker_iqr": (
                        float(np.percentile(fl_c, 75) - np.percentile(fl_c, 25))
                        if fl_c.size
                        else np.nan
                    ),
                    "n_flicker": int(fl_c.size),
                    "baseline_median_iqr": cls.format_median_iqr(base),
                    "flicker_median_iqr": cls.format_median_iqr(fl),
                    "cliffs_delta": delta,
                    "raw_p": p,
                }
            )

        holm_ps = cls.holm_adjust(raw_ps)
        for row, ph in zip(family_rows, holm_ps):
            row["p_holm"] = ph
            row["raw_p_fmt"] = cls.format_p_scientific(row["raw_p"])
            row["p_holm_fmt"] = cls.format_p_scientific(ph)

        rows = family_rows
        for metric, label in exploratory:
            if metric not in points_df.columns:
                base = np.asarray([], dtype=float)
                fl = np.asarray([], dtype=float)
                p, delta = float("nan"), float("nan")
            else:
                base = points_df.loc[
                    points_df["epoch"].isin(["B1", "B2"]), metric
                ].to_numpy(dtype=float)
                fl = points_df.loc[
                    points_df["epoch"] == "Flicker", metric
                ].to_numpy(dtype=float)
                p, delta = cls.pooled_test(points_df, metric)
            base_c, fl_c = cls.clean(base), cls.clean(fl)
            rows.append(
                {
                    "vessel": vessel,
                    "endpoint": label,
                    "metric": metric,
                    "is_exploratory": True,
                    "baseline_median": float(np.median(base_c)) if base_c.size else np.nan,
                    "baseline_iqr": (
                        float(np.percentile(base_c, 75) - np.percentile(base_c, 25))
                        if base_c.size
                        else np.nan
                    ),
                    "n_baseline": int(base_c.size),
                    "flicker_median": float(np.median(fl_c)) if fl_c.size else np.nan,
                    "flicker_iqr": (
                        float(np.percentile(fl_c, 75) - np.percentile(fl_c, 25))
                        if fl_c.size
                        else np.nan
                    ),
                    "n_flicker": int(fl_c.size),
                    "baseline_median_iqr": cls.format_median_iqr(base),
                    "flicker_median_iqr": cls.format_median_iqr(fl),
                    "cliffs_delta": delta,
                    "raw_p": p,
                    "p_holm": np.nan,
                    "raw_p_fmt": cls.format_p_scientific(p),
                    "p_holm_fmt": "--",
                }
            )

        column_order = [
            "vessel",
            "endpoint",
            "metric",
            "is_exploratory",
            "baseline_median_iqr",
            "flicker_median_iqr",
            "cliffs_delta",
            "raw_p",
            "p_holm",
            "baseline_median",
            "baseline_iqr",
            "n_baseline",
            "flicker_median",
            "flicker_iqr",
            "n_flicker",
            "raw_p_fmt",
            "p_holm_fmt",
        ]
        return pd.DataFrame(rows)[column_order]

    @staticmethod
    def pooled_test(
        df: pd.DataFrame,
        metric: str,
        branch_col: str = "epoch",
        flicker_val: str = "Flicker",
        baseline_vals: tuple[str, str] = ("B1", "B2"),
    ) -> tuple[float, float]:
        """Pooled-baseline-vs-flicker Mann-Whitney p-value and Cliff's delta for
        one metric column, reusing this class's own cliffs_delta rather than a
        separate implementation."""
        baseline = df.loc[df[branch_col].isin(baseline_vals), metric].to_numpy(dtype=float)
        flicker = df.loc[df[branch_col] == flicker_val, metric].to_numpy(dtype=float)
        baseline = baseline[np.isfinite(baseline)]
        flicker = flicker[np.isfinite(flicker)]
        if baseline.size < 2 or flicker.size < 2:
            return float("nan"), float("nan")
        p = float(mannwhitneyu(baseline, flicker, alternative="two-sided").pvalue)
        return p, LowRankWaveformStatistics.cliffs_delta(flicker, baseline)

    @staticmethod
    def format_p(p: float) -> str:
        if not np.isfinite(p):
            return "p=n/a"
        if p < 1e-3:
            return f"p={p:.1e}"
        return f"p={p:.3f}"

    @staticmethod
    def format_delta(delta: float) -> str:
        if not np.isfinite(delta):
            return r"$\delta$=n/a"
        return rf"$\delta$={delta:+.2f}"

class LowRankWaveformAcquisitionFigures:
    """Per-acquisition arterial figures under each file's relative folder
    (e.g. ``output/.../baseline1/``): article Figs. 2--4 plus beat-wise
    endpoint evolution within the acquisition."""

    FRMS_MAP_CANDIDATES = (
        "EyeFlow/Processing/FrequencyMaps/fRMS_avg/value",
        "Processing/FrequencyMaps/fRMS_avg/value",
    )
    GLOBAL_VELOCITY_CANDIDATES = {
        "artery": (
            "EyeFlow/Processing/Velocity/global/Artery/Raw/value",
            "Processing/Velocity/global/Artery/Raw/value",
        ),
        "vein": (
            "EyeFlow/Processing/Velocity/global/Vein/Raw/value",
            "Processing/Velocity/global/Vein/Raw/value",
        ),
    }
    PANEL_SIZE = 2.5

    @staticmethod
    def output_dir_for(
        output_dir: Path | str, h5_path: Path | str, input_root: Path | str
    ) -> Path:
        relative_parent = relative_hdf5_parent(h5_path, input_root)
        return h5_output_parent(output_dir, relative_parent)

    @staticmethod
    def _style_axes(ax, *, tick_size: int = 9, label_size: int = 10) -> None:
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.tick_params(axis="both", labelsize=tick_size)
        ax.xaxis.label.set_size(label_size)
        ax.yaxis.label.set_size(label_size)

    @classmethod
    def plot_all(
        cls,
        h5_path: Path | str,
        engine: "LowRankWaveformDecomposition",
        vessel_bundle: dict[str, dict | None],
        out_dir: Path,
        *,
        patient_id: str | None = None,
        signal: str = "raw",
    ) -> list[Path]:
        """Write Figs. 2--4 and the beat-endpoint evolution figure into ``out_dir``."""
        h5_path = Path(h5_path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = h5_path.stem
        written: list[Path] = []
        written.append(
            cls.plot_frequency_velocity(
                h5_path,
                out_dir / prefixed_filename(f"{stem}_fig2_frequency_velocity.png", patient_id),
            )
        )
        written.append(
            cls.plot_waveform_decomposition(
                h5_path,
                engine,
                out_dir / prefixed_filename(f"{stem}_fig3_waveform_decomposition.png", patient_id),
                signal=signal,
            )
        )
        written.append(
            cls.plot_energy_spectrum(
                vessel_bundle,
                out_dir / prefixed_filename(f"{stem}_fig4_energy_spectrum.png", patient_id),
            )
        )
        written.append(
            cls.plot_beat_endpoint_evolution(
                vessel_bundle,
                out_dir
                / prefixed_filename(f"{stem}_fig_beat_endpoint_evolution.png", patient_id),
            )
        )
        return written

    @classmethod
    def plot_frequency_velocity(cls, h5_path: Path | str, out_path: Path) -> Path:
        """Fig. 2: fRMS map + global raw arterial velocity."""
        h5_path = Path(h5_path)
        out_path = Path(out_path)
        with h5py.File(h5_path, "r") as h5:
            frms_path = find_first_existing_path(h5, list(cls.FRMS_MAP_CANDIDATES))
            frms = (
                np.asarray(h5[frms_path], dtype=float)
                if frms_path is not None
                else None
            )
            velocities: dict[str, np.ndarray | None] = {}
            for vessel in FIGURE_VESSELS:
                candidates = cls.GLOBAL_VELOCITY_CANDIDATES.get(vessel, ())
                path = find_first_existing_path(h5, list(candidates))
                velocities[vessel] = (
                    np.asarray(h5[path], dtype=float).reshape(-1)
                    if path is not None
                    else None
                )

        vessels = list(FIGURE_VESSELS)

        fig, axes = plt.subplots(
            len(vessels), 2, figsize=(8.0, 3.4 * len(vessels)), squeeze=False
        )
        for row_idx, vessel in enumerate(vessels):
            ax_map = axes[row_idx, 0]
            if frms is not None:
                ax_map.imshow(frms, cmap="gray", aspect="equal")
            ax_map.set_xticks([])
            ax_map.set_yticks([])
            ax_map.set_ylabel(vessel.capitalize(), fontsize=11)

            ax_wave = axes[row_idx, 1]
            wave = velocities.get(vessel)
            if wave is not None and wave.size:
                ax_wave.plot(np.arange(wave.size), wave, color="black", linewidth=1.0)
            ax_wave.axhline(0, color="#555555", linewidth=0.6, linestyle=":")
            ax_wave.set_xlabel("Time (ms)", fontsize=10)
            ax_wave.set_ylabel("Velocity (mm/s)", fontsize=10)
            cls._style_axes(ax_wave)

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    @staticmethod
    def _median_iqr_curve(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        med = np.nanmedian(matrix, axis=1)
        q25 = np.nanpercentile(matrix, 25, axis=1)
        q75 = np.nanpercentile(matrix, 75, axis=1)
        return med, q25, q75

    @classmethod
    def _waveform_summary_for_vessel(
        cls,
        h5_path: Path,
        engine: "LowRankWaveformDecomposition",
        vessel: str,
        signal: str,
    ) -> dict[str, np.ndarray] | None:
        source_key = f"{vessel}/{signal}"
        with h5py.File(h5_path, "r") as h5:
            schema = _resolve_vessel_sources(h5, veins_flag=False)
            if schema is None:
                return None
            candidates, t_path = schema
            dataset_path = candidates.get(source_key)
            if dataset_path is None or dataset_path not in h5:
                return None
            v_block = np.asarray(h5[dataset_path], dtype=float)
            T = np.asarray(h5[t_path], dtype=float)

        v_block = engine._ensure_segment_shape(v_block, T)
        T = engine._normalize_T(T)
        rep = engine._compute_representation(v_block, T)
        valid = rep["valid_column_mask"]
        mu = np.nanmean(v_block, axis=0, keepdims=True)
        x_full = v_block - mu
        valid_flat = valid.reshape(-1)
        if not np.any(valid_flat):
            return None

        v_cols = v_block.reshape(v_block.shape[0], -1)[:, valid_flat]
        x_cols = x_full.reshape(x_full.shape[0], -1)[:, valid_flat]
        mu_cols = mu.reshape(1, -1)[:, valid_flat]

        if rep.get("svd_available", False) and int(rep.get("n_modes_panel", 0)) >= 1:
            recon_cols = np.outer(rep["U_panel"][:, 0], rep["score_panel_flat"][0, :])
        else:
            recon_cols = np.full_like(x_cols, np.nan)
        if rep.get("svd_available", False) and int(rep.get("n_modes_panel", 0)) >= 2:
            recon2_cols = np.outer(rep["U_panel"][:, 1], rep["score_panel_flat"][1, :])
        else:
            recon2_cols = np.full_like(x_cols, np.nan)

        return {
            "t": np.linspace(0, 1, v_block.shape[0], endpoint=False),
            "v": v_cols,
            "mu": mu_cols,
            "x": x_cols,
            "a1u1": recon_cols,
            "a2u2": recon2_cols,
        }

    @classmethod
    def _row_ylim_first_two(cls, summary: dict[str, np.ndarray]) -> tuple[float, float]:
        bounds: list[float] = []
        med, q25, q75 = cls._median_iqr_curve(summary["v"])
        bounds += [float(np.nanmin(q25)), float(np.nanmax(q75))]
        mu_vals = summary["mu"].reshape(-1)
        med_mu = float(np.nanmedian(mu_vals))
        sd_mu = float(np.nanstd(mu_vals, ddof=1)) if mu_vals.size > 1 else 0.0
        bounds += [med_mu - sd_mu, med_mu + sd_mu]
        lo, hi = min(bounds), max(bounds)
        lo, hi = min(lo, 0.0), max(hi, 0.0)
        pad = 0.08 * (hi - lo if hi > lo else 1.0)
        return lo - pad, hi + pad

    @classmethod
    def _row_ylim_last_three(cls, summary: dict[str, np.ndarray]) -> tuple[float, float]:
        extents: list[float] = []
        for key in ("x", "a1u1", "a2u2"):
            _med, q25, q75 = cls._median_iqr_curve(summary[key])
            extents.append(
                max(abs(float(np.nanmin(q25))), abs(float(np.nanmax(q75))))
            )
        extent = max(extents) if extents else 1.0
        extent = extent if extent > 0 else 1.0
        extent *= 1.12
        return -extent, extent

    @classmethod
    def plot_waveform_decomposition(
        cls,
        h5_path: Path | str,
        engine: "LowRankWaveformDecomposition",
        out_path: Path,
        *,
        signal: str = "raw",
    ) -> Path:
        """Fig. 3: arterial v, mu, w, a1u1, a2u2."""
        h5_path = Path(h5_path)
        out_path = Path(out_path)
        rows: list[tuple[str, dict[str, np.ndarray] | None]] = []
        for vessel in FIGURE_VESSELS:
            summary = cls._waveform_summary_for_vessel(
                h5_path, engine, vessel, signal=signal
            )
            rows.append((vessel.capitalize(), summary))

        panel_defs = [
            ("v", "Beat-aligned\nvelocity"),
            ("mu", r"Baseline level" "\n" r"$\mu$"),
            ("x", r"Baseline-removed" "\n" r"$v-\mu$"),
            ("a1u1", r"Mode-1 recon." "\n" r"$a_1u_1$"),
            ("a2u2", r"Mode-2 recon." "\n" r"$a_2u_2$"),
        ]
        zero_cols = {"x", "a1u1", "a2u2"}
        n_rows = len(rows)
        n_cols = len(panel_defs)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(cls.PANEL_SIZE * n_cols + 1.0, cls.PANEL_SIZE * n_rows),
            sharex=True,
            sharey=False,
        )
        axes = np.atleast_2d(axes)
        for row_idx, (row_label, summary) in enumerate(rows):
            if summary is None:
                for col_idx in range(n_cols):
                    axes[row_idx, col_idx].set_visible(False)
                continue
            t = summary["t"]
            ylim_12 = cls._row_ylim_first_two(summary)
            ylim_345 = cls._row_ylim_last_three(summary)
            for col_idx, (key, title) in enumerate(panel_defs):
                ax = axes[row_idx, col_idx]
                if key == "mu":
                    mu_vals = summary[key].reshape(-1)
                    med = float(np.nanmedian(mu_vals))
                    sd = float(np.nanstd(mu_vals, ddof=1)) if mu_vals.size > 1 else 0.0
                    ax.axhline(med, color="black", linewidth=1.8)
                    ax.axhspan(med - sd, med + sd, color="black", alpha=0.12, linewidth=0)
                else:
                    med, q25, q75 = cls._median_iqr_curve(summary[key])
                    ax.plot(t, med, color="black", linewidth=1.8)
                    ax.fill_between(t, q25, q75, color="black", alpha=0.12, linewidth=0)
                ax.axhline(0, color="black", linewidth=1.0, linestyle=":")
                y_lo, y_hi = ylim_345 if key in zero_cols else ylim_12
                ax.set_ylim(y_lo, y_hi)
                if row_idx == 0:
                    ax.set_title(title, fontsize=11)
                if col_idx == 0:
                    ax.set_ylabel(f"{row_label} (mm/s)", fontsize=10)
                cls._style_axes(ax, tick_size=9, label_size=10)
                ax.set_box_aspect(1)
        fig.supxlabel("Fraction of cardiac cycle", fontsize=11)
        fig.tight_layout(w_pad=1.0, h_pad=1.0)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    @classmethod
    def plot_energy_spectrum(
        cls,
        vessel_bundle: dict[str, dict | None],
        out_path: Path,
    ) -> Path:
        """Fig. 4: arterial mode-wise SVD energy fractions."""
        out_path = Path(out_path)
        vessels = [
            v for v in FIGURE_VESSELS if vessel_bundle.get(v) is not None
        ] or list(FIGURE_VESSELS)
        fig, axes = plt.subplots(
            len(vessels), 1, figsize=(5.0, 2.8 * len(vessels)), squeeze=False
        )
        for row_idx, vessel in enumerate(vessels):
            ax = axes[row_idx, 0]
            data = vessel_bundle.get(vessel)
            energy = (
                np.asarray(data.get("energy_fraction", []), dtype=float)
                if data is not None
                else np.asarray([], dtype=float)
            )
            n_modes = int(min(12, energy.size))
            if n_modes > 0:
                modes = np.arange(1, n_modes + 1)
                ax.bar(modes, energy[:n_modes], color="black")
                ax.set_xticks(modes)
            ax.set_ylabel(f"{vessel.capitalize()}\nenergy fraction", fontsize=10)
            ax.set_xlabel("Mode", fontsize=10)
            if row_idx == 0:
                ax.set_title("SVD energy fraction", fontsize=12)
            cls._style_axes(ax)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    # Beat-indexed joint-SVD endpoints (solid) with optional per-beat SVD
    # overlays (dashed) where a matching *_pb series exists.
    BEAT_EVOLUTION_PANELS = (
        ("beat_period", "Beat period", None),
        ("mu", r"$\mu$", None),
        ("TPR", r"$R_0$", None),
        ("mpr", "MPR", None),
        ("A1", r"$A_1$", "A1_pb"),
        ("A2", r"$A_2$", "A2_pb"),
        ("R1", r"$R_1$", "R1_pb"),
        ("R2", r"$R_2$", "R2_pb"),
        ("rho1", r"$\rho_1$", None),
        ("rho2", r"$\rho_2$", None),
    )

    @classmethod
    def _beat_series(
        cls, vessel_data: dict, key: str
    ) -> np.ndarray:
        if key == "beat_period":
            return np.asarray(vessel_data.get("beat_period_b", []), dtype=float)
        beatwise = vessel_data.get("beatwise") or {}
        per_beat = vessel_data.get("per_beat_svd") or {}
        if key.endswith("_pb"):
            arr = per_beat.get(f"{key[:-3]}_b_pb", per_beat.get(key, []))
            return np.asarray(arr, dtype=float)
        if key in beatwise:
            return np.asarray(beatwise[key], dtype=float)
        # Prefer explicit *_b keys from the joint representation.
        if f"{key}_b" in beatwise:
            return np.asarray(beatwise[f"{key}_b"], dtype=float)
        return np.asarray([], dtype=float)

    @classmethod
    def plot_beat_endpoint_evolution(
        cls,
        vessel_bundle: dict[str, dict | None],
        out_path: Path,
    ) -> Path:
        """Per-acquisition: arterial endpoint trajectories vs beat index.

        One column per endpoint. Solid lines are joint-SVD beatwise
        summaries; dashed overlays (when present) are the independent
        per-beat SVD robustness variants.
        """
        out_path = Path(out_path)
        vessels = [
            v for v in FIGURE_VESSELS if vessel_bundle.get(v) is not None
        ] or list(FIGURE_VESSELS)

        panels = cls.BEAT_EVOLUTION_PANELS
        n_cols = len(panels)
        fig, axes = plt.subplots(
            len(vessels),
            n_cols,
            figsize=(max(1.6 * n_cols, 10.0), 2.6 * len(vessels)),
            squeeze=False,
            sharex=True,
        )
        for row_idx, vessel in enumerate(vessels):
            data = vessel_bundle.get(vessel)
            for col_idx, (key, title, pb_key) in enumerate(panels):
                ax = axes[row_idx, col_idx]
                if data is None:
                    ax.set_visible(False)
                    continue
                y = cls._beat_series(data, key)
                if y.size == 0 and key in ("mu", "TPR", "mpr", "A1", "A2", "R1", "R2", "rho1", "rho2"):
                    y = cls._beat_series(data, f"{key}_b")
                beats = np.arange(y.size)
                if y.size:
                    ax.plot(beats, y, color="black", marker="o", markersize=3.5, linewidth=1.4)
                if pb_key is not None:
                    y_pb = cls._beat_series(data, pb_key)
                    if y_pb.size:
                        ax.plot(
                            np.arange(y_pb.size),
                            y_pb,
                            color="#555555",
                            linestyle="--",
                            marker="s",
                            markersize=3.0,
                            linewidth=1.2,
                            label="per-beat SVD" if col_idx == 0 else None,
                        )
                ax.set_xlabel("Beat", fontsize=9)
                if row_idx == 0:
                    ax.set_title(title, fontsize=11)
                if col_idx == 0:
                    ax.set_ylabel(vessel.capitalize(), fontsize=11)
                    if pb_key is not None and data is not None:
                        handles, labels = ax.get_legend_handles_labels()
                        if handles:
                            ax.legend(frameon=False, fontsize=7, loc="best")
                cls._style_axes(ax, tick_size=8, label_size=9)
        fig.suptitle("Per-beat endpoint evolution", fontsize=12)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path


class LowRankWaveformCohortFigures:
    """Article Figs. 5--7 (arterial), written once per cohort in the cohort
    base directory."""

    PANEL_SIZE = 2.5
    FLICKER_SHADE = "#add8e6"
    _RNG = np.random.default_rng(0)

    FIG5_PANELS = (
        ("beat_period", "Beat period"),
        ("mu", r"Baseline level $\mu$"),
        ("TPR", r"$R_0$"),
        ("mpr", "MPR"),
    )
    FIG6_PANELS = (
        ("A1", r"$A_1$"),
        ("A2", r"$A_2$"),
        ("TPR", r"$R_0$"),
        ("R1", r"$R_1$"),
        ("R2", r"$R_2$"),
    )
    FIG7_PANELS = (
        ("rho1", r"$\rho_1$"),
        ("rho2", r"$\rho_2$"),
        ("effective_rank", r"$R_{\mathrm{eff}}$"),
        ("participation_ratio", "PR"),
        ("alpha", r"$\alpha$"),
        ("G1", r"$G_1$"),
    )

    @staticmethod
    def output_dir_for(output_dir: Path | str) -> Path:
        return Path(output_dir)

    @staticmethod
    def _style_axes(ax, *, tick_size: int = 9) -> None:
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.tick_params(axis="both", labelsize=tick_size)

    @classmethod
    def plot_all(
        cls,
        points_by_vessel: dict[str, pd.DataFrame],
        out_dir: Path,
        group_order: list[str],
        *,
        patient_id: str | None = None,
    ) -> list[Path]:
        """Write Figs. 5, 6, and 7 into ``out_dir``."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written = [
            cls.plot_nonsvd_endpoints(
                points_by_vessel,
                out_dir / prefixed_filename("fig5_nonsvd_endpoints.png", patient_id),
                group_order=group_order,
            ),
            cls.plot_lowrank_endpoints(
                points_by_vessel,
                out_dir / prefixed_filename("fig6_lowrank_endpoints.png", patient_id),
                group_order=group_order,
            ),
            cls.plot_residual_spectrum_endpoints(
                points_by_vessel,
                out_dir
                / prefixed_filename("fig7_residual_spectrum_endpoints.png", patient_id),
                group_order=group_order,
            ),
        ]
        return written

    @classmethod
    def _draw_epoch_panel(
        cls,
        ax,
        df: pd.DataFrame,
        metric: str,
        group_order: list[str],
    ) -> None:
        labels = [group_display_label(g) for g in group_order]
        positions = {label: idx for idx, label in enumerate(labels)}
        if is_flicker_triad(group_order) and len(group_order) >= 3:
            ax.axvspan(0.5, 1.5, color=cls.FLICKER_SHADE, zorder=0)
            ax.axvline(0.5, color="black", linestyle=":", linewidth=1.5, zorder=1)
            ax.axvline(1.5, color="black", linestyle=":", linewidth=1.5, zorder=1)

        for label in labels:
            vals = (
                df.loc[df["epoch"] == label, metric].dropna().to_numpy(dtype=float)
                if metric in df.columns
                else np.asarray([], dtype=float)
            )
            if vals.size == 0:
                continue
            x0 = positions[label]
            jitter = (cls._RNG.random(vals.size) - 0.5) * 0.16
            med = float(np.nanmedian(vals))
            sd = float(np.nanstd(vals, ddof=1)) if vals.size > 1 else 0.0
            ax.errorbar(
                [x0],
                [med],
                yerr=[sd],
                fmt="o",
                markersize=7,
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=1.4,
                ecolor="black",
                elinewidth=1.5,
                capsize=4,
                zorder=4,
            )
            ax.scatter(
                x0 + jitter, vals, s=20, color="black", edgecolors="none", zorder=5
            )

        ax.set_xticks(list(positions.values()))
        ax.set_xticklabels(list(positions.keys()), fontsize=9)
        ax.set_xlim(-0.5, max(len(labels) - 0.5, 0.5))
        cls._style_axes(ax)

    @classmethod
    def _plot_paired_endpoint_grid(
        cls,
        points_by_vessel: dict[str, pd.DataFrame],
        panels: tuple[tuple[str, str], ...],
        out_path: Path,
        group_order: list[str],
    ) -> Path:
        out_path = Path(out_path)
        vessels = [
            v
            for v in FIGURE_VESSELS
            if v in points_by_vessel and not points_by_vessel[v].empty
        ]
        if not vessels:
            vessels = list(FIGURE_VESSELS)
            points_by_vessel = {"artery": pd.DataFrame()}

        n_cols = len(panels)
        fig, axes = plt.subplots(
            len(vessels),
            n_cols,
            figsize=(cls.PANEL_SIZE * n_cols, cls.PANEL_SIZE * len(vessels)),
            squeeze=False,
        )
        for row_idx, vessel in enumerate(vessels):
            df = points_by_vessel.get(vessel, pd.DataFrame())
            for col_idx, (metric, title) in enumerate(panels):
                ax = axes[row_idx, col_idx]
                cls._draw_epoch_panel(ax, df, metric, group_order)
                ax.set_box_aspect(1)
                if row_idx == 0:
                    ax.set_title(title, fontsize=11)
                if col_idx == 0:
                    ax.set_ylabel(vessel.capitalize(), fontsize=11)
        fig.tight_layout(w_pad=1.0, h_pad=1.0)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    @classmethod
    def plot_nonsvd_endpoints(
        cls,
        points_by_vessel: dict[str, pd.DataFrame],
        out_path: Path,
        group_order: list[str] | None = None,
    ) -> Path:
        """Fig. 5: beat period, mu, R0, MPR."""
        if group_order is None:
            labels = pd.concat(points_by_vessel.values(), ignore_index=True)["epoch"]
            group_order = ordered_groups(
                EPOCH_SHORT_TO_KEY.get(label, label) for label in labels.unique()
            )
        return cls._plot_paired_endpoint_grid(
            points_by_vessel, cls.FIG5_PANELS, out_path, group_order
        )

    @classmethod
    def plot_lowrank_endpoints(
        cls,
        points_by_vessel: dict[str, pd.DataFrame],
        out_path: Path,
        group_order: list[str] | None = None,
    ) -> Path:
        """Fig. 6: A1, A2, R0, R1, R2."""
        if group_order is None:
            labels = pd.concat(points_by_vessel.values(), ignore_index=True)["epoch"]
            group_order = ordered_groups(
                EPOCH_SHORT_TO_KEY.get(label, label) for label in labels.unique()
            )
        return cls._plot_paired_endpoint_grid(
            points_by_vessel, cls.FIG6_PANELS, out_path, group_order
        )

    @classmethod
    def plot_residual_spectrum_endpoints(
        cls,
        points_by_vessel: dict[str, pd.DataFrame],
        out_path: Path,
        group_order: list[str] | None = None,
    ) -> Path:
        """Fig. 7: rho1, rho2, Reff, PR, alpha, G1."""
        if group_order is None:
            labels = pd.concat(points_by_vessel.values(), ignore_index=True)["epoch"]
            group_order = ordered_groups(
                EPOCH_SHORT_TO_KEY.get(label, label) for label in labels.unique()
            )
        return cls._plot_paired_endpoint_grid(
            points_by_vessel, cls.FIG7_PANELS, out_path, group_order
        )


class LowRankWaveformConfounds:
    """Sec. V.B confound-control grid (analysis-choice sweep + verdicts) for
    the classic flicker triad, plus the per-cohort collection pipeline
    (collect_acquisitions, run_confound_statistics). Acquisition figures
    always run; cohort figures (Figs. 5--7) run for any 2+ folder split; confound
    tables require baseline1/flicker/baseline2."""

    # Metrics with both a joint-SVD and a per-beat-SVD representation --
    # the ones build_grid sweeps over the svd_method axis for. TPR/mpr/
    # alpha/G1/mpr_prime have no such axis and are handled separately.
    METRICS_SVD = ("A1", "A2", "rho1", "rho2")

    def __init__(self, engine: LowRankWaveformDecomposition | None = None) -> None:
        if engine is None:
            # Confound statistics report both compartments; opt into the
            # pipeline's optional vein processing (artery-only is the
            # pipeline default).
            engine = LowRankWaveformDecomposition()
            engine.veins_flag = True
        self._lr = engine

    @staticmethod
    def residualize_against_beat_period(
        epoch_values: dict[str, np.ndarray],
        epoch_beat_periods: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Sec. V.B.4 beat-period control. Pools the metric and the mean beat
        period across all acquisitions/epochs, fits ``d = alpha + beta*T`` by OLS
        (one global line), and returns each value replaced by its residual from
        that line. If fewer than 3 finite (value, period) pairs exist the fit is
        skipped and the inputs are returned unchanged (as float arrays)."""
        all_values = np.concatenate(
            [np.asarray(v, dtype=float) for v in epoch_values.values()]
        )
        all_periods = np.concatenate(
            [np.asarray(v, dtype=float) for v in epoch_beat_periods.values()]
        )
        mask = np.isfinite(all_values) & np.isfinite(all_periods)
        if np.sum(mask) < 3:
            return {k: np.asarray(v, dtype=float) for k, v in epoch_values.items()}

        slope, intercept = np.polyfit(all_periods[mask], all_values[mask], 1)

        out = {}
        for epoch, values in epoch_values.items():
            values = np.asarray(values, dtype=float)
            periods = np.asarray(epoch_beat_periods[epoch], dtype=float)
            out[epoch] = values - (slope * periods + intercept)
        return out

    def acq_dots(
        self,
        acqs: list[dict],
        metric: str,
        svd_method: str | None,
        stat: str,
    ) -> np.ndarray:
        """One acquisition-level value ("dot") per acquisition for the given metric,
        under a specific analysis choice. ``stat`` is the beat-aggregation
        ("median"/"mean") and ``svd_method`` selects the joint vs per-beat SVD
        representation; both are ignored for metrics that have no such axis (alpha,
        mpr_prime are acquisition scalars; TPR, mpr have no SVD axis). See the
        per-branch comments for how each metric maps to the engine output."""
        out = []
        for a in acqs:
            if metric == "TPR":
                out.append(self._lr.aggregate_beatwise(a["beatwise"]["TPR_b"], stat))
            elif metric == "mpr":
                # Not SVD-derived (ratio of mu to pulsatile RMS, Eq. 18) -- same
                # beat-aggregation treatment as TPR, no joint/per-beat SVD axis.
                out.append(self._lr.aggregate_beatwise(a["beatwise"]["mpr_b"], stat))
            elif metric == "alpha":
                # Single acquisition-level scalar from the joint-SVD singular
                # spectrum (Eq. 19), like effective_rank/participation_ratio --
                # no per-beat array, so svd_method/stat are not applicable axes.
                out.append(float(a["acq"].get("alpha", np.nan)))
            elif metric == "G1":
                # Exploratory dominant-mode gap (Eq. 23): 1 - lambda2/lambda1
                # from the joint-SVD singular values -- same "no beat-aggregation
                # axis" treatment as alpha.
                out.append(float(a["acq"].get("G1", np.nan)))
            elif metric == "mpr_prime":
                # MPR' (Eq. 17): ratio of two separately-aggregated acquisition
                # scalars (median|mu| / R0), not an average of per-beat ratios --
                # same "no beat-aggregation axis" treatment as alpha.
                out.append(float(a["acq"].get("mpr_prime", np.nan)))
            elif metric in ("A1", "A2"):
                if svd_method == "joint":
                    arr = a["beatwise"][f"{metric}_b"]
                else:
                    arr = a["per_beat_svd"][f"{metric}_b_pb"]
                out.append(self._lr.aggregate_beatwise(arr, stat))
            elif metric in ("rho1", "rho2"):
                m = metric[-1]
                if svd_method == "joint":
                    R_b = a["beatwise"][f"R{m}_b"]
                else:
                    R_b = a["per_beat_svd"][f"R{m}_b_pb"]
                out.append(self._lr.aggregate_rho(R_b, a["beatwise"]["TPR_b"], stat))
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
        """``acq_dots`` applied per epoch, returning {epoch: dots} keyed by
        EPOCH_ORDER (the shape LowRankWaveformStatistics.epoch_group_tests /
        residualize_against_beat_period expect)."""
        return {
            epoch: self.acq_dots(acqs_by_epoch[epoch], metric, svd_method, stat)
            for epoch in EPOCH_ORDER
        }

    @staticmethod
    def epoch_beat_periods(
        acqs_by_epoch: dict[str, list[dict]],
    ) -> dict[str, np.ndarray]:
        """{epoch: mean beat period per acquisition}, the period covariate used to
        residualize each metric in the beat-period-controlled grid rows."""
        return {
            epoch: np.array(
                [a["beat_period_mean"] for a in acqs_by_epoch[epoch]], dtype=float
            )
            for epoch in EPOCH_ORDER
        }

    def build_grid(
        self,
        vessel: str,
        acqs_by_epoch: dict[str, list[dict]],
    ) -> tuple[list[dict], list[dict]]:
        """Sec. V.B robustness sweep for one vessel. Runs LowRankWaveformStatistics.epoch_group_tests
        over every applicable combination of analysis choices -- metric x SVD method
        (joint/per_beat, where relevant) x beat aggregation (median/mean) x beat
        period control (native/residualized) -- producing one ``grid_row`` each.

        Returns (grid_rows, verdict_rows): the per-metric ``verdict_rows`` summarise
        whether the flicker effect survived every combination (all pooled p < 0.05)
        with a consistent Cliff's-delta sign across combinations."""
        periods = self.epoch_beat_periods(acqs_by_epoch)
        grid_rows: list[dict] = []
        verdict_source: dict[str, list[dict]] = {}

        def run_combo(
            metric: str, svd_method: str | None, stat: str, regressed: bool
        ) -> dict:
            """Run epoch_group_tests for one grid cell, residualizing against beat
            period first when ``regressed`` is set."""
            values = self.epoch_dots(acqs_by_epoch, metric, svd_method, stat)
            if regressed:
                values = self.residualize_against_beat_period(values, periods)
            return LowRankWaveformStatistics.epoch_group_tests(values)

        def add_row(
            metric: str, svd_method: str | None, stat: str, regressed: bool
        ) -> None:
            """Compute one grid cell and append its flattened row to grid_rows (and
            to verdict_source, grouped by metric for the verdict summary)."""
            tests = run_combo(metric, svd_method, stat, regressed)
            row = {
                "vessel": vessel,
                "metric": metric,
                "svd_method": svd_method if svd_method is not None else "n/a",
                "beat_aggregation": stat,
                "beat_period_control": "residualized" if regressed else "native",
                "kruskal_wallis_p": tests["kruskal_wallis_p"],
                "pooled_baseline_vs_flicker_p": tests["pooled_baseline_vs_flicker_p"],
                "cliffs_delta_flicker_vs_pooled_baseline": tests[
                    "pooled_baseline_vs_flicker_delta"
                ],
                "n_B1": tests["n"]["baseline1"],
                "n_Flicker": tests["n"]["flicker"],
                "n_B2": tests["n"]["baseline2"],
            }
            grid_rows.append(row)
            verdict_source.setdefault(metric, []).append(row)

        for metric in self.METRICS_SVD:
            for svd_method in ("joint", "per_beat"):
                for stat in ("median", "mean"):
                    for regressed in (False, True):
                        add_row(metric, svd_method, stat, regressed)

        for stat in ("median", "mean"):
            for regressed in (False, True):
                add_row("TPR", None, stat, regressed)
                add_row("mpr", None, stat, regressed)

        for regressed in (False, True):
            add_row("alpha", None, "n/a", regressed)
            add_row("G1", None, "n/a", regressed)
            add_row("mpr_prime", None, "n/a", regressed)

        verdict_rows = []
        for metric, rows in verdict_source.items():
            ps = [r["pooled_baseline_vs_flicker_p"] for r in rows]
            deltas = [r["cliffs_delta_flicker_vs_pooled_baseline"] for r in rows]
            n_significant = int(sum(1 for p in ps if np.isfinite(p) and p < 0.05))
            signed = [d for d in deltas if np.isfinite(d)]
            consistent_direction = bool(signed) and (
                all(d > 0 for d in signed) or all(d < 0 for d in signed)
            )
            all_significant = n_significant == len(rows) and len(rows) > 0
            finite_ps = [p for p in ps if np.isfinite(p)]
            verdict_rows.append(
                {
                    "vessel": vessel,
                    "metric": metric,
                    "n_combinations": len(rows),
                    "n_significant_p_lt_0.05": n_significant,
                    "consistent_direction": consistent_direction,
                    "retained_all_combinations_significant_and_consistent": bool(
                        all_significant and consistent_direction
                    ),
                    "max_pooled_p": max(finite_ps) if finite_ps else np.nan,
                    "min_pooled_p": min(finite_ps) if finite_ps else np.nan,
                }
            )

        return grid_rows, verdict_rows

    @staticmethod
    def build_beat_rows(
        vessel: str,
        h5_path: Path,
        sequence: int,
        epoch_short: str,
        vessel_data: dict,
    ) -> list[dict]:
        """Expand one acquisition into one long-format row per beat, keeping the
        per-beat endpoint arrays the engine computes in both representations: the
        joint-SVD ``beatwise`` values and the per-beat-SVD ``*_pb`` variants.

        ``alpha`` and ``mpr_prime`` are acquisition-level-only by construction
        (whole singular spectrum / ratio of aggregates) and have no per-beat value,
        so they appear only in the points table, not here. Missing/short entries
        are filled with NaN."""
        beatwise = vessel_data["beatwise"]
        per_beat_svd = vessel_data["per_beat_svd"]
        n_beats = len(beatwise["TPR_b"])
        vfb = vessel_data["valid_fraction_per_beat"]
        period_b = vessel_data["beat_period_b"]

        def at(arr, b):
            arr = np.asarray(arr, dtype=float)
            return float(arr[b]) if b < arr.size and np.isfinite(arr[b]) else float("nan")

        rows = []
        for b in range(n_beats):
            rows.append(
                {
                    "vessel": vessel,
                    "acquisition": sequence,
                    "file": h5_path.name,
                    "epoch": epoch_short,
                    "beat_index": b,
                    "beat_period": at(period_b, b),
                    "valid_fraction": at(vfb, b),
                    "mu": at(beatwise["mu_b"], b),
                    "TPR": at(beatwise["TPR_b"], b),
                    "mpr": at(beatwise["mpr_b"], b),
                    "A1": at(beatwise["A1_b"], b),
                    "R1": at(beatwise["R1_b"], b),
                    "rho1": at(beatwise["rho1_b"], b),
                    "A2": at(beatwise["A2_b"], b),
                    "R2": at(beatwise["R2_b"], b),
                    "rho2": at(beatwise["rho2_b"], b),
                    "A1_pb": at(per_beat_svd["A1_b_pb"], b),
                    "R1_pb": at(per_beat_svd["R1_b_pb"], b),
                    "A2_pb": at(per_beat_svd["A2_b_pb"], b),
                    "R2_pb": at(per_beat_svd["R2_b_pb"], b),
                }
            )
        return rows

    @staticmethod
    def build_points_row(
        vessel: str,
        h5_path: Path,
        sequence: int,
        epoch_short: str,
        vessel_data: dict,
    ) -> dict:
        """One acquisition-level row (a single dot per endpoint) for the points
        table: the acquisition-aggregated endpoints plus their beat-to-beat SDs,
        mean beat period, and column-validity bookkeeping."""
        acq = vessel_data["acq"]
        vfb = vessel_data["valid_fraction_per_beat"]

        def scalar(key: str) -> float:
            return float(acq.get(key, np.nan))

        return {
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
            "TPR": scalar("TPR"),
            "TPR_sd": scalar("sigma_TPR_beat"),
            "mpr": scalar("mpr"),
            "mpr_sd": scalar("sigma_mpr_beat"),
            "mpr_prime": scalar("mpr_prime"),
            "alpha": scalar("alpha"),
            "G1": scalar("G1"),
            "effective_rank": scalar("effective_rank"),
            "participation_ratio": scalar("participation_ratio"),
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

    def collect_acquisitions(
        self,
        records: list[tuple[str | None, Path]],
        input_root: Path,
        output_dir: Path | None = None,
        patient_id: str | None = None,
        group_order: list[str] | None = None,
    ) -> dict[str, tuple[dict[str, list[dict]], list[dict], list[dict]]]:
        """Runs the low-rank engine over classified ``(group, h5_path)``
        records (see classify_cohort). Every acquisition gets endpoints +
        acquisition figures (article Figs. 2--4) when ``output_dir`` is set --
        including flat (ungrouped) files. Named groups populate
        ``acqs_by_group`` for optional cohort figures/stats (Figs. 5--7).
        Returns
        ``{"artery": (acqs_by_group, points_rows, beat_rows), "vein": (...)}``.
        """
        if group_order is None:
            group_order = ordered_groups(group for group, _ in records)
        # Always allocate known flicker keys too so build_grid can still
        # index them when the triad is present (empty lists otherwise).
        group_keys = list(dict.fromkeys([*group_order, *EPOCH_ORDER]))

        per_vessel: dict[str, dict] = {
            vessel: {
                "acqs_by_group": {g: [] for g in group_keys},
                "points_rows": [],
                "beat_rows": [],
                "sequence_counters": {g: 0 for g in group_keys},
                "flat_sequence": 0,
            }
            for vessel in VESSEL_TYPES
        }

        for group, h5_path in records:
            data = self._lr.compute_acquisition_endpoints(h5_path)
            if data is None:
                continue
            acquisition_out_dir = None
            if output_dir is not None:
                # Persist this acquisition's joint-SVD + per-beat endpoints
                # and article Figs. 2--4 plus beat-endpoint evolution under the
                # acquisition's own relative folder (e.g. baseline1/, ctrl/,
                # or "." for flat).
                acquisition_out_dir = LowRankWaveformAcquisitionFigures.output_dir_for(
                    output_dir, h5_path, input_root
                )
                out_name = prefixed_filename(
                    f"{h5_path.stem}_pipelines_result.h5", patient_id
                )
                self._lr.write_acquisition_h5(h5_path, acquisition_out_dir / out_name)
                LowRankWaveformAcquisitionFigures.plot_all(
                    h5_path,
                    self._lr,
                    data,
                    acquisition_out_dir,
                    patient_id=patient_id,
                    signal="raw",
                )

            for vessel in VESSEL_TYPES:
                vessel_data = data.get(vessel)
                if vessel_data is None:
                    continue
                state = per_vessel[vessel]
                if group is None:
                    seq = state["flat_sequence"]
                    state["flat_sequence"] += 1
                    group_label = "ungrouped"
                else:
                    if group not in state["acqs_by_group"]:
                        state["acqs_by_group"][group] = []
                        state["sequence_counters"][group] = 0
                    state["acqs_by_group"][group].append(vessel_data)
                    seq = state["sequence_counters"][group]
                    state["sequence_counters"][group] += 1
                    group_label = group_display_label(group)

                state["points_rows"].append(
                    self.build_points_row(vessel, h5_path, seq, group_label, vessel_data)
                )
                state["beat_rows"].extend(
                    self.build_beat_rows(vessel, h5_path, seq, group_label, vessel_data)
                )

        result: dict[str, tuple[dict[str, list[dict]], list[dict], list[dict]]] = {}
        for vessel, state in per_vessel.items():
            points_rows = state["points_rows"]
            beat_rows = state["beat_rows"]
            points_rows.sort(
                key=lambda r: (_row_group_sort_key(r["epoch"]), r["acquisition"])
            )
            beat_rows.sort(
                key=lambda r: (
                    _row_group_sort_key(r["epoch"]),
                    r["acquisition"],
                    r["beat_index"],
                )
            )
            result[vessel] = (state["acqs_by_group"], points_rows, beat_rows)
        return result

    def run_confound_statistics(
        self,
        input_h5_paths: Iterable[Path],
        input_root: Path,
        output_dir: Path,
    ) -> tuple[str, list[Path]]:
        """Full cohort regeneration:

        * every acquisition always gets endpoints ``.h5`` + article Figs. 2--4
          under its own relative folder;
        * when the dataset has a multi-folder split (any 2+ named groups --
          bl1/f/bl2, ctrl/path, ...), article Figs. 5--7 are written in the base
          directory;
        * Sec. V.A/V.B endpoint tables + confound grids run only when the
          classic flicker triad (baseline1/flicker/baseline2) is present.

        Returns ``(summary, generated_paths)``.
        """
        input_root = Path(input_root)
        output_dir = Path(output_dir)
        patient_id = extract_patient_id(input_root)
        root_dir = output_dir / "lowrank_confound_statistics"
        for sub in ("points", "beats", "tables", "confound_control"):
            (root_dir / sub).mkdir(parents=True, exist_ok=True)

        records, group_order = classify_cohort(input_h5_paths, input_root)
        if not records:
            raise ValueError(
                f"No acquisition HDF5 files found under {input_root}."
            )

        has_cohort_split = len(group_order) >= 2
        flicker_protocol = is_flicker_triad(group_order)
        figures_dir = LowRankWaveformCohortFigures.output_dir_for(output_dir)
        if has_cohort_split:
            figures_dir.mkdir(parents=True, exist_ok=True)

        generated_paths: list[Path] = []

        def _write_csv(df: pd.DataFrame, subdir: str, filename: str) -> Path:
            path = root_dir / subdir / prefixed_filename(filename, patient_id)
            df.to_csv(path, index=False)
            generated_paths.append(path)
            return path

        per_vessel = self.collect_acquisitions(
            records,
            input_root,
            output_dir=output_dir,
            patient_id=patient_id,
            group_order=group_order,
        )

        vessel_summaries: list[str] = []
        all_points: list[pd.DataFrame] = []
        points_by_vessel: dict[str, pd.DataFrame] = {}
        for vessel, (acqs_by_group, points_rows, beat_rows) in per_vessel.items():
            n_acq = len(points_rows)
            if n_acq == 0:
                continue
            group_counts = ", ".join(
                f"{group_display_label(g)}={len(acqs_by_group.get(g, []))}"
                for g in group_order
            ) or f"ungrouped={n_acq}"
            vessel_summaries.append(f"{vessel}={n_acq} ({group_counts})")

            points_df = pd.DataFrame(points_rows)
            points_by_vessel[vessel] = points_df
            all_points.append(points_df)
            _write_csv(points_df, "points", f"{vessel}_points.csv")
            _write_csv(pd.DataFrame(beat_rows), "beats", f"{vessel}_beats.csv")

            if flicker_protocol:
                _write_csv(
                    LowRankWaveformStatistics.build_endpoint_table(vessel, points_df),
                    "tables",
                    f"{vessel}_endpoint_table.csv",
                )
                _write_csv(
                    LowRankWaveformStatistics.build_table1_pooled_comparison(
                        vessel, points_df
                    ),
                    "tables",
                    f"{vessel}_table1_pooled_baseline_vs_flicker.csv",
                )
                grid_rows, verdict_rows = self.build_grid(vessel, acqs_by_group)
                _write_csv(
                    pd.DataFrame(grid_rows),
                    "confound_control",
                    f"{vessel}_confound_grid.csv",
                )
                _write_csv(
                    pd.DataFrame(verdict_rows),
                    "confound_control",
                    f"{vessel}_confound_verdict.csv",
                )

        if has_cohort_split and points_by_vessel:
            generated_paths.extend(
                LowRankWaveformCohortFigures.plot_all(
                    points_by_vessel,
                    figures_dir,
                    group_order,
                    patient_id=patient_id,
                )
            )

        if not all_points:
            raise ValueError(
                "No vessel had any valid acquisitions; nothing to write."
            )

        split_note = (
            f"cohort split={group_order}"
            if has_cohort_split
            else "no cohort split (acquisition figures only)"
        )
        combined_points = pd.concat(all_points, ignore_index=True)
        summary = (
            f"Low-rank waveform run: {len(combined_points)} "
            f"acquisition-vessel row(s) ({', '.join(vessel_summaries)}); {split_note}."
        )
        return summary, generated_paths

    def run(self, input_h5_paths, input_root, output_dir):
        """Lower-level cohort entry when paths are already resolved.
        Prefer the module-level :func:`run` for ZIP/folder inputs."""
        if not input_h5_paths:
            raise ValueError(
                "No input acquisition files are available for postprocessing."
            )

        return self.run_confound_statistics(
            input_h5_paths=input_h5_paths,
            input_root=input_root,
            output_dir=output_dir,
        )


def run(
    input_path: Path | str,
    output_dir: Path | str,
    *,
    veins: bool = True,
) -> tuple[str, list[Path]]:
    """Full start-to-end regeneration for one cohort dataset.

    Accepts either an extracted cohort folder or a ZIP (e.g.
    ``260803_Flicker_EF.zip`` whose root contains group subfolders such as
    ``baseline1/flicker/baseline2`` or ``ctrl/path``), then:

    1. classifies acquisitions by top-level split folder (any names),
    2. computes joint + per-beat SVD endpoints (artery always; vein when
       ``veins`` is True),
    3. always writes per-acquisition ``.h5`` + article Figs. 2--4 and the
       beat-endpoint evolution figure under each acquisition's folder via
       :class:`LowRankWaveformAcquisitionFigures`,
    4. when 2+ named groups are present, writes article Figs. 5--7 in
       ``output_dir`` via :class:`LowRankWaveformCohortFigures`,
    5. when the classic flicker triad is present, also writes Sec. V
       endpoint/confound CSVs and article Table I
       (``*_table1_pooled_baseline_vs_flicker.csv``) under
       ``output_dir/lowrank_confound_statistics/``.

    Returns ``(summary, generated_paths)``.
    """
    input_path = Path(input_path).expanduser()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    confounds = LowRankWaveformConfounds()
    confounds._lr.veins_flag = bool(veins)

    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        with extracted_zip_tree(input_path) as extracted_root:
            cohort_root = resolve_cohort_root(extracted_root)
            h5_paths = find_hdf5_inputs(cohort_root)
            return confounds.run(h5_paths, cohort_root, output_dir)

    if input_path.is_dir():
        cohort_root = resolve_cohort_root(input_path)
        h5_paths = find_hdf5_inputs(cohort_root)
        return confounds.run(h5_paths, cohort_root, output_dir)

    raise ValueError(
        f"Input must be a cohort folder or .zip archive, got: {input_path}"
    )
