"""Read EyeFlow-packed low-rank endpoints for batch statistics and figures."""

from __future__ import annotations

import threading
import warnings
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from input_output.hdf5_schema import find_pipeline_group

ALL_VESSELS = ("artery", "vein")

PIPELINE_NAME = "lowrank_waveform_decomposition"

COHORT_SIGNAL = "raw"

SIGNALS = ("bandlimited", "raw")

SVD_METHODS = ("joint", "per_beat")

DEFAULT_SVD_METHOD = "joint"

EYEFLOW_LOWRANK_GROUP_CANDIDATES = (
    "Processing/Metrics/lowrank_waveform_decomposition",
    "Metrics/lowrank_waveform_decomposition",
)

SPECTRUM_N_MODES = 12

ENDPOINT_METRICS = (
    "A1",
    "A2",
    "R0",
    "R1",
    "R2",
    "rho1",
    "rho2",
    "MPR",
    "Reff",
    "PR",
)

EYEFLOW_MISC_SPECTRUM_JOINT = "svd_spectrum_joint"

EYEFLOW_MISC_SPECTRUM_PER_BEAT = "svd_spectrum_joint_per_beat"


def enabled_vessels(veins_flag: bool) -> tuple[str, ...]:
    """Return ``("artery",)`` or ``("artery", "vein")`` from the Veins flag."""
    return ("artery", "vein") if veins_flag else ("artery",)


def normalize_svd_method(value: str | None) -> str:
    """Return ``joint`` or ``per_beat`` from a pipeline/pipeline flag."""
    text = str(DEFAULT_SVD_METHOD if value is None else value).strip().lower()
    text = text.replace("-", "_").replace(" ", "_")
    aliases = {
        "joint": "joint",
        "joint_svd": "joint",
        "per_beat": "per_beat",
        "per_beat_svd": "per_beat",
        "perbeat": "per_beat",
        "perbeat_svd": "per_beat",
    }
    if text not in aliases:
        raise ValueError(
            f"Unknown svd_method {value!r}. Expected one of {SVD_METHODS}."
        )
    return aliases[text]


def aggregate_beatwise(values_per_beat: np.ndarray, stat: str) -> float:
    """Collapse a per-beat array to one acquisition scalar (mean or median)."""
    x = np.asarray(values_per_beat, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.mean(x)) if stat == "mean" else float(np.median(x))


def aggregate_rho(R_b: np.ndarray, R0_b: np.ndarray, stat: str) -> float:
    """Residual ratio R/R0 from separately aggregated residual and pulsatile RMS."""
    r = aggregate_beatwise(R_b, stat)
    t = aggregate_beatwise(R0_b, stat)
    if not np.isfinite(r) or not np.isfinite(t) or abs(t) <= 1e-12:
        return float("nan")
    return float(r / (t + 1e-12))


_FIGURE_LOCK = threading.RLock()


def safe_figure(name: str, plotter, /, **kwargs) -> Path | None:
    """Draw one figure under the global lock, turning failure into a warning.

    Two hazards are handled here. Matplotlib is not thread-safe: pyplot
    keeps a global figure registry and mathtext is a module-global
    pyparsing grammar, so drawing from the batch thread pool corrupts the
    parser and raises ParseException on valid labels such as ``$m$`` or
    Matplotlib's own ``\\mathdefault`` log-tick macro, at random. Every
    figure is therefore serialized through ``_FIGURE_LOCK``.

    Separately, one unplottable source (an absent mode, an empty velocity
    trace) must not cost the later figures of that acquisition, which
    would look like whole vessels or epochs going missing. Any open figure
    is closed so a partial draw cannot leak.
    """
    with _FIGURE_LOCK:
        try:
            return plotter(**kwargs)
        except Exception as exc:  # one bad source must not stop the batch
            plt.close("all")
            # ``exc`` is unbound once the except block exits, so keep the
            # parts of it the warning needs.
            kind = type(exc).__name__
            detail = " ".join(str(exc).split())[:200] or kind
    warnings.warn(
        f"low-rank figure {name!r} could not be drawn: {kind}: {detail}",
        RuntimeWarning,
        stacklevel=2,
    )
    return None


OPTIONAL_FIGURES = frozenset({"fig3_waveform_decomposition_per_beat_beats"})


def report_missing_figures(
    source: str,
    expected: list[str],
    written: list[Path],
    *,
    source_label: str = "",
) -> list[str]:
    """Warn once naming the figures a source unexpectedly did not produce.

    A figure is skipped whenever its packed payload is absent, which is a
    normal outcome for some sources but indistinguishable from a bug when
    the run stays silent. Naming the gaps makes an incomplete output set
    self-explaining instead of something to reverse-engineer from folders.
    Variants in ``OPTIONAL_FIGURES`` are known to be unavailable and stay
    quiet, so the warning only ever reports a real surprise.
    """
    produced = {path.stem for path in written}
    missing = [
        name
        for name in expected
        if name not in OPTIONAL_FIGURES
        and not any(stem.endswith(name) for stem in produced)
    ]
    if missing:
        where = f"{source_label} {source}".strip()
        warnings.warn(
            f"low-rank figures not produced for {where} "
            f"({len(missing)} of {len(expected)}): {', '.join(missing)}. "
            "The packed EyeFlow payload for these variants is absent.",
            RuntimeWarning,
            stacklevel=2,
        )
    return missing


def finite_std(values: np.ndarray, *, ddof: int = 1) -> float:
    """Sample SD of finite values; 0 when fewer than two samples."""
    x = np.asarray(values, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return 0.0
    return float(np.std(x, ddof=ddof))


def mean_pm_std(
    values: np.ndarray,
    *,
    axis: int = 0,
    ddof: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nan-aware mean ± 1 sample SD along ``axis``.

    Used for every beat-to-beat (or along-sample) gray band: sample SD
    (``ddof=1``), SD = 0 when fewer than two finite samples. Returns
    ``(mean, mean - sd, mean + sd)``.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty
    with np.errstate(all="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(arr, axis=axis)
        n = np.sum(np.isfinite(arr), axis=axis)
        # Avoid numpy's "dof <= 0" warning when a slice has <2 finite values.
        sd = np.zeros_like(mean, dtype=float)
        ok = n > 1
        if np.any(ok):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                sd_all = np.nanstd(arr, axis=axis, ddof=ddof)
            sd = np.where(ok & np.isfinite(sd_all), sd_all, 0.0)
    mean = np.asarray(mean, dtype=float)
    sd = np.asarray(sd, dtype=float)
    return mean, mean - sd, mean + sd


def coerce_beat_spectra(
    values: np.ndarray | None,
    *,
    n_modes: int = SPECTRUM_N_MODES,
) -> np.ndarray:
    """Return ``(n_beats, n_modes)`` per-beat singular values, or empty."""
    arr = np.asarray([] if values is None else values, dtype=float)
    if arr.size == 0 or arr.ndim != 2 or arr.shape[1] < 2:
        return np.zeros((0, n_modes), dtype=float)
    if arr.shape[1] < n_modes:
        out = np.full((arr.shape[0], n_modes), np.nan, dtype=float)
        out[:, : arr.shape[1]] = arr
        return out
    return np.asarray(arr[:, :n_modes], dtype=float)


def is_usable_beat_spectra(values: np.ndarray | None) -> bool:
    """True if ``values`` is a 2-D spectrum with at least one finite entry."""
    arr = np.asarray([] if values is None else values, dtype=float)
    return arr.ndim == 2 and arr.shape[1] >= 2 and bool(np.any(np.isfinite(arr)))


def find_eyeflow_lowrank_group(h5file: h5py.File) -> h5py.Group | None:
    """Return EyeFlow's packed low-rank metrics group, if present."""
    for path in EYEFLOW_LOWRANK_GROUP_CANDIDATES:
        if path in h5file and isinstance(h5file[path], h5py.Group):
            return h5file[path]
    return None


def find_lowrank_metrics_group(h5file: h5py.File) -> h5py.Group | None:
    """Locate packed low-rank metrics in an open HDF5 file.

    Prefers the AngioEye ingest copy under ``/AngioEye/Processing/...``, then
    falls back to EyeFlow's ``Processing/Metrics/lowrank_waveform_decomposition``.
    Same loaders then work on ``*_EF.h5`` (pipeline run) and ``*_AE.h5`` (cohort).
    """
    angio = find_pipeline_group(h5file, PIPELINE_NAME)
    if angio is not None:
        return angio
    return find_eyeflow_lowrank_group(h5file)


def result_h5_has_lowrank(h5_path: Path | str) -> bool:
    """True when ``h5_path`` contains packed low-rank metrics (AE or EyeFlow)."""
    try:
        with h5py.File(h5_path, "r") as h5file:
            return find_lowrank_metrics_group(h5file) is not None
    except OSError:
        return False


def _group_dataset_map(group: h5py.Group) -> dict[str, np.ndarray]:
    """Copy immediate datasets in ``group`` to a name → array dict (no recursion)."""
    out: dict[str, np.ndarray] = {}
    for key, obj in group.items():
        if isinstance(obj, h5py.Dataset):
            out[key] = np.asarray(obj)
    return out


def _dataset_vector(group: h5py.Group | None, name: str) -> np.ndarray | None:
    """Read one numeric dataset as a finite-length vector."""
    if group is None or name not in group:
        return None
    obj = group[name]
    if not isinstance(obj, h5py.Dataset):
        return None
    arr = np.asarray(obj, dtype=float).reshape(-1)
    return arr if arr.size else None


def _group_has_datasets(group: h5py.Group | None) -> bool:
    """True when ``group`` directly contains at least one dataset."""
    return isinstance(group, h5py.Group) and any(
        isinstance(obj, h5py.Dataset) for obj in group.values()
    )


def _endpoint_group(source: h5py.Group, method: str) -> h5py.Group | None:
    """Official EyeFlow endpoint group: ``endpoints/joint`` or ``endpoints/per_beat``."""
    endpoints = source.get("endpoints")
    if not isinstance(endpoints, h5py.Group):
        return None
    child = endpoints.get(method)
    return child if isinstance(child, h5py.Group) else None


def _read_endpoint_values(group: h5py.Group | None) -> dict[str, np.ndarray]:
    """Read the official EyeFlow endpoint metric datasets from one endpoint group."""
    if not isinstance(group, h5py.Group):
        return {}
    out: dict[str, np.ndarray] = {}
    for key in ENDPOINT_METRICS:
        arr = _dataset_vector(group, key)
        if arr is not None:
            out[key] = arr
    return out


def _misc_group(source: h5py.Group, name: str) -> h5py.Group | None:
    """Return one official ``misc/<name>`` group."""
    misc = source.get("misc")
    if not isinstance(misc, h5py.Group):
        return None
    group = misc.get(name)
    return group if isinstance(group, h5py.Group) else None


_LEGACY_SPECTRUM_GROUPS = {
    "joint": ("fig4_energy_spectrum", {"mean": "lambda"}),
    "per_beat": (
        "fig4_energy_spectrum_pb",
        {"mean": "lambda_mean", "lo": "lambda_lo", "hi": "lambda_hi"},
    ),
}


def _legacy_figure_group(source: h5py.Group, name: str) -> h5py.Group | None:
    """Return one pre-``misc`` ``figures/<name>`` group."""
    figures = source.get("figures")
    if not isinstance(figures, h5py.Group):
        return None
    group = figures.get(name)
    return group if isinstance(group, h5py.Group) else None


def _spectrum_payload_from_source(
    source: h5py.Group,
    *,
    method: str,
    cumulative: bool,
) -> dict[str, np.ndarray] | None:
    """Official Fig. 4 payload from joint/per-beat ``misc`` spectrum groups.

    Cumulative spectra are no longer packed by EyeFlow.
    """
    if cumulative:
        return None
    method = normalize_svd_method(method)
    if method == "per_beat":
        group_name = EYEFLOW_MISC_SPECTRUM_PER_BEAT
        y_name = "lambda_mean"
        lo_name = "lambda_lo"
        hi_name = "lambda_hi"
    else:
        group_name = EYEFLOW_MISC_SPECTRUM_JOINT
        y_name = "lambda"
        lo_name = hi_name = ""

    group = _misc_group(source, group_name)
    mode = _dataset_vector(group, "svd_mode_index")
    mean = _dataset_vector(group, y_name)
    if mode is None or mean is None:
        # Archives packed before the ``misc`` layout keep the spectrum under
        # ``figures/fig4_energy_spectrum*`` with a ``mode`` index. Reading
        # both keeps cohort Fig. 4 available when reprocessing older data.
        legacy_group, legacy_names = _LEGACY_SPECTRUM_GROUPS[method]
        group = _legacy_figure_group(source, legacy_group)
        mode = _dataset_vector(group, "mode")
        mean = _dataset_vector(group, legacy_names["mean"])
        if mode is None or mean is None:
            return None
        lo_name = legacy_names.get("lo", "")
        hi_name = legacy_names.get("hi", "")
    n = min(mode.size, mean.size)
    if n == 0:
        return None
    payload = {"mode": mode[:n], "mean": mean[:n]}
    if method == "per_beat":
        lo = _dataset_vector(group, lo_name)
        hi = _dataset_vector(group, hi_name)
        if lo is not None and hi is not None:
            n = min(n, lo.size, hi.size)
            payload = {
                "mode": payload["mode"][:n],
                "mean": payload["mean"][:n],
                "lo": lo[:n],
                "hi": hi[:n],
            }
    return payload


def _group_flag_or_payload(
    source: h5py.Group,
    flag_name: str,
    *,
    default_from_payload: bool,
) -> bool:
    """Read ``qc/<flag_name>`` when present, else use payload inference."""
    flag = _scalar_from_group(source, f"qc/{flag_name}", float("nan"))
    if np.isfinite(flag):
        return int(flag) == 1
    return bool(default_from_payload)


def _source_group_has_metric_payload(source: h5py.Group) -> bool:
    """True when a source group has official endpoints, misc, or spectra."""
    endpoints = source.get("endpoints")
    if isinstance(endpoints, h5py.Group):
        for method in SVD_METHODS:
            if _group_has_datasets(endpoints.get(method)):
                return True
    misc = source.get("misc")
    if isinstance(misc, h5py.Group):
        for obj in misc.values():
            if isinstance(obj, h5py.Group):
                if _group_has_datasets(obj):
                    return True
                if any(isinstance(child, h5py.Group) for child in obj.values()):
                    return True
    decomp = source.get("decomposition")
    if isinstance(decomp, h5py.Group) and any(
        name in decomp for name in ("singular_values", "singular_energy_fraction")
    ):
        return True
    per_beat = source.get("per_beat")
    return isinstance(per_beat, h5py.Group) and any(
        name in per_beat for name in ("singular_values_b", "u_mode1_tb")
    )


def _scalar_from_group(
    group: h5py.Group | None,
    name: str,
    default: float = float("nan"),
) -> float:
    """Read one numeric dataset (nested path ok) as a Python float."""
    if group is None:
        return float(default)
    try:
        obj = group[name]
    except KeyError:
        return float(default)
    arr = np.asarray(obj).reshape(-1)
    if arr.size == 0:
        return float(default)
    return float(arr[0])


def load_vessel_data_from_result_h5(
    h5_path: Path | str,
    vessel: str,
    *,
    signal: str = COHORT_SIGNAL,
) -> dict | None:
    """Load one vessel's packed endpoints from HDF5 for cohort tables / Fig. 4.

    EyeFlow already wrote A1, λ_m, per-beat arrays, etc. into the file; this
    pipeline never recomputes them, so the H5 is the only source.
    """
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as h5file:
        root = find_lowrank_metrics_group(h5file)
        if root is None:
            return None
        source = root.get(f"{vessel}/{signal}")
        if not isinstance(source, h5py.Group):
            return None
        has_payload = _source_group_has_metric_payload(source)
        if not _group_flag_or_payload(
            source,
            "input_available",
            default_from_payload=has_payload,
        ):
            return None
        if not _group_flag_or_payload(
            source,
            "svd_available",
            default_from_payload=has_payload,
        ):
            return None

        joint_endpoints = _endpoint_group(source, "joint")
        per_beat_endpoints = _endpoint_group(source, "per_beat")
        variability = source["variability"] if "variability" in source else None
        decomposition = source["decomposition"] if "decomposition" in source else None
        baseline = source["baseline"] if "baseline" in source else None
        beatwise_g = source["beatwise"] if "beatwise" in source else None
        inputs = source["inputs"] if "inputs" in source else None
        beat_period = source["beat_period"] if "beat_period" in source else None

        acq: dict[str, float] = {}
        for key, arr in _read_endpoint_values(joint_endpoints).items():
            flat = np.asarray(arr, dtype=float).reshape(-1)
            acq[key] = float(flat[0]) if flat.size else float("nan")
        if isinstance(variability, h5py.Group):
            for key, arr in _group_dataset_map(variability).items():
                flat = np.asarray(arr, dtype=float).reshape(-1)
                acq[key] = float(flat[0]) if flat.size else float("nan")
        if isinstance(decomposition, h5py.Group):
            for key in (
                "Reff",
                "PR",
                "spectrum_mode_count_M",
            ):
                if key in decomposition:
                    acq[key] = _scalar_from_group(decomposition, key)
        if isinstance(baseline, h5py.Group):
            for key in ("mu_acq", "abs_mu_acq", "sigma_mu_beat", "mad_mu_beat"):
                if key in baseline:
                    acq[key] = _scalar_from_group(baseline, key)

        beatwise = (
            _group_dataset_map(beatwise_g) if isinstance(beatwise_g, h5py.Group) else {}
        )
        # EyeFlow packs per-beat means under baseline/, but cohort code expects
        # them on the beatwise dict (same shape cohort collectors expect).
        if isinstance(baseline, h5py.Group):
            if "mu_b" in baseline and "mu_b" not in beatwise:
                beatwise["mu_b"] = np.asarray(baseline["mu_b"], dtype=float)
            if "abs_mu_b" in baseline and "abs_mu_b" not in beatwise:
                beatwise["abs_mu_b"] = np.asarray(baseline["abs_mu_b"], dtype=float)

        per_beat_svd = _read_endpoint_values(per_beat_endpoints)
        legacy_per_beat = source["per_beat"] if "per_beat" in source else None
        if isinstance(legacy_per_beat, h5py.Group):
            legacy_per_beat_svd = {
                key: value
                for key, value in _group_dataset_map(legacy_per_beat).items()
                if not key.startswith("u_mode") and not key.startswith("scores_mode")
            }
            if "singular_values_b" in legacy_per_beat_svd:
                per_beat_svd["singular_values_b"] = legacy_per_beat_svd[
                    "singular_values_b"
                ]
        per_beat_svd["singular_values_b"] = coerce_beat_spectra(
            per_beat_svd.get("singular_values_b")
        )

        singular_values = np.asarray([], dtype=float)
        energy_fraction = np.asarray([], dtype=float)
        per_beat_spectrum = np.asarray([], dtype=float)
        per_beat_spectrum_sd = np.asarray([], dtype=float)
        joint_spectrum_payload = _spectrum_payload_from_source(
            source, method="joint", cumulative=False
        )
        if joint_spectrum_payload is not None:
            singular_values = np.asarray(joint_spectrum_payload["mean"], dtype=float)
        pb_spectrum_payload = _spectrum_payload_from_source(
            source, method="per_beat", cumulative=False
        )
        if pb_spectrum_payload is not None:
            per_beat_spectrum = np.asarray(pb_spectrum_payload["mean"], dtype=float)
            lo = np.asarray(pb_spectrum_payload.get("lo", []), dtype=float)
            hi = np.asarray(pb_spectrum_payload.get("hi", []), dtype=float)
            n_sd = min(lo.size, hi.size, per_beat_spectrum.size)
            if n_sd:
                # EyeFlow packs lambda_lo/hi as mean -/+ 1 SD across beats,
                # so the half-width recovers the across-beats sample SD.
                per_beat_spectrum_sd = (hi[:n_sd] - lo[:n_sd]) / 2.0
        if isinstance(decomposition, h5py.Group):
            if singular_values.size == 0 and "singular_values" in decomposition:
                singular_values = np.asarray(
                    decomposition["singular_values"], dtype=float
                )
            if "singular_energy_fraction" in decomposition:
                energy_fraction = np.asarray(
                    decomposition["singular_energy_fraction"], dtype=float
                )

        if (
            isinstance(inputs, h5py.Group)
            and "valid_fraction_columns_per_beat" in inputs
        ):
            vfb = np.asarray(inputs["valid_fraction_columns_per_beat"], dtype=float)
        else:
            vfb = np.asarray([], dtype=float)

        if isinstance(beat_period, h5py.Group):
            beat_period_mean = _scalar_from_group(beat_period, "mean")
            beat_period_sd = _scalar_from_group(beat_period, "std")
        else:
            beat_period_mean = float("nan")
            beat_period_sd = float("nan")

        if "mu_b" in beatwise:
            period_b = np.full(np.asarray(beatwise["mu_b"]).shape[0], beat_period_mean)
        elif "R0_b" in beatwise:
            period_b = np.full(np.asarray(beatwise["R0_b"]).shape[0], beat_period_mean)
        elif "R0" in per_beat_svd:
            period_b = np.full(
                np.asarray(per_beat_svd["R0"]).shape[0], beat_period_mean
            )
        else:
            period_b = np.asarray([], dtype=float)

        n_valid = (
            int(_scalar_from_group(inputs, "n_valid_columns", 0))
            if isinstance(inputs, h5py.Group)
            else 0
        )
        n_total = (
            int(_scalar_from_group(inputs, "n_total_columns", 0))
            if isinstance(inputs, h5py.Group)
            else 0
        )

    return {
        "acq": acq,
        "beatwise": beatwise,
        "per_beat_svd": per_beat_svd,
        "mu": acq.get("mu_acq", float("nan")),
        "energy_fraction": energy_fraction,
        "singular_values": singular_values,
        "per_beat_spectrum": per_beat_spectrum,
        "per_beat_spectrum_sd": per_beat_spectrum_sd,
        "beat_period_mean": beat_period_mean,
        "beat_period_sd": beat_period_sd,
        "beat_period_b": period_b,
        "valid_fraction_per_beat": vfb,
        "n_valid_columns": n_valid,
        "n_total_columns": n_total,
    }


def load_acquisition_from_result_h5(
    h5_path: Path | str,
    *,
    veins_flag: bool = False,
    signal: str = COHORT_SIGNAL,
) -> dict[str, dict | None] | None:
    """Load artery/(optional) vein bundles from packed AE or EyeFlow metrics."""
    h5_path = Path(h5_path)
    if not result_h5_has_lowrank(h5_path):
        return None
    result: dict[str, dict | None] = {}
    any_ok = False
    for vessel in enabled_vessels(veins_flag):
        data = load_vessel_data_from_result_h5(h5_path, vessel, signal=signal)
        result[vessel] = data
        if data is not None:
            any_ok = True
    return result if any_ok else None
