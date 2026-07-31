import numpy as np

from input_output.eyeflow_schema import has_path, require_dataset

from .core.base import ProcessResult, registerPipeline, with_attrs

# À adapter si le nom du fichier contenant ArterialSegExample est différent.
# Exemple possible :
# from .waveform_shape_metrics_denoised import ArterialSegExample
from .waveform_shape_metrics_denoised import ArterialSegExample


@registerPipeline(name="waveform_shape_metrics_filter_sweep")
class WaveformShapeMetricsFilterSweep(ArterialSegExample):
    """
    Pipeline qui applique plusieurs filtres candidats sur les signaux segmentaires
    artériels, puis calcule les mêmes métriques waveform pour chaque filtre.

    Sortie H5 finale attendue :

    /AngioEye/Processing/waveform_shape_metrics_filter_sweep/
        artery/
            by_segment_filter_sweep/
                raw/
                    raw_segment/
                        RI
                        PI
                        ...
                hampel_w7_s3/
                    raw_segment/
                        RI
                        PI
                        ...
                hampel_savgol_w9/
                    raw_segment/
                        RI
                        PI
                        ...
                ...

    Chaque métrique raw_segment est stockée en :
        (beat, branch, radius)

    Le signal d'entrée correspondant était :
        signal[:, beat, branch, radius]
    """

    description = (
        "Filter sweep on arterial segment waveforms: compute waveform metrics "
        "for several filter candidates."
    )
    FILTER_SWEEP_METRICS = [
        "RI",
        "PI",
        "t50_over_T",
        "N_eff_over_T",
        "N_t_over_T",
        "Q_t_width",
    ]

    # -------------------------------------------------------------------------
    # Configurations de filtres à tester
    # -------------------------------------------------------------------------

    FILTER_SWEEP_CONFIGS = [
        # 1. Référence indispensable
        {
            "id": "raw",
            "hampel": False,
            "harmonic": False,
            "gaussian": False,
            "savgol": False,
            "weights": None,
        },
        # 2. Outlier removal seul
        # Très intéressant pour voir si les artefacts ponctuels expliquent la séparation.
        {
            "id": "hampel_w7_s3",
            "hampel": True,
            "hampel_window": 7,
            "hampel_nsigmas": 3.0,
            "harmonic": False,
            "gaussian": False,
            "savgol": False,
            "weights": None,
        },
        # 3. Hampel plus Savitzky-Golay doux
        # Bon compromis pour préserver RI/PI et t50.
        {
            "id": "hampel_savgol_w7",
            "hampel": True,
            "hampel_window": 7,
            "hampel_nsigmas": 3.0,
            "savgol": True,
            "savgol_window": 7,
            "savgol_polyorder": 2,
            "harmonic": False,
            "gaussian": False,
            "weights": {"savgol": 1.0},
        },
        # 4. Hampel plus Savitzky-Golay moyen
        # Probablement le meilleur candidat général.
        {
            "id": "hampel_savgol_w9",
            "hampel": True,
            "hampel_window": 7,
            "hampel_nsigmas": 3.0,
            "savgol": True,
            "savgol_window": 9,
            "savgol_polyorder": 2,
            "harmonic": False,
            "gaussian": False,
            "weights": {"savgol": 1.0},
        },
        # 5. Hampel plus harmonic lowpass
        # Intéressant pour N_eff, N_t et Q_t_width, mais peut modifier RI/PI.
        {
            "id": "hampel_harmonic_h8",
            "hampel": True,
            "hampel_window": 7,
            "hampel_nsigmas": 3.0,
            "harmonic": True,
            "harmonic_count": 8,
            "gaussian": False,
            "savgol": False,
            "weights": {"harmonic": 1.0},
        },
        # 6. Consensus morphologique
        # À tester comme candidat final : Savitzky-Golay dominant + un peu d'harmonique.
        {
            "id": "hampel_savgol70_harmonic30",
            "hampel": True,
            "hampel_window": 7,
            "hampel_nsigmas": 3.0,
            "savgol": True,
            "savgol_window": 9,
            "savgol_polyorder": 2,
            "harmonic": True,
            "harmonic_count": 8,
            "gaussian": False,
            "weights": {
                "savgol": 0.70,
                "harmonic": 0.30,
            },
        },
    ]

    # -------------------------------------------------------------------------
    # Helpers paramètres temporaires
    # -------------------------------------------------------------------------

    def _call_with_temp_attrs(self, updates: dict, func, *args, **kwargs):
        old_values = {}

        for name, value in updates.items():
            old_values[name] = getattr(self, name)
            setattr(self, name, value)

        try:
            return func(*args, **kwargs)
        finally:
            for name, value in old_values.items():
                setattr(self, name, value)

    # -------------------------------------------------------------------------
    # Application d'un filtre à un pulse 1D
    # -------------------------------------------------------------------------

    def _apply_filter_config_1d(
        self,
        pulse: np.ndarray,
        config: dict,
    ) -> np.ndarray:
        pulse = np.asarray(pulse, dtype=float)
        finite_mask = np.isfinite(pulse)

        if pulse.size == 0 or not np.any(finite_mask):
            return np.full_like(pulse, np.nan, dtype=float)

        if config["id"] == "raw":
            return pulse.copy()

        x = np.arange(pulse.size, dtype=float)
        filled = np.interp(x, x[finite_mask], pulse[finite_mask])
        base = filled.copy()

        if config.get("hampel", False):
            base = self._hampel_filter_1d(
                base,
                window=int(config.get("hampel_window", self.denoise_hampel_window)),
                n_sigmas=float(
                    config.get("hampel_nsigmas", self.denoise_hampel_nsigmas)
                ),
            )

        candidates = {}

        if config.get("harmonic", False):
            candidates["harmonic"] = self._call_with_temp_attrs(
                {
                    "denoise_harmonic_count": int(
                        config.get("harmonic_count", self.denoise_harmonic_count)
                    )
                },
                self._denoise_harmonic_lowpass,
                base,
            )

        if config.get("gaussian", False):
            candidates["gaussian"] = self._call_with_temp_attrs(
                {
                    "denoise_gaussian_sigma_samples": float(
                        config.get(
                            "gaussian_sigma",
                            self.denoise_gaussian_sigma_samples,
                        )
                    )
                },
                self._denoise_gaussian_smooth,
                base,
            )

        if config.get("savgol", False):
            candidates["savgol"] = self._call_with_temp_attrs(
                {
                    "denoise_savgol_window": int(
                        config.get("savgol_window", self.denoise_savgol_window)
                    ),
                    "denoise_savgol_polyorder": int(
                        config.get(
                            "savgol_polyorder",
                            self.denoise_savgol_polyorder,
                        )
                    ),
                },
                self._denoise_savgol_smooth,
                base,
            )

        weights = config.get("weights", None)

        if not candidates:
            filtered = base.copy()

        elif weights is None:
            filtered = np.nanmean(
                np.stack(list(candidates.values()), axis=0),
                axis=0,
            )

        else:
            filtered = np.zeros_like(base, dtype=float)
            total_weight = 0.0

            for name, weight in weights.items():
                if name not in candidates:
                    continue

                filtered += float(weight) * candidates[name]
                total_weight += float(weight)

            if total_weight <= self.eps:
                filtered = np.nanmean(
                    np.stack(list(candidates.values()), axis=0),
                    axis=0,
                )
            else:
                filtered /= total_weight

        filtered = self._clip_to_input_range(filtered, pulse)
        filtered[~finite_mask] = np.nan

        return filtered

    # -------------------------------------------------------------------------
    # Application d'un filtre à tout le bloc segmentaire
    # -------------------------------------------------------------------------

    def _filter_segment_block_with_config(
        self,
        v_block: np.ndarray,
        config: dict,
    ) -> tuple[np.ndarray, dict]:
        if v_block.ndim != 4:
            raise ValueError(
                f"Expected (time, beat, branch, radius), got {v_block.shape}"
            )

        n_time, n_beats, n_branches, n_radii = v_block.shape

        out = np.full_like(v_block, np.nan, dtype=float)
        corr = np.full((n_beats, n_branches, n_radii), np.nan, dtype=float)
        finite_fraction = np.zeros((n_beats, n_branches, n_radii), dtype=float)
        status_code = np.full((n_beats, n_branches, n_radii), 1, dtype=int)

        for beat_idx in range(n_beats):
            for branch_idx in range(n_branches):
                for radius_idx in range(n_radii):
                    index = (beat_idx, branch_idx, radius_idx)
                    pulse = np.asarray(
                        v_block[:, beat_idx, branch_idx, radius_idx],
                        dtype=float,
                    )

                    finite_mask = np.isfinite(pulse)
                    finite_count = int(np.sum(finite_mask))
                    finite_fraction[index] = finite_count / float(n_time)

                    if finite_count == 0:
                        status_code[index] = 1
                        continue

                    if not self._denoise_has_enough_valid_samples(
                        finite_count,
                        n_time,
                    ):
                        status_code[index] = 2
                        continue

                    if float(np.nanstd(pulse)) <= self.eps:
                        status_code[index] = 3
                        out[:, beat_idx, branch_idx, radius_idx] = np.where(
                            finite_mask,
                            pulse,
                            np.nan,
                        )
                        continue

                    filtered = self._apply_filter_config_1d(pulse, config)

                    out[:, beat_idx, branch_idx, radius_idx] = filtered
                    corr[index] = self._pearson_corr(pulse, filtered)
                    status_code[index] = 0

        diagnostics = {
            "original_vs_filtered_corr": corr,
            "finite_fraction": finite_fraction,
            "status_code": status_code,
        }

        return out, diagnostics

    # -------------------------------------------------------------------------
    # Écriture des métriques pour tous les filtres
    # -------------------------------------------------------------------------

    def _pack_filter_sweep_segment_outputs(
        self,
        metrics: dict,
        vessel_prefix: str,
        v_raw_seg: np.ndarray,
        T: np.ndarray,
    ) -> None:
        for config in self.FILTER_SWEEP_CONFIGS:
            filter_id = config["id"]

            v_filtered, diag = self._filter_segment_block_with_config(
                v_raw_seg,
                config,
            )

            seg, br, gl, _nb, _nr, seg_note = self._compute_block_segment(
                v_filtered,
                T,
            )

            base = f"{vessel_prefix}/by_segment_filter_sweep/{filter_id}"

            for metric_name, arr in seg.items():
                if metric_name not in self.FILTER_SWEEP_METRICS:
                    continue
                metrics[f"{base}/raw_segment/{metric_name}"] = with_attrs(
                    arr,
                    {
                        "definition": [
                            "per-segment metrics stored as (beat, branch, radius)"
                        ],
                        "segment_indexing": [seg_note],
                        "filter_id": [filter_id],
                    },
                )

            for metric_name, arr in br.items():
                if metric_name not in self.FILTER_SWEEP_METRICS:
                    continue
                metrics[f"{base}/raw_branch/{metric_name}"] = with_attrs(
                    arr,
                    {
                        "definition": ["median over radii per branch"],
                        "filter_id": [filter_id],
                    },
                )

            for metric_name, arr in gl.items():
                if metric_name not in self.FILTER_SWEEP_METRICS:
                    continue
                metrics[f"{base}/raw_global/{metric_name}"] = with_attrs(
                    arr,
                    {
                        "definition": [
                            "median over all branch-radius segment values per beat"
                        ],
                        "filter_id": [filter_id],
                    },
                )

            metrics[f"{base}/diagnostics/original_vs_filtered_corr"] = with_attrs(
                diag["original_vs_filtered_corr"],
                {
                    "definition": [
                        "Pearson correlation between original and filtered signal"
                    ],
                    "axis_order": ["beat, branch, radius"],
                    "filter_id": [filter_id],
                },
            )

            metrics[f"{base}/diagnostics/finite_fraction"] = with_attrs(
                diag["finite_fraction"],
                {
                    "definition": ["Fraction of finite samples in original signal"],
                    "axis_order": ["beat, branch, radius"],
                    "filter_id": [filter_id],
                },
            )

            metrics[f"{base}/diagnostics/status_code"] = with_attrs(
                diag["status_code"],
                {
                    "definition": [
                        "0 filtered, 1 all_nan, 2 too_sparse, 3 low_variance"
                    ],
                    "axis_order": ["beat, branch, radius"],
                    "filter_id": [filter_id],
                },
            )

            metrics[f"{base}/params/filter_id"] = np.asarray(filter_id, dtype="S")

            for key, value in config.items():
                if key == "weights":
                    continue

                if isinstance(value, bool):
                    metrics[f"{base}/params/{key}"] = np.asarray(value, dtype=bool)
                elif isinstance(value, int):
                    metrics[f"{base}/params/{key}"] = np.asarray(value, dtype=int)
                elif isinstance(value, float):
                    metrics[f"{base}/params/{key}"] = np.asarray(value, dtype=float)
                elif isinstance(value, str):
                    metrics[f"{base}/params/{key}"] = np.asarray(value, dtype="S")
                else:
                    metrics[f"{base}/params/{key}"] = np.asarray(str(value), dtype="S")

            weights = config.get("weights", None)
            if weights is not None:
                for name, weight in weights.items():
                    metrics[f"{base}/params/weights/{name}"] = np.asarray(
                        weight,
                        dtype=float,
                    )

    # -------------------------------------------------------------------------
    # Entrée principale
    # -------------------------------------------------------------------------

    def run(self, h5file) -> ProcessResult:
        metrics = {}

        if not has_path(h5file, self.T_input):
            return ProcessResult(metrics=metrics)

        T = np.asarray(require_dataset(h5file, self.T_input))

        artery_have_seg = (
            has_path(h5file, self.v_raw_segment_input)
            and has_path(h5file, self.v_band_segment_input)
        )

        if artery_have_seg:
            v_raw_seg = np.asarray(
                require_dataset(h5file, self.v_raw_segment_input),
                dtype=float,
            )

            self._pack_filter_sweep_segment_outputs(
                metrics=metrics,
                vessel_prefix="artery",
                v_raw_seg=v_raw_seg,
                T=T,
            )

        return ProcessResult(metrics=metrics)
