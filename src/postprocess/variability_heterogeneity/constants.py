from input_output.hdf5_schema import pipeline_path_candidates


def _segment_metric_folders() -> tuple[str, ...]:
    folders: list[str] = []
    for pipeline_name in (
        "waveform_shape_metrics_denoised",
        "waveform_shape_metrics",
    ):
        folders.extend(
            path.rstrip("/")
            for path in pipeline_path_candidates(
                pipeline_name,
                "artery",
                "by_segment",
            )
        )
    return tuple(folders)


SEGMENT_METRIC_FOLDERS = (
    *_segment_metric_folders(),
)
SEGMENT_MODE = "bandlimited_segment"
EPS = 1e-12

DEFAULT_TOP_N = 10
PLOT_STYLE = "default"
PNG_PIL_KWARGS = {"compress_level": 1}

CONTROL_GROUP_PATTERNS = [
    r"^control$",
    r"^controle$",
    r"^controls$",
    r"^ctrl$",
    r"^ctl$",
    r"^BL$",
    r"^healthy$",
    r"^healthy_control$",
    r"^healthy_controls$",
    r"^BL$",
]

INPUT_METRICS = [
    "mu_t_over_T",
    "RI",
    "PI",
    "R_VTI",
    "SF_VTI",
    "sigma_t_over_T",
    "W50_over_T",
    "W80_over_T",
    "E_LF_over_E_HF",
    "t_max_over_T",
    "t_min_over_T",
    "S_rise",
    "S_fall",
    "t_rise_over_T",
    "t_fall_over_T",
    "CF",
    "Delta_DTI",
    "gamma_t",
    "N_eff_over_T",
    "N_t_over_T",
    "Q_t_skew",
    "Q_t_width",
    "Q_d_skew",
    "Q_d_width",
    "v_end_over_vbar",
    "E_slope",
    "t50_over_T",
]

METRIC_LABELS = {
    "RI": r"$\rm RI$",
    "CF": r"$\rm CF$",
    "t50_over_T": r"$t_{50}/T$",
    "R_VTI": r"$R_{\mathrm{VTI}}$",
    "mu_t_over_T": r"$\mu_t/T$",
    "PI": r"$\rm PI$",
    "SF_VTI": r"$SF_{\mathrm{VTI}}$",
    "sigma_t_over_T": r"$\sigma_t/T$",
    "t_max_over_T": r"$t_{\mathrm{max}}/T$",
    "t_min_over_T": r"$t_{\mathrm{min}}/T$",
    "t_rise_over_T": r"$t_{\mathrm{rise}}/T$",
    "t_fall_over_T": r"$t_{\mathrm{fall}}/T$",
    "Delta_DTI": r"$\Delta_{\mathrm{DTI}}$",
    "E_LF_over_E_HF": r"$E_{\mathrm{LF}}/E_{\mathrm{HF}}$",
    "S_fall": r"$S_{\mathrm{fall}}$",
    "S_rise": r"$S_{\mathrm{rise}}$",
    "gamma_t": r"$\gamma_t$",
    "N_eff_over_T": r"$N_{\mathrm{eff}}/T$",
    "Q_t_skew": r"$Q_{\mathrm{t,skew}}$",
    "Q_t_width": r"$Q_{\mathrm{t,width}}$",
    "Q_d_skew": r"$Q_{\mathrm{d,skew}}$",
    "Q_d_width": r"$Q_{\mathrm{d,width}}$",
    "v_end_over_vbar": r"$\bar{\mathrm{v}}_{\mathrm{end}}/\bar{\mathrm{v}}$",
    "E_slope": r"$E_{\mathrm{slope}}$",
    "W50_over_T": r"$W_{50}/T$",
    "W80_over_T": r"$W_{80}/T$",
    "N_t_over_T": r"$N_t/T$",
    "eta_h": r"$\eta_h$",
}

COLUMN_LABELS = {
    "MED_seg_medbeat": r"$\mathrm{med}_{b}(\mathrm{med}_{seg})$",
    "STD_seg_medbeat": r"$\mathrm{med}_{b}(\mathrm{STD}_{seg})$",
    "IQR_seg_medbeat": r"$\mathrm{med}_{b}(\mathrm{IQR}_{seg})$",
    "MAD_seg_medbeat": r"$\mathrm{med}_{b}(\mathrm{MAD}_{seg})$",
    "CV_seg_medbeat": r"$\mathrm{med}_{b}(\mathrm{CV}_{seg})$",
    "STD_beat_medseg": r"$\mathrm{med}_{seg}(\mathrm{STD}_{b})$",
    "IQR_beat_medseg": r"$\mathrm{med}_{seg}(\mathrm{IQR}_{b})$",
    "MAD_beat_medseg": r"$\mathrm{med}_{seg}(\mathrm{MAD}_{b})$",
    "CV_beat_medseg": r"$\mathrm{med}_{seg}(\mathrm{CV}_{b})$",
}

SPATIAL_VARIABILITY_COLUMNS = [
    "STD_seg_medbeat",
    "IQR_seg_medbeat",
    "MAD_seg_medbeat",
    "CV_seg_medbeat",
]
TEMPORAL_VARIABILITY_COLUMNS = [
    "STD_beat_medseg",
    "MAD_beat_medseg",
    "CV_beat_medseg",
]
SPATIAL_RAW_COLUMNS = ["MED_seg_medbeat", *SPATIAL_VARIABILITY_COLUMNS]
TEMPORAL_RAW_COLUMNS = ["MED_seg_medbeat", *TEMPORAL_VARIABILITY_COLUMNS]

SUMMARY_PVALUE_METRICS = ["RI", "PI", "N_t_over_T", "N_eff_over_T"]
SPATIAL_SELECTED_METRICS = ["RI", "PI", "t50_over_T", "v_end_over_vbar"]
TEMPORAL_SELECTED_METRICS = ["N_t_over_T", "N_eff_over_T", "RI", "t50_over_T"]

DESCRIPTOR_LABELS = {
    "STD": r"$\mathrm{STD}$",
    "IQR": r"$\mathrm{IQR}$",
    "MAD": r"$\mathrm{MAD}$",
    "CV": r"$\mathrm{CV}$",
}
SPATIAL_DESCRIPTOR_MAP = {
    "STD": "STD_seg_medbeat",
    "IQR": "IQR_seg_medbeat",
    "MAD": "MAD_seg_medbeat",
    "CV": "CV_seg_medbeat",
}
TEMPORAL_DESCRIPTOR_MAP = {
    "STD": "STD_beat_medseg",
    "MAD": "MAD_beat_medseg",
    "CV": "CV_beat_medseg",
}
