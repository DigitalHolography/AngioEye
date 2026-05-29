from .dataclasses import Domain, Metric


REPRESENTATIONS = ("raw", "bandlimited")
VESSEL_TYPES = ("artery", "vein")
POSTPROCESS_GROUP = "composite_scoring"
PLOT_VESSEL_TYPE = "artery"


# Metrics Definitions
M_STROKE_FRACTION = "stroke_fraction"
M_MED_DISPLACEMENT_TIMING = "med_displacement_timing"
M_LOW_FREQ_SPECTRAL_FRACTION = "low_freq_spectral_fraction"
M_LATE_CYCLE_MEAN_FRACTION = "late_cycle_mean_fraction"
M_PARTICIPATION_RATIO_EFF_SUPP = "participation_ratio_eff_supp"
M_RESISTIVITY_INDEX = "resistivity_index"
M_PULSATILITY_INDEX = "pulsatility_index"
M_NEAR_PEAK_CREST_WIDTH = "near_peak_crest_width"

# Metrics Threshold Directions
GREATER = 1
LESS = -1

# Metrics Settings
METRICS: dict[str, Metric] = {
    M_STROKE_FRACTION: Metric(
        name="SF_VTI",
        threshold=0.5,
        direction=GREATER,
        control_std={"artery": 0.02130616468479075, "vein": 0.012238142379889327},
    ),
    M_MED_DISPLACEMENT_TIMING: Metric(
        name="t50_over_T",
        threshold=0.36,
        direction=LESS,
        control_std={"artery": 0.02170372191846459, "vein": 0.011114571157947383},
    ),
    M_LOW_FREQ_SPECTRAL_FRACTION: Metric(
        name="E_low_over_E_total",
        threshold=0.76,
        direction=GREATER,
        control_std={"artery": 0.0669936550252201, "vein": 0.06952508377563454},
        numerator_name="E_low",
        denominator_name="E_total",
    ),
    M_LATE_CYCLE_MEAN_FRACTION: Metric(
        name="v_end_over_vbar",
        threshold=0.59,
        direction=LESS,
        control_std={"artery": 0.06584969658865907, "vein": 0.0382835422777238},
    ),
    M_PARTICIPATION_RATIO_EFF_SUPP: Metric(
        name="N_eff_over_T",
        threshold=0.90,
        direction=LESS,
        control_std={"artery": 0.02584899777470274, "vein": 0.0055469432407653264},
    ),
    M_RESISTIVITY_INDEX: Metric(
        name="RI",
        threshold=0.75,
        direction=GREATER,
        control_std={"artery": 0.08357600130504828, "vein": 0.063285564839462},
    ),
    M_PULSATILITY_INDEX: Metric(
        name="PI",
        threshold=1.30,
        direction=GREATER,
        control_std={"artery": 0.2003058879383459, "vein": 0.07992210522433744},
    ),
    M_NEAR_PEAK_CREST_WIDTH: Metric(
        name="W50_over_T",
        threshold=0.60,
        direction=LESS,
        control_std={"artery": 0.13043404441873044, "vein": 0.003223005055989731},
    ),
}

# Separation of Metrics into Domains, with weights
DOMAINS: dict[str, Domain] = {
    "timing": Domain(
        metrics=(M_STROKE_FRACTION, M_MED_DISPLACEMENT_TIMING),
        weight=1,
    ),
    "spectral": Domain(metrics=(M_LOW_FREQ_SPECTRAL_FRACTION,), weight=1.5),
    "persistence": Domain(
        metrics=(
            M_LATE_CYCLE_MEAN_FRACTION,
            M_PARTICIPATION_RATIO_EFF_SUPP,
            M_NEAR_PEAK_CREST_WIDTH,
        ),
        weight=1,
    ),
    "pulsatility": Domain(
        metrics=(M_RESISTIVITY_INDEX, M_PULSATILITY_INDEX),
        weight=1,
    ),
}