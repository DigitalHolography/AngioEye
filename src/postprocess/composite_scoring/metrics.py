from .dataclasses import Metric

REPRESENTATIONS = ("raw", "bandlimited")
VESSEL_TYPES = ("artery", "vein")
POSTPROCESS_GROUP = "composite_scoring"
PLOT_VESSEL_TYPE = "artery"

# Direction constants used after automatic calibration.
GREATER = 1
LESS = -1

# -----------------------------------------------------------------------------
# Complete metric panel from:
# "Transportable retinal Doppler holography waveform-shape metrics..."
#
# IMPORTANT:
# - No fixed threshold here.
# - No fixed direction here.
# - No fixed control_std here.
# These are filled automatically by optimal split calibration.
# -----------------------------------------------------------------------------

# VTI-weighted temporal moments
M_CENTROID_TIMING = "centroid_timing"
M_TEMPORAL_SPREAD = "temporal_spread"
M_TEMPORAL_SKEWNESS = "temporal_skewness"

# Displacement timing quantiles tq/T
M_T10_TIMING = "t10_displacement_timing"
M_T25_TIMING = "t25_displacement_timing"
M_T50_TIMING = "t50_displacement_timing"
M_T75_TIMING = "t75_displacement_timing"
M_T90_TIMING = "t90_displacement_timing"

# Time-quantile geometry
M_QT_WIDTH = "time_quantile_width"
M_QT_SKEW = "time_quantile_skew"

# Cumulative-distance geometry
M_DELTA_DTI = "pulse_front_loading_index"
M_D10_FRACTION = "d10_cumulative_distance_fraction"
M_D25_FRACTION = "d25_cumulative_distance_fraction"
M_D50_FRACTION = "d50_cumulative_distance_fraction"
M_D75_FRACTION = "d75_cumulative_distance_fraction"
M_D90_FRACTION = "d90_cumulative_distance_fraction"
M_QD_WIDTH = "cumulative_distance_width"
M_QD_SKEW = "cumulative_distance_skew"
M_EARLY_LATE_BALANCE = "early_late_balance"
M_STROKE_FRACTION = "stroke_fraction"

# Crest morphology
M_NEAR_PEAK_CREST_WIDTH = "near_peak_crest_width"
M_SUMMIT_CREST_WIDTH = "summit_crest_width"

# Excursion and pulsatility
M_RESISTIVITY_INDEX = "resistivity_index"
M_PULSATILITY_INDEX = "pulsatility_index"
M_CREST_FACTOR = "crest_factor"

# Event timings and slope kinetics
M_PEAK_TIMING = "peak_timing"
M_TROUGH_TIMING = "trough_timing"
M_UPSTROKE_STEEPNESS = "upstroke_steepness"
M_DOWNSTROKE_STEEPNESS = "downstroke_steepness"
M_UPSTROKE_TIMING = "upstroke_timing"
M_DOWNSTROKE_TIMING = "downstroke_timing"
M_SLOPE_ENERGY = "slope_energy"

# Late-cycle persistence / temporal support
M_LATE_CYCLE_MEAN_FRACTION = "late_cycle_mean_fraction"
M_EFFECTIVE_DURATION = "effective_duration"
M_ENTROPIC_DURATION = "entropic_duration"

# Spectral and representation-fidelity descriptors
M_LOW_FREQ_SPECTRAL_RATIO = "low_freq_spectral_ratio"
M_RECONSTRUCTION_FIDELITY = "reconstruction_fidelity"


METRIC_PANEL: dict[str, Metric] = {
    # VTI-weighted temporal moments
    M_CENTROID_TIMING: Metric(name="mu_t_over_T"),
    M_TEMPORAL_SPREAD: Metric(name="sigma_t_over_T"),
    M_TEMPORAL_SKEWNESS: Metric(name="gamma_t"),
    # Displacement timing quantiles tq/T
    M_T50_TIMING: Metric(name="t50_over_T"),
    # Time-quantile geometry
    M_QT_WIDTH: Metric(name="Q_t_width"),
    M_QT_SKEW: Metric(name="Q_t_skew"),
    # Cumulative-distance geometry
    M_DELTA_DTI: Metric(name="Delta_DTI"),
    M_QD_WIDTH: Metric(name="Q_d_width"),
    M_QD_SKEW: Metric(name="Q_d_skew"),
    M_EARLY_LATE_BALANCE: Metric(name="R_VTI"),
    M_STROKE_FRACTION: Metric(name="SF_VTI"),
    # Crest morphology
    M_NEAR_PEAK_CREST_WIDTH: Metric(name="W50_over_T"),
    M_SUMMIT_CREST_WIDTH: Metric(name="W80_over_T"),
    # Excursion and pulsatility
    M_RESISTIVITY_INDEX: Metric(name="RI"),
    M_PULSATILITY_INDEX: Metric(name="PI"),
    M_CREST_FACTOR: Metric(name="CF"),
    # Event timings and slope kinetics
    M_PEAK_TIMING: Metric(name="tmax_over_T"),
    M_TROUGH_TIMING: Metric(name="tmin_over_T"),
    M_UPSTROKE_STEEPNESS: Metric(name="S_rise"),
    M_DOWNSTROKE_STEEPNESS: Metric(name="S_fall"),
    M_UPSTROKE_TIMING: Metric(name="t_rise_over_T"),
    M_DOWNSTROKE_TIMING: Metric(name="t_fall_over_T"),
    M_SLOPE_ENERGY: Metric(name="E_slope"),
    # Late-cycle persistence / temporal support
    M_LATE_CYCLE_MEAN_FRACTION: Metric(name="v_end_over_vbar"),
    M_EFFECTIVE_DURATION: Metric(name="N_eff_over_T"),
    M_ENTROPIC_DURATION: Metric(name="N_t_over_T"),
    # Spectral and representation-fidelity descriptors
    # The paper defines ELF/EHF. If your pipeline stores E_low/E_high instead,
    # keep numerator_name/denominator_name but adapt names to the HDF5 datasets.
    M_LOW_FREQ_SPECTRAL_RATIO: Metric(
        name="E_LF_over_E_HF",
    ),
}

# Backward-compatible alias for older imports.
METRICS = METRIC_PANEL
