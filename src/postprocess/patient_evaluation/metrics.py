from __future__ import annotations

from .dataclasses import FixedMetric

GREATER = 1
LESS = -1

DEFAULT_VESSEL_TYPE = "artery"
DEFAULT_REPRESENTATION = "bandlimited"
DEFAULT_AGGREGATION = "median"
POSTPROCESS_GROUP = "fixed_threshold_evaluation"

# The score is an index of compatibility with the fixed pathological pattern.
# These bands are descriptive only and are not clinically validated probabilities.
EVALUATION_BANDS: tuple[tuple[float, str], ...] = (
    (0.33, "faible compatibilité avec le profil pathologique"),
    (0.67, "compatibilité intermédiaire avec le profil pathologique"),
    (1.01, "forte compatibilité avec le profil pathologique"),
)

# Edit this tuple to define the fixed panel used in production.
#
# control_std=None:
#     binary contribution: 0 before the threshold, 1 after the threshold.
#
# control_std=<positive float>:
#     graded contribution:
#       z = max(0, direction * (value - threshold) / control_std)
#       z_capped = min(1, z)
#
# The initial thresholds below are the illustrative pathology-oriented thresholds
# from the composite-waveform-biomarker manuscript. They must be replaced by
# thresholds validated for the intended cohort/application before clinical use.
FIXED_METRICS: tuple[FixedMetric, ...] = (
    FixedMetric(
        key="stroke_fraction",
        name="SF_VTI",
        latex_name=r"$\mathrm{SF}_{\mathrm{VTI}}$",
        threshold=0.48,
        direction=GREATER,
    ),
    FixedMetric(
        key="t50_displacement_timing",
        name="t50_over_T",
        latex_name=r"$t_{50}/T$",
        threshold=0.36,
        direction=LESS,
    ),
    FixedMetric(
        key="low_frequency_spectral_fraction",
        name="E_low_over_E_total",
        latex_name=r"$E_{\mathrm{low}}/E_{\mathrm{total}}$",
        threshold=0.76,
        direction=GREATER,
        numerator_name="E_low",
        denominator_name="E_total",
    ),
    FixedMetric(
        key="late_cycle_mean_fraction",
        name="v_end_over_vbar",
        latex_name=r"$\bar{v}_{\mathrm{end}}/\bar{v}$",
        threshold=0.59,
        direction=LESS,
    ),
    FixedMetric(
        key="effective_duration",
        name="N_eff_over_T",
        latex_name=r"$N_{\mathrm{eff}}/T$",
        threshold=0.90,
        direction=LESS,
    ),
    FixedMetric(
        key="resistivity_index",
        name="RI",
        latex_name=r"$\mathrm{RI}$",
        threshold=0.70,
        direction=GREATER,
    ),
    FixedMetric(
        key="pulsatility_index",
        name="PI",
        latex_name=r"$\mathrm{PI}$",
        threshold=1.30,
        direction=GREATER,
    ),
    FixedMetric(
        key="near_peak_crest_width",
        name="W50_over_T",
        latex_name=r"$W_{50}/T$",
        threshold=0.60,
        direction=LESS,
    ),
)
