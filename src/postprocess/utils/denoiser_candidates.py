"""
The five denoiser candidates screened throughout the flicker-detection
project (synthetic ground-truth benchmark in test/denoising_benchmark.py,
real-data separability check in test/flicker_denoising_separability.py):
the untouched production six-step chain, plus the three alternative
strategies added in waveform_shape_metrics_denoised_alternatives.py
(graph-Laplacian neighbor-smoothing, low-rank SVD, and two Kalman-filter
axis variants).

Each accessor takes an ArterialSegExample instance (from
waveform_shape_metrics_denoised_alternatives.py) and returns the bound
method to call with a v_block. Laplacian/lowrank/Kalman live on nested
denoiser objects (pipeline.laplacian / .lowrank / .kalman) rather than
directly on the pipeline. Promoted from test/denoising_benchmark.py's
module-level CANDIDATES list to a shared location so both benchmark/
separability scripts and any real analysis code import the same single
source of truth rather than one script importing it from another.
"""

from __future__ import annotations

CANDIDATES = [
    ("no_denoising (baseline)", None),
    ("six_step_chain", lambda p: p._denoise_segment_block),
    ("graph_laplacian", lambda p: p.laplacian._denoise_segment_block),
    ("lowrank_svd", lambda p: p.lowrank._denoise_segment_block),
    ("kalman_beat_axis", lambda p: p.kalman._denoise_segment_block_beat_axis),
    ("kalman_radius_axis", lambda p: p.kalman._denoise_segment_block_radius_axis),
]
