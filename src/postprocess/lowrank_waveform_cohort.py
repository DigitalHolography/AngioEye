"""GUI postprocess: cohort low-rank figures and Table I from a ZIP/folder."""

from __future__ import annotations

from pathlib import Path

from .core.base import (
    BatchPostprocess,
    PostprocessContext,
    PostprocessResult,
    registerPostprocess,
)


@registerPostprocess(
    name="Low-rank waveform cohort figures",
    description=(
        "From a cohort ZIP or folder (group subfolders such as "
        "baseline1/flicker/baseline2), compute arterial low-rank endpoints, "
        "write per-acquisition Figs. 2--4, cohort Figs. 5--7 when 2+ groups "
        "are present, and Table I / confound CSVs for the flicker triad."
    ),
    required_deps=["numpy>=1.24", "pandas>=2.1", "scipy>=1.10", "matplotlib>=3.7", "h5py>=3.8"],
)
class LowRankWaveformCohortPostprocess(BatchPostprocess):
    def run(self, context: PostprocessContext) -> PostprocessResult:
        from pipelines.lowrank_waveform_decomposition import run as cohort_run

        input_path = Path(context.input_path).expanduser()
        output_dir = Path(context.output_dir).expanduser()
        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Cohort figures recompute from source acquisitions (ZIP/folder layout),
        # not from pipeline result H5s. veins=False: arterial figures only.
        summary, generated_paths = cohort_run(
            input_path,
            output_dir,
            veins=False,
        )
        return PostprocessResult(
            summary=summary,
            generated_paths=[str(path) for path in generated_paths],
            metadata={
                "input_path": str(input_path),
                "output_dir": str(output_dir),
                "n_generated": len(generated_paths),
            },
        )
