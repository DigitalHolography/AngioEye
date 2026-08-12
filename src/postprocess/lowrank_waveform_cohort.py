"""GUI postprocess: cohort low-rank figures and Table I from packed AE H5s."""

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
        "From packed AngioEye result H5s (`{stem}_AE.h5` under each "
        "acquisition's `{stem}_AE/` folder, typically after a pipeline run "
        "with lowrank_waveform_decomposition), build cohort Figs. 5--7 when "
        "2+ group folders are present, plus Table I / confound CSVs for the "
        "flicker triad. Per-acquisition Figs 2--4 and the AE H5 itself are "
        "produced by the pipeline, not this postprocess."
    ),
    required_deps=[
        "numpy>=1.24",
        "pandas>=2.1",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "h5py>=3.8",
    ],
    required_pipelines=["lowrank_waveform_decomposition"],
)
class LowRankWaveformCohortPostprocess(BatchPostprocess):
    def run(self, context: PostprocessContext) -> PostprocessResult:
        from pipelines.lowrank_waveform_decomposition import run as cohort_run

        input_path = Path(context.input_path).expanduser()
        output_dir = Path(context.output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prefer AE result H5s from the just-finished pipeline run; otherwise
        # discover packed low-rank results under the input cohort tree.
        result_paths = tuple(context.processed_files) or None
        if result_paths is None and not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        # Cohort figures aggregate from packed AE H5s. veins=False: arterial
        # figures only.
        summary, generated_paths = cohort_run(
            input_path if input_path.exists() else output_dir,
            output_dir,
            veins=False,
            result_h5_paths=result_paths,
        )
        return PostprocessResult(
            summary=summary,
            generated_paths=[str(path) for path in generated_paths],
            metadata={
                "input_path": str(input_path),
                "output_dir": str(output_dir),
                "n_generated": len(generated_paths),
                "n_result_h5": len(result_paths) if result_paths else 0,
            },
        )
