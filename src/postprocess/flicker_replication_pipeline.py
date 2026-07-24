from __future__ import annotations

from .core.base import (
    BatchPostprocess,
    PostprocessContext,
    PostprocessResult,
    registerPostprocess,
)


@registerPostprocess(
    name="Flicker confound-controlled replication",
    description=(
        "Confound-controlled baseline-vs-flicker replication report (total "
        "pulsatile RMS, rho1/rho2/rho3) from low-rank pulsatility metrics, "
        "grouped by participant/eye and epoch."
    ),
    required_deps=["scipy>=1.10"],
    required_pipelines=["lowrank_pulsatility_metrics"],
    visibility="hidden",
)
class FlickerReplicationPostprocess(BatchPostprocess):
    def run(self, context: PostprocessContext) -> PostprocessResult:
        if not context.processed_files:
            raise ValueError(
                "No processed HDF5 outputs are available for postprocessing."
            )

        output_dir = context.output_dir.expanduser().resolve()
        if not output_dir.exists() or not output_dir.is_dir():
            raise FileNotFoundError(f"Output folder does not exist: {output_dir}")

        from .utils import flicker_replication

        report = flicker_replication.build_replication_report(output_dir)
        if not report:
            raise ValueError(
                "No compatible lowrank_pulsatility_metrics output with a "
                "recognized baseline1/flicker/baseline2 epoch layout was "
                "found for the flicker replication report."
            )

        report_text = flicker_replication.format_report_text(report)
        report_path = output_dir / "flicker_replication_report.txt"
        report_path.write_text(report_text)

        n_prefixes = len(report)
        n_retained = sum(
            1
            for metrics in report.values()
            for battery in metrics.values()
            if (battery.get("vb_report") or {}).get("retained")
        )
        summary = (
            f"Flicker replication report: {n_prefixes} participant/eye group(s), "
            f"{n_retained} (metric, group) pair(s) retained under Sec. V.B "
            "replication."
        )
        return PostprocessResult(
            summary=summary, generated_paths=[str(report_path)]
        )
