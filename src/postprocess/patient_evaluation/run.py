from __future__ import annotations

from pathlib import Path
from typing import Iterable

from postprocess.core.base import PostprocessContext, PostprocessResult

from .dataclasses import (
    EvaluationFailure,
    EvaluationRunResult,
    PatientEvaluation,
)
from .inputs import default_output_dir, materialize_h5_inputs
from .metrics import (
    DEFAULT_AGGREGATION,
    DEFAULT_REPRESENTATION,
    DEFAULT_VESSEL_TYPE,
    FIXED_METRICS,
)
from .plots import write_evaluation_plots
from .reports import write_evaluation_reports
from .scoring import evaluate_h5_patient


WRITE_PLOTS = True


def evaluate_selected_path(
    selected_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    vessel_type: str = DEFAULT_VESSEL_TYPE,
    representation: str = DEFAULT_REPRESENTATION,
    aggregation: str = DEFAULT_AGGREGATION,
) -> EvaluationRunResult:
    """Evaluate one selected H5, ZIP, or directory.

    When output_dir is omitted, the output directory is created next to the
    selected input.
    """
    selected = Path(selected_path)
    resolved_output = (
        Path(output_dir)
        if output_dir is not None
        else default_output_dir(selected)
    )
    return evaluate_input_paths(
        [selected],
        output_dir=resolved_output,
        vessel_type=vessel_type,
        representation=representation,
        aggregation=aggregation,
    )


def evaluate_input_paths(
    input_paths: Iterable[str | Path],
    *,
    output_dir: str | Path,
    vessel_type: str = DEFAULT_VESSEL_TYPE,
    representation: str = DEFAULT_REPRESENTATION,
    aggregation: str = DEFAULT_AGGREGATION,
) -> EvaluationRunResult:
    """Evaluate every H5 patient found in the supplied inputs."""
    paths = [Path(path) for path in input_paths]
    output = Path(output_dir)
    evaluations: list[PatientEvaluation] = []
    failures: list[EvaluationFailure] = []

    with materialize_h5_inputs(paths) as resolved_files:
        for resolved in resolved_files:
            try:
                evaluation = evaluate_h5_patient(
                    resolved.h5_path,
                    patient_id=resolved.patient_id,
                    source_file=str(resolved.source_path),
                    archive_member=resolved.archive_member,
                    metric_panel=FIXED_METRICS,
                    vessel_type=vessel_type,
                    representation=representation,
                    aggregation=aggregation,
                )
                evaluations.append(evaluation)
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    EvaluationFailure(
                        source_file=str(resolved.source_path),
                        archive_member=resolved.archive_member,
                        patient_id=resolved.patient_id,
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )

    generated_paths = write_evaluation_reports(
        evaluations,
        failures,
        output,
    )
    if WRITE_PLOTS:
        generated_paths.extend(write_evaluation_plots(evaluations, output))

    return EvaluationRunResult(
        evaluations=tuple(evaluations),
        failures=tuple(failures),
        generated_paths=tuple(generated_paths),
        output_dir=output,
    )


def run_fixed_threshold_evaluation(
    context: PostprocessContext,
) -> PostprocessResult:
    """Adapter matching the architecture of the existing postprocess pipeline.

    The application normally gives postprocesses one processed H5 per patient in
    context.processed_files. If one of those paths is a ZIP, it is expanded and
    every contained H5 is evaluated independently.
    """
    output_dir = Path(context.output_dir) / "fixed_threshold_evaluation"
    try:
        result = evaluate_input_paths(
            context.processed_files,
            output_dir=output_dir,
        )
    except Exception as exc:  # noqa: BLE001
        return PostprocessResult(
            summary=(
                "Fixed-threshold patient evaluation failed: "
                f"{type(exc).__name__}: {exc}"
            ),
            generated_paths=[],
            metadata={"failures": [str(exc)]},
        )

    return PostprocessResult(
        summary=(
            f"Evaluated {len(result.evaluations)} patient HDF5 file(s) with "
            f"{len(FIXED_METRICS)} fixed metric(s). "
            f"Skipped {len(result.failures)} patient file(s). "
            f"Generated {len(result.generated_paths)} output file(s)."
        ),
        generated_paths=list(result.generated_paths),
        metadata={
            "failures": [
                {
                    "patient_id": item.patient_id,
                    "error_type": item.error_type,
                    "message": item.message,
                }
                for item in result.failures
            ],
            "n_evaluated_patients": len(result.evaluations),
            "vessel_type": DEFAULT_VESSEL_TYPE,
            "representation": DEFAULT_REPRESENTATION,
        },
    )
