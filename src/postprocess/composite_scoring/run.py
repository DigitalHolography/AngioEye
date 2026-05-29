from postprocess.core.base import PostprocessContext, PostprocessResult
from postprocess.core.grouped_batch import extract_group_name

from .dataclasses import ScoreRecord
from .plots import write_score_plots
from .scoring import append_scores_to_file, score_records_for_tree

def run_composite_scoring(context: PostprocessContext) -> PostprocessResult:
    updated_paths: list[str] = []
    score_records: list[ScoreRecord] = []
    failures: list[str] = []
    for file_path in context.processed_files:
        try:
            tree = append_scores_to_file(file_path)
        except Exception as exc:  # noqa: BLE001
            failures.append(
                f"Composite Scoring skipped {file_path}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        cohort = extract_group_name(file_path.parent, context.output_dir)
        score_records.extend(
            score_records_for_tree(
                tree,
                cohort=cohort,
                file_path=file_path,
            )
        )
        updated_paths.append(str(file_path))

    png_paths = write_score_plots(score_records, context.output_dir)
    return PostprocessResult(
        summary=(
            f"Appended Composite Scoring to {len(updated_paths)} processed HDF5 "
            f"file(s). Generated {len(png_paths)} PNG plot(s). "
            f"Skipped {len(failures)} file(s)."
        ),
        generated_paths=[*updated_paths, *png_paths],
        metadata={"failures": failures},
    )