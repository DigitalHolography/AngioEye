from __future__ import annotations

import json

from .core.base import (
    BatchPostprocess,
    PostprocessContext,
    PostprocessResult,
    registerPostprocess,
)


@registerPostprocess(
    name="Postprocess Tutorial",
    description=(
        "Minimal tutorial showing the available PostprocessContext fields and "
        "the PostprocessResult output format."
    ),
)
class PostprocessTutorial(BatchPostprocess):
    """
    Minimal postprocess example.

    It does not inspect HDF5 contents. It only shows:
    - which fields are available on `context`
    - which fields are returned through `PostprocessResult`
    """

    def run(self, context: PostprocessContext) -> PostprocessResult:
        tutorial_path = context.output_dir / "postprocess_tutorial.json"

        result = PostprocessResult(
            summary="Generated postprocess_tutorial.json.",
            generated_paths=[str(tutorial_path)],
            metadata={
                "processed_file_count": len(context.processed_files),
                "selected_pipelines": list(context.selected_pipelines),
            },
        )

        payload = {
            "postprocess_name": self.name,
            "context_fields": {
                "output_dir": str(context.output_dir),
                "processed_files": [str(path) for path in context.processed_files],
                "input_h5_paths": [str(path) for path in context.input_h5_paths],
                "selected_pipelines": list(context.selected_pipelines),
                "input_path": str(context.input_path),
                "zip_outputs": context.zip_outputs,
            },
            "result_format": {
                "summary": result.summary,
                "generated_paths": result.generated_paths,
                "metadata": result.metadata,
            },
        }

        tutorial_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return result
