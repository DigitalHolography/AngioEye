import numpy as np

from .core.base import ProcessPipeline, ProcessResult, registerPipeline, with_attrs


@registerPipeline(name="Static Example")
class StaticExample(ProcessPipeline):
    """
    Tutorial pipeline showing the full surface area of a pipeline:

    - Subclass ProcessPipeline and implement `run(self, h5file) -> ProcessResult`.
    - Return metrics (scalars, vectors, matrices, cubes) and optional artifacts.
    - Use "/" inside metric/artifact keys to create nested groups in output HDF5.
    - Attach HDF5 attributes to any metric via `with_attrs(data, attrs_dict)`.
    - Add attributes to the pipeline group (`attrs`) or root file (`file_attrs`).
    - No input data is required; this pipeline is purely illustrative.
    """

    description = "Tutorial: metrics/artifacts + nested output groups + attrs."

    def run(self, h5file) -> ProcessResult:
        # Each key becomes a dataset under /Pipelines/<name>/metrics.
        # Keys with "/" automatically create sub-groups (e.g. artery/tauH_10).
        metrics = {
            "scalar_example": 42.0,
            "vector_example": [1.0, 2.0, 3.0],
            "subfolder/metrics_1": 2.5,
            "subfolder/metrics_2": 3.1,
            # Attach dataset-level attributes (min/max/name/unit) using with_attrs.
            "aggregates/matrix_example": with_attrs(
                [[1, 2], [3, 4]],
                {
                    "minimum": [1],
                    "maximum": [4],
                    "nameID": ["matrix_example"],
                    "unit": ["a.u."],
                },
            ),
            "cube_example": with_attrs(
                np.arange(8).reshape(2, 2, 2),
                {"minimum": [0], "maximum": [7], "original_class": "int", "unit": [""]},
            ),
        }

        # Artifacts support the same nested-key behavior.
        artifacts = {
            "note": "Static data for demonstration",
            "logs/run_id": "static_demo_001",
            "logs/message": "Nested artifact example",
        }

        # Optional attributes applied to the pipeline group and the root file.
        attrs = {"pipeline_version": "1.0", "author": "StaticExample"}
        file_attrs = {"example_generated": True}

        return ProcessResult(
            metrics=metrics, artifacts=artifacts, attrs=attrs, file_attrs=file_attrs
        )
