import numpy as np

from .core.base import ProcessPipeline, ProcessResult, with_attrs


class StaticExample(ProcessPipeline):
    """
    Tutorial pipeline showing the full surface area of a pipeline:

    - Subclass ProcessPipeline and implement `run(self, h5file) -> ProcessResult`.
    - Return metrics (scalars, vectors, matrices, cubes) and optional artifacts.
    - Attach HDF5 attributes to any metric via `with_attrs(data, attrs_dict)`.
    - Add attributes to the pipeline group (`attrs`) or root file (`file_attrs`).
    - No input data is required; this pipeline is purely illustrative.
    """

    description = "Tutorial: metrics + artifacts + dataset attrs + file/pipeline attrs."

    def run(self, _h5file) -> ProcessResult:
        # Metrics are the main numerical outputs; each key becomes a dataset under /pipelines/<name>/metrics.
        metrics = {
            "scalar_example": 42.0,
            "vector_example": [1.0, 2.0, 3.0],
            # Attach dataset-level attributes (min/max/name/unit) using with_attrs.
            "matrix_example": with_attrs(
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

        # Artifacts can store non-metric outputs (strings, paths, etc.).
        artifacts = {"note": "Static data for demonstration"}

        # Optional attributes applied to the pipeline group and the root file.
        attrs = {"pipeline_version": "1.0", "author": "StaticExample"}
        file_attrs = {"example_generated": True}

        return ProcessResult(
            metrics=metrics, artifacts=artifacts, attrs=attrs, file_attrs=file_attrs
        )
