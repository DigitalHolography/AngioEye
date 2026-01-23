import numpy as np

from .core.base import ProcessPipeline, ProcessResult


class StaticMatrixExamplePipeline(ProcessPipeline):
    """
    Minimal pipeline demonstrating scalar, vector, matrix, and 3D array outputs.
    Useful for validating HDF5 writing of non-scalar results.
    """

    name = "Static matrix example"
    description = "Outputs fixed scalar, vector, matrix, and cube results."

    def run(self, _h5file) -> ProcessResult:
        metrics = {
            "scalar_example": 42.0,
            "vector_example": [1.0, 2.0, 3.0],
            "matrix_example": [[1, 2], [3, 4]],
            "cube_example": np.arange(8).reshape(2, 2, 2),
        }
        artifacts = {
            "note": "Static data for demonstration",
        }
        return ProcessResult(metrics=metrics, artifacts=artifacts)
