"""
Dummy pipeline to demonstrate optional dependencies handling.

This pipeline pretends to need `torch` and `pandas`. It won't run unless those
packages are installed, but it should appear in the UI as disabled with a tooltip
showing the missing deps.
"""

from .core.base import ProcessPipeline, ProcessResult

REQUIRES = ["torch>=2.2", "pandas>=2.1"]


class DummyHeavy(ProcessPipeline):
    description = "Demo pipeline that requires torch+pandas; computes a trivial metric."

    def run(self, _h5file) -> ProcessResult:
        import pandas as pd  # noqa: F401
        import torch  # noqa: F401

        return ProcessResult(metrics={"dummy": 1})
