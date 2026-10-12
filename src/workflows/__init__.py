"""Run statistical pipelines over batches of exported EyeFlow metrics."""

from .batch import BatchRunResult, run_batch

__all__ = ["BatchRunResult", "run_batch"]
