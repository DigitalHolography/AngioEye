import csv
from dataclasses import dataclass, field
from typing import Any

import h5py

from dependency_utils import find_missing_dependencies

# Global Registry of all imports needed by the pipelines
PIPELINE_REGISTRY: dict[str, type["ProcessPipeline"]] = {}


# Decorator to register all neede pipelines
def registerPipeline(
    name: str, description: str = "", required_deps: list[str] | None = None
):
    def decorator(cls):
        # metadata for the class
        cls.name = name
        cls.description = description or getattr(cls, "description", "")
        cls.requires = required_deps or []

        missing = find_missing_dependencies(cls.requires)
        cls.missing_deps = missing
        cls.available = len(missing) == 0

        # Add to registry
        PIPELINE_REGISTRY[name] = cls
        return cls

    return decorator


@dataclass
class ProcessResult:
    metrics: dict[str, Any]
    attrs: dict[str, Any] | None = None  # attributes stored on the pipeline group
    output_h5_path: str | None = None


@dataclass
class DatasetValue:
    """Represents a dataset payload plus optional attributes for that dataset."""

    data: Any
    attrs: dict[str, Any] | None = None


def with_attrs(data: Any, attrs: dict[str, Any]) -> DatasetValue:
    """Convenience helper to attach attributes to a dataset value."""
    return DatasetValue(data=data, attrs=attrs)


# +==========================================================================+ #
# |                            PIPELINES CLASSES                             | #
# +==========================================================================+ #


@dataclass
class PipelineDescriptor:
    name: str
    description: str
    available: bool
    # To avoid Python Mutable Default Arguments
    requires: list[str] = field(default_factory=list)
    missing_deps: list[str] = field(default_factory=list)
    pipeline_cls: type["ProcessPipeline"] | None = None
    error_msg: str = ""

    def instantiate(self) -> "ProcessPipeline":
        """Factory method to create the actual pipeline instance."""
        if not self.available or self.pipeline_cls is None:
            return MissingPipeline(
                self.name,
                self.error_msg or self.description,
                self.missing_deps,
                self.requires,
            )
        return self.pipeline_cls()


class ProcessPipeline:
    name: str
    description: str
    available: bool
    missing_deps: list[str]
    requires: list[str]

    def __init__(self) -> None:
        # Derive the pipeline name from the module filename (e.g., basic_stats.py -> basic_stats).
        if not getattr(self, "name", None):
            module_name = (self.__class__.__module__ or "").rsplit(".", 1)[-1]
            self.name: str = module_name or self.__class__.__name__

    def run(self, h5file: h5py.File) -> ProcessResult:
        raise NotImplementedError

    def export(self, result: ProcessResult, output_path: str) -> str:
        """Default CSV export for metrics."""
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["metric", "value"])
            for key, value in result.metrics.items():
                writer.writerow([key, value])
        return output_path


class MissingPipeline(ProcessPipeline):
    """Placeholder for pipelines whose dependencies are missing."""

    available = False

    def __init__(
        self, name: str, description: str, missing_deps: list[str], requires: list[str]
    ) -> None:
        # super().__init__()
        self.name = name
        self.description = description or "Pipeline unavailable (missing dependencies)."
        self.missing_deps = missing_deps
        self.requires = requires

    def run(self, h5file):
        missing = ", ".join(
            self.missing_deps or self.requires or ["unknown dependency"]
        )
        raise ImportError(
            f"Pipeline '{self.name}' unavailable. Missing dependencies: {missing}"
        )
