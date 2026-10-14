"""Batch analysis contracts for statistical pipelines over EyeFlow outputs."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias

from dependency_utils import find_missing_dependencies

PIPELINE_REGISTRY: dict[str, type["BatchPipeline"]] = {}
PipelineInputMethod: TypeAlias = Literal[
    "single_file", "file_batch", "cohort_batch", "zip_batch"
]
DEFAULT_INPUT_METHODS: tuple[PipelineInputMethod, ...] = (
    "single_file",
    "file_batch",
    "cohort_batch",
    "zip_batch",
)


def normalize_input_methods(
    input_methods: list[PipelineInputMethod] | None,
) -> list[PipelineInputMethod]:
    if input_methods is None:
        return list(DEFAULT_INPUT_METHODS)
    if not input_methods:
        raise ValueError("input_methods must include at least one supported method.")
    for method in input_methods:
        if method not in DEFAULT_INPUT_METHODS:
            raise ValueError(
                f"Unknown pipeline input method '{method}'. "
                f"Valid: {', '.join(DEFAULT_INPUT_METHODS)}."
            )
    return list(dict.fromkeys(input_methods))


def required_metric_options_for(obj: object) -> tuple[tuple[str, ...], ...]:
    options = getattr(obj, "required_metric_options", None)
    if options:
        return tuple(tuple(option) for option in options if option)
    return tuple((name,) for name in getattr(obj, "required_metrics", ()))


def format_required_metric_options(obj: object) -> str:
    return " and ".join(
        f"({' or '.join(group)})" if len(group) > 1 else group[0]
        for group in required_metric_options_for(obj)
    )


def registerPipeline(
    name: str,
    description: str = "",
    required_deps: list[str] | None = None,
    required_metrics: list[str] | None = None,
    required_metric_options: list[list[str]] | None = None,
    input_methods: list[PipelineInputMethod] | None = None,
    visibility: str = "visible",
    minimum_input_files: int = 1,
    accepted_input_modes: list[str] | None = None,
):
    """Register a class or function that analyzes a batch of metric HDF5 files.

    Required metric groups describe data written by EyeFlow. Each option group
    is an OR choice; distinct groups must all be present in the input file.
    """

    def decorator(target):
        if isinstance(target, type):
            cls = target
        else:
            cls = type(
                "".join(part.capitalize() for part in name.replace("_", " ").split())
                or "FunctionPipeline",
                (FunctionPipeline,),
                {"func": staticmethod(target)},
            )
        cls.name = name
        cls.description = description or getattr(target, "description", "")
        cls.requires = list(required_deps or [])
        cls.missing_deps = find_missing_dependencies(cls.requires)
        cls.available = not cls.missing_deps
        options = (
            required_metric_options
            if required_metric_options is not None
            else [[metric] for metric in required_metrics or []]
        )
        cls.required_metric_options = [
            list(dict.fromkeys(group)) for group in options if group
        ]
        cls.required_metrics = list(
            dict.fromkeys(
                metric for group in cls.required_metric_options for metric in group
            )
        )
        cls.input_methods = normalize_input_methods(input_methods)
        cls.visibility = visibility
        cls.minimum_input_files = max(1, int(minimum_input_files))
        cls.accepted_input_modes = tuple(dict.fromkeys(accepted_input_modes or []))
        PIPELINE_REGISTRY[name] = cls
        return target

    return decorator


@dataclass
class PipelineContext:
    output_dir: Path
    processed_files: tuple[Path, ...]
    selected_pipelines: tuple[str, ...]
    input_path: Path
    zip_outputs: bool
    input_h5_paths: tuple[Path, ...] = ()
    idle_callback: Callable[[], None] | None = None
    record_timing: Callable[[str, float], None] | None = None
    processing_branch: str = "Processing"
    analysis_root: Path | None = None

    @property
    def batch_root(self) -> Path:
        """Root of staged acquisitions used to resolve cohort identities."""
        return self.output_dir if self.analysis_root is None else self.analysis_root

    @property
    def input_files(self) -> tuple[Path, ...]:
        return self.processed_files


@dataclass
class PipelineResult:
    summary: str = ""
    generated_paths: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineDescriptor:
    name: str
    description: str
    available: bool
    requires: list[str] = field(default_factory=list)
    missing_deps: list[str] = field(default_factory=list)
    required_metrics: list[str] = field(default_factory=list)
    required_metric_options: list[list[str]] = field(default_factory=list)
    input_methods: list[PipelineInputMethod] = field(
        default_factory=lambda: list(DEFAULT_INPUT_METHODS)
    )
    pipeline_cls: type["BatchPipeline"] | None = None
    error_msg: str = ""
    visibility: str = "visible"
    minimum_input_files: int = 1
    accepted_input_modes: tuple[str, ...] = ()

    def instantiate(self) -> "BatchPipeline":
        if not self.available or self.pipeline_cls is None:
            return MissingPipeline(
                self.name, self.error_msg or self.description, self.missing_deps
            )
        return self.pipeline_cls()


class BatchPipeline:
    name: str
    description: str
    available: bool
    requires: list[str]
    missing_deps: list[str]
    required_metrics: list[str]
    required_metric_options: list[list[str]]
    input_methods: list[PipelineInputMethod]
    visibility: str = "visible"
    minimum_input_files: int = 1
    accepted_input_modes: tuple[str, ...] = ()

    def __init__(self) -> None:
        if not getattr(self, "name", None):
            self.name = self.__class__.__name__

    def run(self, context: PipelineContext) -> PipelineResult:
        raise NotImplementedError


class FunctionPipeline(BatchPipeline):
    func: Callable[[PipelineContext], PipelineResult]

    def run(self, context: PipelineContext) -> PipelineResult:
        return self.func(context)


class MissingPipeline(BatchPipeline):
    available = False

    def __init__(self, name: str, description: str, missing_deps: list[str]) -> None:
        self.name = name
        self.description = description
        self.missing_deps = missing_deps

    def run(self, context: PipelineContext) -> PipelineResult:
        reason = ", ".join(self.missing_deps) or self.description
        raise RuntimeError(f"Pipeline '{self.name}' unavailable ({reason}).")
