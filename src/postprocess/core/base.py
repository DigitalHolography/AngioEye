from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dependency_utils import find_missing_dependencies

POSTPROCESS_REGISTRY: dict[str, type["BatchPostprocess"]] = {}


def registerPostprocess(
    name: str,
    description: str = "",
    required_deps: list[str] | None = None,
    required_pipelines: list[str] | None = None,
):
    def decorator(cls):
        cls.name = name
        cls.description = description or getattr(cls, "description", "")
        cls.requires = required_deps or []
        cls.required_pipelines = required_pipelines or []

        missing = find_missing_dependencies(cls.requires)
        cls.missing_deps = missing
        cls.available = len(missing) == 0
        POSTPROCESS_REGISTRY[name] = cls
        return cls

    return decorator


@dataclass
class PostprocessContext:
    output_dir: Path
    processed_files: tuple[Path, ...]
    selected_pipelines: tuple[str, ...]
    input_path: Path
    zip_outputs: bool
    input_h5_paths: tuple[Path, ...] = ()


@dataclass
class PostprocessResult:
    summary: str = ""
    generated_paths: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PostprocessDescriptor:
    name: str
    description: str
    available: bool
    requires: list[str] = field(default_factory=list)
    missing_deps: list[str] = field(default_factory=list)
    required_pipelines: list[str] = field(default_factory=list)
    missing_pipelines: list[str] = field(default_factory=list)
    postprocess_cls: type["BatchPostprocess"] | None = None
    error_msg: str = ""

    def instantiate(self) -> "BatchPostprocess":
        if not self.available or self.postprocess_cls is None:
            return MissingPostprocess(
                name=self.name,
                description=self.error_msg or self.description,
                missing_deps=self.missing_deps,
                required_pipelines=self.required_pipelines,
                missing_pipelines=self.missing_pipelines,
            )
        return self.postprocess_cls()


class BatchPostprocess:
    name: str
    description: str
    available: bool
    missing_deps: list[str]
    requires: list[str]
    required_pipelines: list[str]
    missing_pipelines: list[str]

    def __init__(self) -> None:
        if not getattr(self, "name", None):
            module_name = (self.__class__.__module__ or "").rsplit(".", 1)[-1]
            self.name = module_name or self.__class__.__name__

    def run(self, context: PostprocessContext) -> PostprocessResult:
        raise NotImplementedError


class MissingPostprocess(BatchPostprocess):
    available = False

    def __init__(
        self,
        name: str,
        description: str,
        missing_deps: list[str],
        required_pipelines: list[str],
        missing_pipelines: list[str],
    ) -> None:
        self.name = name
        self.description = description or "Postprocess unavailable."
        self.missing_deps = missing_deps
        self.requires = missing_deps
        self.required_pipelines = required_pipelines
        self.missing_pipelines = missing_pipelines

    def run(self, context: PostprocessContext) -> PostprocessResult:
        parts: list[str] = []
        if self.missing_deps:
            parts.append(f"missing dependencies: {', '.join(self.missing_deps)}")
        if self.missing_pipelines:
            parts.append(
                f"missing required pipelines: {', '.join(self.missing_pipelines)}"
            )
        if not parts and self.required_pipelines:
            parts.append(
                f"required pipelines not selected: {', '.join(self.required_pipelines)}"
            )
        reason = "; ".join(parts) if parts else "unknown reason"
        raise RuntimeError(f"Postprocess '{self.name}' unavailable ({reason}).")
