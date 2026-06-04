from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from dependency_utils import find_missing_dependencies

POSTPROCESS_REGISTRY: dict[str, type["BatchPostprocess"]] = {}


def registerPostprocess(
    name: str,
    description: str = "",
    required_deps: list[str] | None = None,
    required_pipelines: list[str] | None = None,
    required_pipeline_options: list[list[str]] | None = None,
):
    def decorator(target):
        requires = required_deps or []
        pipeline_options = normalize_required_pipeline_options(
            required_pipelines=required_pipelines,
            required_pipeline_options=required_pipeline_options,
        )
        pipelines = flatten_required_pipeline_options(pipeline_options)
        missing = find_missing_dependencies(requires)

        if isinstance(target, type):
            target.name = name
            target.description = description or getattr(target, "description", "")
            target.requires = requires
            target.required_pipelines = pipelines
            target.required_pipeline_options = pipeline_options
            target.missing_deps = missing
            target.available = len(missing) == 0
            POSTPROCESS_REGISTRY[name] = target
            return target

        postprocess_cls = _build_function_postprocess(
            name=name,
            description=description or getattr(target, "__doc__", "") or "",
            func=target,
            requires=requires,
            missing_deps=missing,
            required_pipelines=pipelines,
            required_pipeline_options=pipeline_options,
        )
        POSTPROCESS_REGISTRY[name] = postprocess_cls
        return target

    return decorator


def normalize_required_pipeline_options(
    *,
    required_pipelines: list[str] | None = None,
    required_pipeline_options: list[list[str]] | None = None,
) -> list[list[str]]:
    if required_pipeline_options is not None:
        return [
            list(dict.fromkeys(option))
            for option in required_pipeline_options
            if option
        ]
    if required_pipelines:
        return [list(dict.fromkeys(required_pipelines))]
    return []


def flatten_required_pipeline_options(
    required_pipeline_options: list[list[str]],
) -> list[str]:
    flattened: list[str] = []
    for option in required_pipeline_options:
        for pipeline_name in option:
            if pipeline_name not in flattened:
                flattened.append(pipeline_name)
    return flattened


def required_pipeline_options_for(obj: object) -> tuple[tuple[str, ...], ...]:
    options = getattr(obj, "required_pipeline_options", None)
    if options:
        return tuple(tuple(option) for option in options if option)
    required = tuple(getattr(obj, "required_pipelines", ()))
    return (required,) if required else ()


def format_required_pipeline_options(obj: object) -> str:
    groups = []
    for option in required_pipeline_options_for(obj):
        groups.append(" + ".join(option))
    return " or ".join(groups)


def _build_function_postprocess(
    *,
    name: str,
    description: str,
    func: Callable[["PostprocessContext"], "PostprocessResult"],
    requires: list[str],
    missing_deps: list[str],
    required_pipelines: list[str],
    required_pipeline_options: list[list[str]],
) -> type["FunctionPostprocess"]:
    class RegisteredFunctionPostprocess(FunctionPostprocess):
        pass

    RegisteredFunctionPostprocess.__name__ = "".join(
        part.capitalize() for part in name.replace("_", " ").split()
    ) or "FunctionPostprocess"
    RegisteredFunctionPostprocess.name = name
    RegisteredFunctionPostprocess.description = description
    RegisteredFunctionPostprocess.func = staticmethod(func)
    RegisteredFunctionPostprocess.requires = requires
    RegisteredFunctionPostprocess.missing_deps = missing_deps
    RegisteredFunctionPostprocess.available = len(missing_deps) == 0
    RegisteredFunctionPostprocess.required_pipelines = required_pipelines
    RegisteredFunctionPostprocess.required_pipeline_options = required_pipeline_options
    RegisteredFunctionPostprocess.missing_pipelines = []
    return RegisteredFunctionPostprocess


@dataclass
class PostprocessContext:
    output_dir: Path
    processed_files: tuple[Path, ...]
    selected_pipelines: tuple[str, ...]
    input_path: Path
    zip_outputs: bool
    input_h5_paths: tuple[Path, ...] = ()
    idle_callback: Callable[[], None] | None = None


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
    required_pipeline_options: list[list[str]] = field(default_factory=list)
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
                required_pipeline_options=self.required_pipeline_options,
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
    required_pipeline_options: list[list[str]]
    missing_pipelines: list[str]

    def __init__(self) -> None:
        if not getattr(self, "name", None):
            module_name = (self.__class__.__module__ or "").rsplit(".", 1)[-1]
            self.name = module_name or self.__class__.__name__

    def run(self, context: PostprocessContext) -> PostprocessResult:
        raise NotImplementedError


class FunctionPostprocess(BatchPostprocess):
    func: Callable[[PostprocessContext], PostprocessResult]

    def run(self, context: PostprocessContext) -> PostprocessResult:
        return self.func(context)


class MissingPostprocess(BatchPostprocess):
    available = False

    def __init__(
        self,
        name: str,
        description: str,
        missing_deps: list[str],
        required_pipelines: list[str],
        required_pipeline_options: list[list[str]],
        missing_pipelines: list[str],
    ) -> None:
        self.name = name
        self.description = description or "Postprocess unavailable."
        self.missing_deps = missing_deps
        self.requires = missing_deps
        self.required_pipelines = required_pipelines
        self.required_pipeline_options = required_pipeline_options
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
                "required pipelines not selected: "
                f"{format_required_pipeline_options(self)}"
            )
        reason = "; ".join(parts) if parts else "unknown reason"
        raise RuntimeError(f"Postprocess '{self.name}' unavailable ({reason}).")
