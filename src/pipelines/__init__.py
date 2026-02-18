import importlib
import pkgutil

# import inspect
from .core.base import (
    PIPELINE_REGISTRY,
    MissingPipeline,
    PipelineDescriptor,
    ProcessPipeline,
    ProcessResult,
)
from .core.utils import write_combined_results_h5, write_result_h5


def _discover_pipelines() -> tuple[list[PipelineDescriptor], list[PipelineDescriptor]]:
    available: list[PipelineDescriptor] = []
    missing: list[PipelineDescriptor] = []

    for module_info in pkgutil.iter_modules(__path__):
        if module_info.name in {"core"} or module_info.name.startswith("_"):
            continue

        module_name = f"{__name__}.{module_info.name}"

        try:
            importlib.import_module(module_name)
        except Exception as e:
            # Fallback for unknown failures (SyntaxError, etc.)
            missing.append(
                PipelineDescriptor(
                    name=module_info.name,
                    description=f"Import Error: {e}",
                    available=False,
                    error_msg=str(e),
                )
            )

    for _name, cls in PIPELINE_REGISTRY.items():
        desc = PipelineDescriptor(
            name=cls.name,
            description=cls.description,
            available=cls.available,
            requires=cls.requires,
            missing_deps=cls.missing_deps,
            pipeline_cls=cls,
        )
        if getattr(cls, "is_available", True):
            available.append(desc)
        else:
            missing.append(desc)

    available.sort(key=lambda p: p.name.lower())
    missing.sort(key=lambda p: p.name.lower())
    return available, missing


def load_pipeline_catalog() -> tuple[
    list[PipelineDescriptor], list[PipelineDescriptor]
]:
    """Return (available, missing) pipelines for UI/CLI surfaces."""
    return _discover_pipelines()


# Expose pipeline classes at package level for convenience and star-imports.
_AVAILABLE, _MISSING = _discover_pipelines()
for _cls in (p.__class__ for p in _AVAILABLE):
    globals().setdefault(_cls.__name__, _cls)


__all__ = [
    "ProcessPipeline",
    "ProcessResult",
    "write_result_h5",
    "write_combined_results_h5",
    "load_pipeline_catalog",
    "MissingPipeline",
    *[_cls.__name__ for _cls in (p.__class__ for p in _AVAILABLE)],
]
