import importlib
import inspect
import pkgutil
from typing import List

from .core.base import ProcessPipeline, ProcessResult
from .core.utils import write_combined_results_h5, write_result_h5


def _discover_pipeline_classes():
    """Return all ProcessPipeline subclasses defined in pipeline modules."""
    classes = []
    seen = set()
    for module_info in pkgutil.iter_modules(__path__):
        if module_info.name in {"core"} or module_info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__name__}.{module_info.name}")
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if not issubclass(cls, ProcessPipeline) or cls is ProcessPipeline:
                continue
            if cls.__module__ != module.__name__:
                continue
            if cls in seen:
                continue
            seen.add(cls)
            classes.append(cls)
    classes.sort(key=lambda c: c.__name__.lower())
    return classes


def load_all_pipelines() -> List[ProcessPipeline]:
    """
    Dynamically discover and instantiate all pipeline classes in this package.

    Skips utility modules and any class that requires constructor arguments.
    """
    pipelines: List[ProcessPipeline] = []
    for cls in _discover_pipeline_classes():
        try:
            pipelines.append(cls())
        except TypeError:
            # Skip classes requiring constructor args
            continue
    pipelines.sort(key=lambda p: p.name.lower())
    return pipelines


# Expose pipeline classes at package level for convenience and star-imports.
_PIPELINE_CLASSES = _discover_pipeline_classes()
for _cls in _PIPELINE_CLASSES:
    globals().setdefault(_cls.__name__, _cls)


__all__ = [
    "ProcessPipeline",
    "ProcessResult",
    "write_result_h5",
    "write_combined_results_h5",
    "load_all_pipelines",
    *[_cls.__name__ for _cls in _PIPELINE_CLASSES],
]
