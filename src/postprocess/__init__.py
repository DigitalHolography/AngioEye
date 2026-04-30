import importlib
import os
import pkgutil
import sys
from pathlib import Path

from pipelines import load_pipeline_catalog

from .core.base import (
    POSTPROCESS_REGISTRY,
    BatchPostprocess,
    MissingPostprocess,
    PostprocessContext,
    PostprocessDescriptor,
    PostprocessResult,
    registerPostprocess,
)


def _extend_with_external_postprocess_dir() -> None:
    candidates: list[Path] = []
    env_path = os.getenv("ANGIOEYE_POSTPROCESS_DIR")
    if env_path:
        candidates.append(Path(env_path))
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent / "postprocess")

    for candidate in reversed(candidates):
        if candidate.is_dir():
            path_value = str(candidate.resolve())
            if path_value not in __path__:
                __path__.insert(0, path_value)


_extend_with_external_postprocess_dir()


def _discover_postprocesses() -> tuple[
    list[PostprocessDescriptor], list[PostprocessDescriptor]
]:
    available: list[PostprocessDescriptor] = []
    missing: list[PostprocessDescriptor] = []
    POSTPROCESS_REGISTRY.clear()
    importlib.invalidate_caches()

    pipeline_catalog, _ = load_pipeline_catalog()
    available_pipeline_names = {pipeline.name for pipeline in pipeline_catalog}

    for module_info in pkgutil.iter_modules(__path__):
        if module_info.name in {"core"} or module_info.name.startswith("_"):
            continue

        module_name = f"{__name__}.{module_info.name}"

        try:
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
            else:
                importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            missing.append(
                PostprocessDescriptor(
                    name=module_info.name,
                    description=f"Import Error: {exc}",
                    available=False,
                    error_msg=str(exc),
                )
            )

    for _name, cls in POSTPROCESS_REGISTRY.items():
        missing_pipelines = sorted(
            {
                pipeline_name
                for pipeline_name in getattr(cls, "required_pipelines", [])
                if pipeline_name not in available_pipeline_names
            }
        )
        is_available = getattr(cls, "available", True) and not missing_pipelines
        desc = PostprocessDescriptor(
            name=cls.name,
            description=cls.description,
            available=is_available,
            requires=cls.requires,
            missing_deps=cls.missing_deps,
            required_pipelines=getattr(cls, "required_pipelines", []),
            missing_pipelines=missing_pipelines,
            postprocess_cls=cls,
            error_msg=(
                ""
                if is_available
                else (
                    f"Missing required pipelines: {', '.join(missing_pipelines)}"
                    if missing_pipelines
                    else ""
                )
            ),
        )
        if is_available:
            available.append(desc)
        else:
            missing.append(desc)

    available.sort(key=lambda postprocess: postprocess.name.lower())
    missing.sort(key=lambda postprocess: postprocess.name.lower())
    return available, missing


def load_postprocess_catalog() -> tuple[
    list[PostprocessDescriptor], list[PostprocessDescriptor]
]:
    return _discover_postprocesses()


_AVAILABLE, _MISSING = _discover_postprocesses()
_EXPORTED_POSTPROCESS_CLASSES: dict[str, type[BatchPostprocess]] = {}
for _descriptor in _AVAILABLE:
    if _descriptor.postprocess_cls is None:
        continue
    _EXPORTED_POSTPROCESS_CLASSES.setdefault(
        _descriptor.postprocess_cls.__name__,
        _descriptor.postprocess_cls,
    )

for _name, _cls in _EXPORTED_POSTPROCESS_CLASSES.items():
    globals().setdefault(_name, _cls)


__all__ = [
    "BatchPostprocess",
    "MissingPostprocess",
    "PostprocessContext",
    "PostprocessDescriptor",
    "PostprocessResult",
    "registerPostprocess",
    "load_postprocess_catalog",
    *_EXPORTED_POSTPROCESS_CLASSES.keys(),
]
