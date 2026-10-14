"""Discover statistical batch pipelines for exported EyeFlow data."""

import importlib
import os
import pkgutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from .core.base import (
    DEFAULT_INPUT_METHODS,
    PIPELINE_REGISTRY,
    BatchPipeline,
    MissingPipeline,
    PipelineContext,
    PipelineDescriptor,
    PipelineInputMethod,
    PipelineResult,
    format_required_metric_options,
    registerPipeline,
    required_metric_options_for,
)


def _extend_with_external_pipeline_dir() -> None:
    candidates = []
    env_path = os.getenv("ANGIOEYE_PIPELINE_DIR")
    if env_path:
        candidates.append(Path(env_path))
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent / "pipelines")
    for candidate in reversed(candidates):
        if candidate.is_dir():
            resolved = str(candidate.resolve())
            if resolved not in __path__:
                __path__.insert(0, resolved)


_extend_with_external_pipeline_dir()


def load_pipeline_catalog() -> tuple[
    list[PipelineDescriptor], list[PipelineDescriptor]
]:
    available = []
    missing = []
    PIPELINE_REGISTRY.clear()
    importlib.invalidate_caches()
    for module_info in pkgutil.iter_modules(__path__):
        if module_info.name in {"core", "utils"} or module_info.name.startswith("_"):
            continue
        module_name = f"{__name__}.{module_info.name}"
        try:
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
            else:
                importlib.import_module(module_name)
        except Exception as exc:
            missing.append(
                PipelineDescriptor(
                    name=module_info.name,
                    description=f"Import Error: {exc}",
                    available=False,
                    error_msg=str(exc),
                )
            )
    for cls in PIPELINE_REGISTRY.values():
        desc = PipelineDescriptor(
            name=cls.name,
            description=cls.description,
            available=cls.available,
            requires=list(cls.requires),
            missing_deps=list(cls.missing_deps),
            required_metrics=list(cls.required_metrics),
            required_metric_options=[
                list(group) for group in cls.required_metric_options
            ],
            input_methods=list(cls.input_methods),
            pipeline_cls=cls,
            visibility=cls.visibility,
            minimum_input_files=cls.minimum_input_files,
            accepted_input_modes=cls.accepted_input_modes,
        )
        (available if desc.available else missing).append(desc)
    available.sort(key=lambda item: item.name.lower())
    missing.sort(key=lambda item: item.name.lower())
    return available, missing


__all__ = [
    "BatchPipeline",
    "DEFAULT_INPUT_METHODS",
    "MissingPipeline",
    "PipelineContext",
    "PipelineDescriptor",
    "PipelineInputMethod",
    "PipelineResult",
    "format_required_metric_options",
    "required_metric_options_for",
    "registerPipeline",
    "load_pipeline_catalog",
]
