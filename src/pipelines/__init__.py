import ast
import importlib
import importlib.util
import inspect
import pkgutil
from typing import List, Tuple

from .core.base import ProcessPipeline, ProcessResult
from .core.utils import write_combined_results_h5, write_result_h5


class MissingPipeline(ProcessPipeline):
    """Placeholder for pipelines whose dependencies are missing."""

    available = False
    missing_deps: List[str]
    requires: List[str]

    def __init__(self, name: str, description: str, missing_deps: List[str], requires: List[str]) -> None:
        super().__init__()
        self.name = name
        self.description = description or "Pipeline unavailable (missing dependencies)."
        self.missing_deps = missing_deps
        self.requires = requires

    def run(self, _h5file):
        missing = ", ".join(self.missing_deps or self.requires or ["unknown dependency"])
        raise ImportError(f"Pipeline '{self.name}' unavailable. Missing dependencies: {missing}")


def _module_docstring(module_name: str) -> str:
    spec = importlib.util.find_spec(module_name)
    if not spec or not spec.origin or not spec.origin.endswith(".py"):
        return ""
    with open(spec.origin, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source)
    return ast.get_docstring(tree) or ""


def _parse_requires_from_source(module_name: str) -> List[str]:
    spec = importlib.util.find_spec(module_name)
    if not spec or not spec.origin or not spec.origin.endswith(".py"):
        return []
    try:
        with open(spec.origin, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=spec.origin)
    except OSError:
        return []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "REQUIRES":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        vals = []
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                vals.append(elt.value)
                        return vals
    return []


def _normalize_req_name(req: str) -> str:
    """Extract importable package name from a requirement string."""
    for sep in ("[", "==", ">=", "<=", "~=", "!=", ">", "<"):
        if sep in req:
            return req.split(sep, 1)[0]
    return req


def _missing_requirements(requires: List[str]) -> List[str]:
    missing: List[str] = []
    for req in requires:
        pkg = _normalize_req_name(req).strip()
        if not pkg:
            continue
        if importlib.util.find_spec(pkg) is None:
            missing.append(pkg)
    return missing


def _discover_pipelines() -> Tuple[List[ProcessPipeline], List[MissingPipeline]]:
    available: List[ProcessPipeline] = []
    missing: List[MissingPipeline] = []
    seen_classes = set()

    for module_info in pkgutil.iter_modules(__path__):
        if module_info.name in {"core"} or module_info.name.startswith("_"):
            continue
        module_name = f"{__name__}.{module_info.name}"
        requires = _parse_requires_from_source(module_name)
        doc = _module_docstring(module_name)

        # First, check for missing requirements before importing heavy modules.
        pre_missing = _missing_requirements(requires)
        if pre_missing:
            missing.append(MissingPipeline(module_info.name, doc, pre_missing, requires))
            continue

        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            # Capture missing dependency if ModuleNotFoundError has a name.
            missing_deps = []
            if isinstance(exc, ModuleNotFoundError) and exc.name and exc.name not in {module_name, module_info.name}:
                missing_deps = [exc.name]
            if not missing_deps:
                missing_deps = requires
            missing.append(MissingPipeline(module_info.name, doc, missing_deps, requires))
            continue

        module_requires = getattr(module, "REQUIRES", requires)
        post_missing = _missing_requirements(module_requires)
        if post_missing:
            missing.append(MissingPipeline(module_info.name, doc, post_missing, module_requires))
            continue
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if not issubclass(cls, ProcessPipeline) or cls is ProcessPipeline:
                continue
            if cls.__module__ != module.__name__:
                continue
            if cls in seen_classes:
                continue
            seen_classes.add(cls)
            try:
                inst = cls()
                inst.available = True  # type: ignore[attr-defined]
                inst.requires = module_requires  # type: ignore[attr-defined]
                available.append(inst)
            except TypeError:
                # Skip classes requiring constructor args.
                continue

    available.sort(key=lambda p: p.name.lower())
    missing.sort(key=lambda p: p.name.lower())
    return available, missing


def load_all_pipelines(include_missing: bool = False) -> List[ProcessPipeline]:
    """
    Discover and instantiate pipelines. Optionally include placeholders for missing deps.
    """
    available, missing = _discover_pipelines()
    return available + missing if include_missing else available


def load_pipeline_catalog() -> Tuple[List[ProcessPipeline], List[MissingPipeline]]:
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
    "load_all_pipelines",
    "load_pipeline_catalog",
    "MissingPipeline",
    *[_cls.__name__ for _cls in (p.__class__ for p in _AVAILABLE)],
]
