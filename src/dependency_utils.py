from __future__ import annotations

import importlib
import importlib.util
from functools import cache


def _module_name_from_requirement(requirement: str) -> str:
    module_name = requirement.split(";", 1)[0].strip()
    module_name = module_name.split("[", 1)[0].strip()
    for separator in ("<", ">", "=", "!", "~"):
        if separator in module_name:
            module_name = module_name.split(separator, 1)[0].strip()
    return module_name.replace("-", "_")


@cache
def is_module_available(module_name: str) -> bool:
    try:
        if importlib.util.find_spec(module_name) is not None:
            return True
    except (ImportError, ValueError, AttributeError):
        pass

    try:
        importlib.import_module(module_name)
    except Exception:
        return False
    return True


def find_missing_dependencies(requirements: list[str] | None) -> list[str]:
    missing: list[str] = []
    for requirement in requirements or []:
        module_name = _module_name_from_requirement(requirement)
        if module_name and not is_module_available(module_name):
            missing.append(module_name)
    return missing
