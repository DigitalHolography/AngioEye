from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any


def _find_checkout_src(
    module_filename: str,
    start_dir: Path | None = None,
) -> Path | None:
    current_dir = (start_dir or Path.cwd()).resolve()
    for candidate in (current_dir, *current_dir.parents):
        src_dir = candidate / "src"
        if (candidate / "pyproject.toml").is_file() and (
            src_dir / module_filename
        ).is_file():
            return src_dir
    return None


def _load_local_module(module_name: str, module_path: Path) -> Any:
    module_alias = f"_angioeye_checkout_{module_name.replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(module_alias, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_alias] = module
    spec.loader.exec_module(module)
    return module


def _call_entry(
    module_name: str,
    module_filename: str,
    func_name: str,
    *args: Any,
    start_dir: Path | None = None,
    **kwargs: Any,
) -> Any:
    src_dir = _find_checkout_src(module_filename, start_dir=start_dir)
    if src_dir is not None:
        src_dir_str = str(src_dir)
        if src_dir_str in sys.path:
            sys.path.remove(src_dir_str)
        sys.path.insert(0, src_dir_str)
        module = _load_local_module(module_name, src_dir / module_filename)
    else:
        module = importlib.import_module(module_name)
    entry = getattr(module, func_name)
    return entry(*args, **kwargs)


def main() -> Any:
    return _call_entry("angio_eye", "angio_eye.py", "main")


def cli_main(argv: list[str] | None = None) -> Any:
    return _call_entry("cli", "cli.py", "main", argv)
