#!/usr/bin/env python
"""
Generate an optional requirements file by aggregating `required_deps` declared
in `@registerPipeline(...)` decorators across the built-in pipeline modules.

Output: requirements-optional.txt
Usage:   python scripts/gen_optional_reqs.py
"""

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINES_DIR = PROJECT_ROOT / "src" / "pipelines"
OUTPUT_PATH = PROJECT_ROOT / "requirements-optional.txt"


def _decorator_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _parse_string_list(node: ast.AST) -> list[str]:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return []

    values: list[str] = []
    for elt in node.elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            values.append(elt.value)
    return values


def parse_required_deps(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except OSError:
        return []

    required_deps: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue

        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            if _decorator_name(decorator.func) != "registerPipeline":
                continue

            for keyword in decorator.keywords:
                if keyword.arg == "required_deps":
                    required_deps.update(_parse_string_list(keyword.value))

    return sorted(required_deps)


def main() -> None:
    requirements: set[str] = set()
    for path in PIPELINES_DIR.glob("*.py"):
        if path.name.startswith("_") or path.stem == "core":
            continue
        for req in parse_required_deps(path):
            requirements.add(req)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sorted_reqs = sorted(requirements)
    OUTPUT_PATH.write_text(
        "\n".join(sorted_reqs) + ("\n" if sorted_reqs else ""), encoding="utf-8"
    )
    print(f"Wrote {len(sorted_reqs)} optional requirement(s) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
