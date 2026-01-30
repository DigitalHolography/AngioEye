#!/usr/bin/env python
"""
Generate an optional requirements file by aggregating REQUIRES lists from all pipeline modules.

Output: AngioEye/pipelines/requirements-optional.txt
Usage:   python scripts/gen_optional_reqs.py
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINES_DIR = PROJECT_ROOT / "AngioEye" / "pipelines"
OUTPUT_PATH = PIPELINES_DIR / "requirements-optional.txt"


def parse_requires(path: Path) -> List[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
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


def main() -> None:
    requirements: Set[str] = set()
    for path in PIPELINES_DIR.glob("*.py"):
        if path.name.startswith("_") or path.stem == "core":
            continue
        for req in parse_requires(path):
            requirements.add(req)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sorted_reqs = sorted(requirements)
    OUTPUT_PATH.write_text("\n".join(sorted_reqs) + ("\n" if sorted_reqs else ""), encoding="utf-8")
    print(f"Wrote {len(sorted_reqs)} optional requirement(s) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
