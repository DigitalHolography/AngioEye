"""
Parses pipeline scripts in 'src/pipelines/' and emits a JSON document
describing each pipeline's name, class docstring, HDF5 input paths, and
metrics output keys.

Variable expansion: method parameters become {param_name} placeholders;
simple local assignments (e.g. base = f"{prefix}/raw") are resolved inline
so that metrics key templates are fully expanded.
"""

import ast
import json
import re
import sys


# ---------------------------------------------------------------------------
# Decorator / class discovery
# ---------------------------------------------------------------------------

def get_pipeline_name(decorator):
    if isinstance(decorator, ast.Call) and getattr(decorator.func, 'id', '') == 'registerPipeline':
        for keyword in decorator.keywords:
            if (
                keyword.arg == 'name'
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            ):
                return keyword.value.value
    return None


def find_pipeline_class(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for decorator in node.decorator_list:
                pipeline_name = get_pipeline_name(decorator)
                if pipeline_name:
                    return node, pipeline_name
    return None, None


# ---------------------------------------------------------------------------
# HDF5 inputs from class-level string attributes
# ---------------------------------------------------------------------------

def extract_class_h5_inputs(class_node):
    """
    Class-level assignments whose value is a '/' -prefixed string constant
    are HDF5 paths the pipeline reads from h5file.
    """
    inputs = {}
    for node in class_node.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            and node.value.value.startswith("/")
        ):
            inputs[node.targets[0].id] = node.value.value
    return inputs


# ---------------------------------------------------------------------------
# Metrics key extraction with variable expansion
# ---------------------------------------------------------------------------

def try_resolve_str(node, env):
    """
    Attempt to turn an AST node into a plain string using *env*.
    - ast.Constant strings  → returned as-is.
    - ast.Name              → looked up in env; unknown names become {name}.
    - ast.JoinedStr (f"…")  → each part is resolved recursively and concatenated.
    Returns None when the node is not a string-like expression at all.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id, "{" + node.id + "}")
    if isinstance(node, ast.IfExp):
        left = try_resolve_str(node.body, env)
        right = try_resolve_str(node.orelse, env)
        if left is None and right is None:
            return None
        if left is None:
            return right
        if right is None:
            return left
        if left == right:
            return left
        return "{" + left + "|" + right + "}"
    if isinstance(node, ast.JoinedStr):
        parts = []
        for part in node.values:
            if isinstance(part, ast.Constant):
                parts.append(str(part.value))
            elif isinstance(part, ast.FormattedValue):
                inner = try_resolve_str(part.value, env)
                parts.append(inner if inner is not None else "{?}")
            else:
                parts.append("{?}")
        return "".join(parts)
    return None


def is_h5_path_template(value):
    """
    Heuristic: HDF5-like path template should contain at least one '/'.
    This accepts concrete paths and parameterized templates such as
    '{vessel}/VelocityPerBeat/...'.
    """
    return isinstance(value, str) and "/" in value


def build_method_env(method_node):
    """
    Build a string-value environment for one method.

    1. All parameters (except self) are seeded as {param_name} placeholders.
    2. Simple local assignments (single Name target, string-like value) are
       collected in BFS order and added to the env so that later assignments
       that reference earlier locals are expanded (e.g. base = f"{prefix}/raw"
       correctly substitutes into metrics[f"{base}/..."]).
    """
    env = {}
    # Seed parameters as {name} placeholders
    for arg in method_node.args.args:
        if arg.arg != "self":
            env[arg.arg] = "{" + arg.arg + "}"

    # Collect local assignments. ast.walk is BFS, so function-body statements
    # are visited left-to-right before their own children, preserving source order
    # for top-level locals.
    for node in ast.walk(method_node):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name = node.targets[0].id
            val = try_resolve_str(node.value, env)
            if val is not None:
                env[name] = val
    return env


def extract_metrics_keys(class_node):
    """
    Walk every method in the class and collect resolved metrics key strings.

    Two sources are handled:
    - Direct  metrics[key] = ...  assignments.
    - Calls to self._pack_split_complex(metrics, path, ...) which always write
      metrics[path + '_real'] and metrics[path + '_imag'].

    Variable references in key expressions are expanded using the per-method
    environment built by build_method_env().
    """
    keys = []
    for method_node in class_node.body:
        if not isinstance(method_node, ast.FunctionDef):
            continue

        env = build_method_env(method_node)

        for child in ast.walk(method_node):
            # Metrics directly assigned
            if isinstance(child, ast.Assign) and child.targets:
                target = child.targets[0]
                if isinstance(target, ast.Subscript):
                    try:
                        if "metrics" not in ast.unparse(target):
                            continue
                        resolved = try_resolve_str(target.slice, env)
                        if resolved is not None:
                            keys.append(resolved)
                    except Exception:
                        pass

            # Special handling for functions filling metrics 
            if isinstance(child, ast.Call):
                func = child.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "_pack_split_complex"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "self"
                    and len(child.args) >= 2
                ):
                    path_resolved = try_resolve_str(child.args[1], env)
                    if path_resolved is not None:
                        keys.append(path_resolved + "_real")
                        keys.append(path_resolved + "_imag")

    return list(dict.fromkeys(keys))


def extract_h5_inputs(class_node):
    """
    Collect HDF5 input path templates from:
    1) class-level '/...'-prefixed constants
    2) method-local '*path' string assignments
    3) direct h5file[...] access expressions
    """
    inputs = dict(extract_class_h5_inputs(class_node))

    for method_node in class_node.body:
        if not isinstance(method_node, ast.FunctionDef):
            continue

        env = build_method_env(method_node)

        for node in ast.walk(method_node):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                var_name = node.targets[0].id
                val = try_resolve_str(node.value, env)
                if val is not None:
                    env[var_name] = val
                    if "path" in var_name.lower() and is_h5_path_template(val):
                        inputs[f"{method_node.name}.{var_name}"] = val

            if isinstance(node, ast.Subscript):
                # Detect h5file[...] patterns and resolve the indexed key
                if isinstance(node.value, ast.Name) and node.value.id == "h5file":
                    key = try_resolve_str(node.slice, env)
                    if is_h5_path_template(key):
                        inputs[f"{method_node.name}.h5file[{key}]"] = key

    return inputs


def summarize_description(docstring):
    if not docstring:
        return None
    cleaned = [line.strip() for line in docstring.splitlines() if line.strip()]
    if not cleaned:
        return None
    return cleaned[0]


def extract_structure_from_docstring(docstring):
    if not docstring:
        return None
    for line in docstring.splitlines():
        stripped = line.strip()
        if "[" in stripped and "]" in stripped:
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\[.*\]$", stripped):
                return stripped
    return None


def normalize_label(text):
    value = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return value or "input"


def choose_input_label(source, path):
    source_leaf = source.split(".")[-1].lower()
    path_lower = path.lower()

    if "beatperiod" in source_leaf or "t_input" == source_leaf or "beatperiod" in path_lower:
        return "beat_period"
    if "arter" in source_leaf or "/artery/" in path_lower:
        if "segment" in source_leaf or "signal" in path_lower or "waveform" in path_lower:
            return "artery_waveforms"
        return "artery_input"
    if "vein" in source_leaf or "/vein/" in path_lower:
        if "segment" in source_leaf or "signal" in path_lower or "waveform" in path_lower:
            return "vein_waveforms"
        return "vein_input"

    if source_leaf.endswith("_path"):
        return normalize_label(source_leaf[:-5])
    if source_leaf.endswith("_input"):
        return normalize_label(source_leaf[:-6])
    return normalize_label(source_leaf)


def make_legible_inputs(inputs_dict):
    path_to_source = {}
    for source, path in inputs_dict.items():
        if ".h5file[" in source:
            continue
        path_to_source.setdefault(path, source)

    result = {}
    used_labels = set()
    for path, source in path_to_source.items():
        base_label = choose_input_label(source, path)
        label = base_label
        idx = 2
        while label in used_labels:
            label = f"{base_label}_{idx}"
            idx += 1
        used_labels.add(label)
        result[label] = path
    return result


def merge_tree_nodes(dst, src):
    """Merge src tree node into dst in-place."""
    for key, value in src.items():
        if key == "items":
            dst_items = dst.setdefault("items", [])
            for item in value:
                if item not in dst_items:
                    dst_items.append(item)
            continue

        if key in dst and isinstance(dst[key], dict) and isinstance(value, dict):
            merge_tree_nodes(dst[key], value)
        elif key not in dst:
            dst[key] = value


def compact_child_chain(parent, key):
    """
    Compact parent[key] by folding single-child layers directly into the key.
    This runs during insertion, so no full-tree post-process pass is needed.
    """
    if key not in parent or not isinstance(parent[key], dict):
        return

    node = parent[key]
    merged_key = key

    while True:
        child_keys = [k for k in node.keys() if k != "items"]
        if "items" in node or len(child_keys) != 1:
            break
        child_key = child_keys[0]
        child_node = node[child_key]
        if not isinstance(child_node, dict):
            break
        merged_key = f"{merged_key}/{child_key}"
        node = child_node

    if merged_key == key:
        return

    del parent[key]
    if merged_key in parent and isinstance(parent[merged_key], dict):
        merge_tree_nodes(parent[merged_key], node)
    else:
        parent[merged_key] = node


def build_metrics_tree(metrics_keys):
    """
    Generic token-saving regrouping for metric paths.

    - Slash-delimited keys are organized in a nested tree where leaf names are
      stored in a compact "items" list under each branch.
    - Non-slash keys are kept in "flat_items".
    - No hardcoded metric families are used.
    """
    tree = {}
    flat_items = []

    for key in metrics_keys:
        if key.startswith("{path}_"):
            continue

        if "/" not in key:
            if key not in flat_items:
                flat_items.append(key)
            continue

        segments = [segment for segment in key.split("/") if segment]
        if not segments:
            continue

        branch = tree
        path_chain = []
        for segment in segments[:-1]:
            path_chain.append((branch, segment))
            branch = branch.setdefault(segment, {})

        leaf = segments[-1]
        items = branch.setdefault("items", [])
        if leaf not in items:
            items.append(leaf)

        for parent, segment in reversed(path_chain):
            compact_child_chain(parent, segment)

    payload = {
        "grouped": tree,
    }
    if flat_items:
        payload["flat_items"] = flat_items
    return payload


def process_pipeline_script(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)
    class_node, pipeline_name = find_pipeline_class(tree)

    if class_node is None:
        return {"error": "No @registerPipeline class found", "filepath": filepath}

    docstring = ast.get_docstring(class_node)
    inputs_dict = extract_h5_inputs(class_node)
    metrics_keys = extract_metrics_keys(class_node)

    return {
        "filepath": filepath,
        "pipeline_name": pipeline_name,
        "description": summarize_description(docstring),
        "inputs": make_legible_inputs(inputs_dict),
        "structure": extract_structure_from_docstring(docstring),
        "metrics": build_metrics_tree(metrics_keys),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: pipeline_docs_ast.py <pipeline_file.py>", file=sys.stderr)
        sys.exit(1)

    result = process_pipeline_script(sys.argv[1])
    print(json.dumps(result, indent=2))