"""
Parses pipeline scripts in 'src/pipelines/' and emits a JSON document
describing each pipeline's name, class docstring, HDF5 input paths, and
metrics output keys.
"""

import ast
import json
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
# Value resolution (rigorous AST-based)
# ---------------------------------------------------------------------------

def resolve_value(node, env):
    """
    Rigorous resolution of AST expressions to their values.
    Returns a value (str, dict, list, etc.) or None if unresolvable.
    """
    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        return "{" + node.id + "}"

    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            return "{self." + node.attr + "}"
        return None

    if isinstance(node, ast.IfExp):
        body_val = resolve_value(node.body, env)
        orelse_val = resolve_value(node.orelse, env)
        if body_val == orelse_val:
            return body_val
        if body_val is not None and orelse_val is not None:
            return "{" + str(body_val) + "|" + str(orelse_val) + "}"
        return body_val or orelse_val

    if isinstance(node, ast.JoinedStr):
        parts = []
        for part in node.values:
            if isinstance(part, ast.Constant):
                parts.append(str(part.value))
            elif isinstance(part, ast.FormattedValue):
                val = resolve_value(part.value, env)
                parts.append(str(val) if val is not None else "{?}")
            else:
                parts.append("{?}")
        return "".join(parts)

    if isinstance(node, ast.Dict):
        out = {}
        for k_node, v_node in zip(node.keys, node.values):
            k = resolve_value(k_node, env)
            v = resolve_value(v_node, env)
            if isinstance(k, str):
                out[k] = v
        return out

    if isinstance(node, ast.List):
        return [resolve_value(elt, env) for elt in node.elts]

    if isinstance(node, ast.Tuple):
        return tuple(resolve_value(elt, env) for elt in node.elts)

    if isinstance(node, ast.Subscript):
        base = resolve_value(node.value, env)
        key = resolve_value(node.slice, env)
        if isinstance(base, dict) and key in base:
            return base[key]
        if isinstance(base, (list, tuple)) and isinstance(key, int):
            if -len(base) <= key < len(base):
                return base[key]
        return None

    return None


def build_method_env(method_node):
    """Build local variable environment for a method."""
    env = {}

    for arg in method_node.args.args:
        if arg.arg != "self":
            env[arg.arg] = "{" + arg.arg + "}"

    for node in ast.walk(method_node):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            if isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                val = resolve_value(node.value, env)
                if val is not None and isinstance(val, str):
                    env[name] = val

    return env


# ---------------------------------------------------------------------------
# H5 input extraction
# ---------------------------------------------------------------------------

def extract_h5_inputs(class_node):
    """Trace h5file[...] accesses in run() and called methods."""
    
    # Collect class string constants 
    class_strs = {}
    for item in class_node.body:
        if (
            isinstance(item, ast.Assign)
            and len(item.targets) == 1
            and isinstance(item.targets[0], ast.Name)
        ):
            value = resolve_value(item.value, {})
            if isinstance(value, str):
                class_strs[item.targets[0].id] = value
    
    methods = {n.name: n for n in class_node.body if isinstance(n, ast.FunctionDef)}
    run_method = methods.get("run")
    if not run_method:
        return {}
    
    run_args = [a.arg for a in run_method.args.args if a.arg != "self"]
    if not run_args:
        return {}
    
    h5_param = run_args[0]
    inputs = {}
    
    # BFS through methods starting from run()
    queue = [(run_method, h5_param)]
    visited = set()
    
    while queue:
        method, h5_alias = queue.pop(0)
        key = (method.name, h5_alias)
        if key in visited:
            continue
        visited.add(key)
        
        # Build local variable environment by walking statements (not just nodes)
        env = {a.arg: "{" + a.arg + "}" for a in method.args.args if a.arg != "self"}
        _walk_statements(method.body, env, class_strs, h5_alias, inputs, methods, queue)

    return inputs


def _walk_statements(stmts, env, class_strs, h5_alias, inputs, methods, queue):
    """Walk statement list, handling For loops and environments properly."""
    for stmt in stmts:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            if isinstance(stmt.targets[0], ast.Name):
                val = _resolve(stmt.value, env, class_strs)
                if val is not None:
                    env[stmt.targets[0].id] = val
        
        elif isinstance(stmt, ast.For):
            # Handle for loops: bind loop var to iterable values
            if isinstance(stmt.target, ast.Name):
                iterable = _resolve(stmt.iter, env, class_strs)
                loop_var = stmt.target.id
                
                if isinstance(iterable, list):
                    # Try each value from the list
                    for item in iterable:
                        loop_env = env.copy()
                        loop_env[loop_var] = item
                        _walk_statements(stmt.body, loop_env, class_strs, h5_alias, inputs, methods, queue)
                else:
                    # Fallback: just use the iterable as a placeholder
                    loop_env = env.copy()
                    loop_env[loop_var] = iterable
                    _walk_statements(stmt.body, loop_env, class_strs, h5_alias, inputs, methods, queue)
        
        elif isinstance(stmt, ast.If):
            # Walk if body with current env
            _walk_statements(stmt.body, env.copy(), class_strs, h5_alias, inputs, methods, queue)
            if stmt.orelse:
                _walk_statements(stmt.orelse, env.copy(), class_strs, h5_alias, inputs, methods, queue)
        
        elif isinstance(stmt, ast.With):
            _walk_statements(stmt.body, env.copy(), class_strs, h5_alias, inputs, methods, queue)

        elif isinstance(stmt, ast.Try):
            _walk_statements(stmt.body, env.copy(), class_strs, h5_alias, inputs, methods, queue)
            for handler in stmt.handlers:
                _walk_statements(handler.body, env.copy(), class_strs, h5_alias, inputs, methods, queue)
            if stmt.orelse:
                _walk_statements(stmt.orelse, env.copy(), class_strs, h5_alias, inputs, methods, queue)
            if stmt.finalbody:
                _walk_statements(stmt.finalbody, env.copy(), class_strs, h5_alias, inputs, methods, queue)
        
        # Also check for subscripts and calls in all statement expressions
        _collect_h5_paths(stmt, env, class_strs, h5_alias, inputs, {"kind": "unknown"})

        for node in ast.walk(stmt):
            
            # Follow calls to other methods
            if isinstance(node, ast.Call):
                if (isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "self"
                    and node.func.attr in methods):
                    
                    callee = methods[node.func.attr]
                    params = [a.arg for a in callee.args.args if a.arg != "self"]
                    
                    for i, arg in enumerate(node.args):
                        if i < len(params) and isinstance(arg, ast.Name) and arg.id == h5_alias:
                            queue.append((callee, params[i]))
                    
                    for kw in node.keywords:
                        if kw.arg in params and isinstance(kw.value, ast.Name) and kw.value.id == h5_alias:
                            queue.append((callee, kw.arg))


def _set_type(type_map, key, type_name):
    new_type = _normalize_type(type_name)
    old_type = type_map.get(key)
    if old_type is None or _type_score(new_type) > _type_score(old_type):
        type_map[key] = new_type


def _normalize_type(type_name):
    if isinstance(type_name, dict):
        out = dict(type_name)
        kind = out.get("kind", "unknown")
    else:
        kind = str(type_name)
        out = {"kind": kind}

    if kind in {"ndarray", "list", "tuple", "set"}:
        if kind == "ndarray":
            out.setdefault("dtype", "Any")

    return out


def _type_score(type_desc):
    kind = type_desc.get("kind", "unknown")
    score = 0 if kind == "unknown" else 1
    if type_desc.get("dtype") not in (None, "Any"):
        score += 1
    return score


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


# numpy function name → output dtype (None = preserve input dtype)
_NP_RETURNS = {
    "real": "float64", "imag": "float64", "angle": "float64",
    "abs": None, "absolute": None, "conj": None, "conjugate": None,
    "log": "float64", "log2": "float64", "log10": "float64",
    "exp": "float64", "sqrt": "float64",
    "sin": "float64", "cos": "float64", "tan": "float64",
    "mean": "float64", "std": "float64", "var": "float64",
    "nanmean": "float64", "nanstd": "float64", "nanmedian": "float64",
    "fft": "complex128", "ifft": "complex128", "rfft": "complex128", "irfft": "float64",
}


def _dtype_from_node(node):
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name):
            return node.attr
        return ast.unparse(node)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant):
        return str(node.value)
    return "Any"


def _array_desc(dtype="Any"):
    return {"kind": "ndarray", "dtype": dtype}


def _annotation_type(annotation):
    try:
        text = ast.unparse(annotation)
    except Exception:
        return {"kind": "unknown"}

    low = text.lower()
    if "ndarray" in low:
        return _array_desc()
    if low.startswith("list"):
        return {"kind": "list"}
    if low.startswith("tuple"):
        return {"kind": "tuple"}
    if low.startswith("set"):
        return {"kind": "set"}
    if low.startswith("dict"):
        return {"kind": "dict"}
    if low in {"int", "float", "str", "bool"}:
        return {"kind": low}
    return {"kind": "unknown"}


def _set_env_type(env_types, name, type_desc):
    new_type = _normalize_type(type_desc)
    old_type = env_types.get(name)
    if old_type is None or _type_score(new_type) > _type_score(old_type):
        env_types[name] = new_type


def _collect_h5_paths(node, env, class_strs, h5_alias, inputs, hint_type):
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == h5_alias:
        path = _resolve(node.slice, env, class_strs)
        if isinstance(path, str) and "/" in path:
            _set_type(inputs, path, hint_type)

    if isinstance(node, ast.Call):
        name = _call_name(node.func)
        call_hint = hint_type
        if name in {"asarray", "array"}:
            kw_dtype = "Any"
            for kw in node.keywords:
                if kw.arg == "dtype":
                    kw_dtype = _dtype_from_node(kw.value)
                    break
            if isinstance(hint_type, dict) and hint_type.get("kind") == "ndarray":
                dtype = kw_dtype if kw_dtype != "Any" else hint_type.get("dtype", "Any")
                call_hint = _array_desc(dtype=dtype)
            else:
                call_hint = _array_desc(dtype=kw_dtype)
        elif name in {"zeros", "ones", "full", "empty"}:
            kw_dtype = "Any"
            for kw in node.keywords:
                if kw.arg == "dtype":
                    kw_dtype = _dtype_from_node(kw.value)
                    break
            if kw_dtype == "Any" and len(node.args) > 1:
                kw_dtype = _dtype_from_node(node.args[1])
            call_hint = _array_desc(dtype=kw_dtype if kw_dtype != "Any" else "float64")
        elif name == "astype":
            dtype = _dtype_from_node(node.args[0]) if node.args else "Any"
            call_hint = _array_desc(dtype=dtype)
        elif name in {"float", "int", "str", "bool"}:
            call_hint = {"kind": name}

        _collect_h5_paths(node.func, env, class_strs, h5_alias, inputs, call_hint)
        for arg in node.args:
            _collect_h5_paths(arg, env, class_strs, h5_alias, inputs, call_hint)
        for kw in node.keywords:
            _collect_h5_paths(kw.value, env, class_strs, h5_alias, inputs, call_hint)
        return

    for child in ast.iter_child_nodes(node):
        _collect_h5_paths(child, env, class_strs, h5_alias, inputs, hint_type)


def _resolve(node, env, class_strs):
    """Resolve AST node to value using env and class string constants."""
    if isinstance(node, ast.Constant):
        return node.value
    
    if isinstance(node, ast.Name):
        return env.get(node.id, "{" + node.id + "}")
    
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            return class_strs.get(node.attr)
        return None
    
    if isinstance(node, ast.Dict):
        out = {}
        for k_node, v_node in zip(node.keys, node.values):
            k = _resolve(k_node, env, class_strs)
            v = _resolve(v_node, env, class_strs)
            if isinstance(k, str):
                out[k] = v
        return out
    
    if isinstance(node, ast.Subscript):
        base = _resolve(node.value, env, class_strs)
        key = _resolve(node.slice, env, class_strs)
        if isinstance(base, dict) and key in base:
            return base[key]
        if isinstance(base, (list, tuple)) and isinstance(key, int):
            if -len(base) <= key < len(base):
                return base[key]
        return None
    
    if isinstance(node, ast.List):
        return [_resolve(elt, env, class_strs) for elt in node.elts]
    
    if isinstance(node, ast.JoinedStr):
        parts = []
        for part in node.values:
            if isinstance(part, ast.Constant):
                parts.append(str(part.value))
            elif isinstance(part, ast.FormattedValue):
                v = _resolve(part.value, env, class_strs)
                parts.append(str(v) if v else "")
            else:
                parts.append("")
        return "".join(parts) if parts else None
    
    if isinstance(node, ast.IfExp):
        l = _resolve(node.body, env, class_strs)
        r = _resolve(node.orelse, env, class_strs)
        return l if l == r else None
    
    return None


# ---------------------------------------------------------------------------
# Metrics key extraction
# ---------------------------------------------------------------------------

def _expr_type(node, env_types=None):
    if env_types is None:
        env_types = {}

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return {"kind": "bool"}
        if isinstance(node.value, int):
            return {"kind": "int"}
        if isinstance(node.value, float):
            return {"kind": "float"}
        if isinstance(node.value, str):
            return {"kind": "str"}
        return {"kind": type(node.value).__name__}

    if isinstance(node, (ast.List, ast.ListComp)):
        return {"kind": "list"}

    if isinstance(node, (ast.Tuple, ast.Set)):
        return {"kind": "tuple" if isinstance(node, ast.Tuple) else "set"}

    if isinstance(node, ast.Dict):
        return {"kind": "dict"}

    if isinstance(node, ast.Name):
        return env_types.get(node.id, {"kind": "unknown"})

    if isinstance(node, ast.Attribute):
        if node.attr == "shape":
            return {"kind": "tuple"}
        if node.attr == "ndim":
            return {"kind": "int"}
        return {"kind": "unknown"}

    if isinstance(node, ast.Call):
        name = _call_name(node.func)
        if name == "with_attrs" and node.args:
            return _expr_type(node.args[0], env_types)
        if name in {"asarray", "array", "zeros", "ones", "full", "empty", "stack", "concatenate"}:
            kw_dtype = "Any"
            for kw in node.keywords:
                if kw.arg == "dtype":
                    kw_dtype = _dtype_from_node(kw.value)
                    break
            if name in {"zeros", "ones", "full", "empty"} and node.args:
                pos_dtype = _dtype_from_node(node.args[1]) if len(node.args) > 1 else "float64"
                final_dtype = pos_dtype if pos_dtype != "Any" else kw_dtype if kw_dtype != "Any" else "float64"
                return _array_desc(dtype=final_dtype)
            return _array_desc(dtype=kw_dtype)
        if name in _NP_RETURNS:
            dtype = _NP_RETURNS[name]
            if dtype is None and node.args:
                base = _expr_type(node.args[0], env_types)
                dtype = base.get("dtype", "float64") if base.get("kind") == "ndarray" else "float64"
            return _array_desc(dtype=dtype or "float64")
        if name == "astype":
            dtype = _dtype_from_node(node.args[0]) if node.args else "Any"
            base = {"kind": "unknown"}
            if isinstance(node.func, ast.Attribute):
                base = _expr_type(node.func.value, env_types)
            return _array_desc(dtype=dtype)
        if name in {"reshape", "ravel", "flatten", "squeeze"}:
            base = {"kind": "unknown"}
            if isinstance(node.func, ast.Attribute):
                base = _expr_type(node.func.value, env_types)
            return _array_desc(dtype=base.get("dtype", "Any") if base.get("kind") == "ndarray" else "Any")
        if name in {"float", "int", "str", "bool"}:
            return {"kind": name}
        return {"kind": "unknown"}

    if isinstance(node, ast.BinOp):
        lt = _expr_type(node.left, env_types)
        rt = _expr_type(node.right, env_types)
        for t in (lt, rt):
            if t.get("kind") == "ndarray":
                return _array_desc(dtype=t.get("dtype", "Any"))
        return {"kind": "float"}

    if isinstance(node, ast.UnaryOp):
        return _expr_type(node.operand, env_types)

    if isinstance(node, ast.Compare):
        lt = _expr_type(node.left, env_types)
        if lt.get("kind") == "ndarray":
            return _array_desc(dtype="bool")
        return {"kind": "bool"}

    return {"kind": "unknown"}


def extract_metrics_types(class_node):
    """Collect metrics full path -> inferred type."""
    metric_types = {}

    for method_node in class_node.body:
        if not isinstance(method_node, ast.FunctionDef):
            continue

        env = build_method_env(method_node)
        env_types = {}

        for arg in method_node.args.args:
            if arg.arg == "self":
                continue
            if arg.annotation is not None:
                _set_env_type(env_types, arg.arg, _annotation_type(arg.annotation))

        for node in ast.walk(method_node):
            if isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    _set_env_type(env_types, node.target.id, _annotation_type(node.annotation))
                continue

            if isinstance(node, ast.Assign) and node.targets:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    _set_env_type(env_types, target.id, _expr_type(node.value, env_types))

                if isinstance(target, ast.Subscript):
                    if isinstance(target.value, ast.Name) and target.value.id == "metrics":
                        key = resolve_value(target.slice, env)
                        if isinstance(key, str):
                            _set_type(metric_types, key, _expr_type(node.value, env_types))

            if isinstance(node, ast.Call):
                func = node.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "_pack_split_complex"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "self"
                    and len(node.args) >= 2
                ):
                    path = resolve_value(node.args[1], env)
                    if isinstance(path, str):
                        for suffix in ["_real", "_imag"]:
                            combined = path + suffix
                            _set_type(metric_types, combined, _array_desc())

    return metric_types


# ---------------------------------------------------------------------------
# Tree building for grouped metrics
# ---------------------------------------------------------------------------

def compact_child_chain(parent, key):
    """Compact single-child layers by merging them into the key."""
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
        for k, v in node.items():
            if k == "items":
                parent[merged_key].setdefault("items", {}).update(v)
            else:
                parent[merged_key][k] = v
    else:
        parent[merged_key] = node


def build_metrics_tree(metric_types):
    """Build grouped tree with automatic single-child compaction and typed leaves."""
    tree = {}
    flat_items = {}

    for key, value_type in metric_types.items():
        if key.startswith("{path}_"):
            continue

        if "/" not in key:
            _set_type(flat_items, key, value_type)
            continue

        segments = [s for s in key.split("/") if s]
        if not segments:
            continue

        branch = tree
        path_chain = []
        for segment in segments[:-1]:
            path_chain.append((branch, segment))
            branch = branch.setdefault(segment, {})

        leaf = segments[-1]
        items = branch.setdefault("items", {})
        _set_type(items, leaf, value_type)

        for parent, segment in reversed(path_chain):
            compact_child_chain(parent, segment)

    payload = {"grouped": tree}
    if flat_items:
        payload["flat_items"] = flat_items
    return payload


# ---------------------------------------------------------------------------
# Description extraction
# ---------------------------------------------------------------------------

def get_description(docstring):
    """Get first non-empty line from docstring."""
    if not docstring:
        return None
    for line in docstring.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def process_pipeline_script(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)
    class_node, pipeline_name = find_pipeline_class(tree)

    if class_node is None:
        return {"error": "No @registerPipeline class found", "filepath": filepath}

    docstring = ast.get_docstring(class_node)
    inputs = extract_h5_inputs(class_node)
    metric_types = extract_metrics_types(class_node)

    return {
        "filepath": filepath,
        "pipeline_name": pipeline_name,
        "description": get_description(docstring),
        "inputs": inputs,
        "metrics": build_metrics_tree(metric_types),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: pipeline_docs_ast.py <pipeline_file.py>", file=sys.stderr)
        sys.exit(1)

    result = process_pipeline_script(sys.argv[1])
    print(json.dumps(result, indent=2))