import json
import sys
from collections import defaultdict
from pathlib import Path


LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape(value):
    text = "" if value is None else str(value)
    return "".join(LATEX_ESCAPES.get(char, char) for char in text)


def render_type(type_desc):
    if not isinstance(type_desc, dict):
        return latex_escape(type_desc)
    kind = latex_escape(type_desc.get("kind", "unknown"))
    dtype = type_desc.get("dtype")
    if dtype and dtype != "Any":
        return f"{kind} ({latex_escape(dtype)})"
    return kind


def code_font_size(value):
    text = "" if value is None else str(value)
    segments = []
    for slash_part in text.split("/"):
        segments.extend(piece for piece in slash_part.split("_") if piece)
    max_segment_len = max((len(part) for part in segments), default=0)

    if max_segment_len >= 30 or len(text) >= 95:
        return r"\scriptsize"
    if max_segment_len >= 22 or len(text) >= 70:
        return r"\footnotesize"
    return ""


def pick_table_key_size(values):
    rank = {"": 0, r"\footnotesize": 1, r"\scriptsize": 2}
    max_rank = 0
    for value in values:
        size_cmd = code_font_size(value)
        max_rank = max(max_rank, rank.get(size_cmd, 0))
    for cmd, cmd_rank in rank.items():
        if cmd_rank == max_rank:
            return cmd
    return ""


def render_breakable_code(value, dynamic_size=True, size_override=None):
    raw = "" if value is None else str(value)
    escaped = latex_escape(raw)
    escaped = escaped.replace("/", r"/\allowbreak{}")
    escaped = escaped.replace(r"\_", r"\_\allowbreak{}")
    code = rf"\texttt{{{escaped}}}"
    if size_override is not None:
        size_cmd = size_override
        if not size_cmd:
            return code
        return rf"{{{size_cmd} {code}}}"

    if not dynamic_size:
        return code

    size_cmd = code_font_size(raw)
    if not size_cmd:
        return code
    return rf"{{{size_cmd} {code}}}"


def render_summary(payload):
    title = latex_escape(payload.get("pipeline_name", "Unknown Pipeline"))
    description = latex_escape(payload.get("description") or "No description available.")
    filepath = payload.get("filepath", "")
    return [
        r"\begin{center}",
        rf"\LARGE\textbf{{{title}}}",
        r"\end{center}",
        rf"\textbf{{Source:}} {render_breakable_code(filepath, dynamic_size=False)}\\",
        rf"\textbf{{Description:}} {description}",
    ]


def render_table(headers, rows, colspec=r"|p{0.70\textwidth}|p{0.24\textwidth}|"):
    lines = [rf"\begin{{longtable}}{{{colspec}}}"]
    header_line = (
        f"\\textbf{{{latex_escape(headers[0])}}} & "
        f"\\textbf{{{latex_escape(headers[1])}}} \\\\ \\hline"
    )
    lines.append(r"\hline")
    lines.append(header_line)
    lines.append(r"\endfirsthead")
    lines.append(r"\hline")
    lines.append(header_line)
    lines.append(r"\endhead")
    for left, right in rows:
        lines.append(f"{left} & {right} \\\\ \\hline")
    lines.append(r"\end{longtable}")
    return lines


def render_inputs(inputs):
    if not inputs:
        return [r"\emph{No inputs detected.}"]

    key_size = pick_table_key_size(inputs.keys())
    rows = []
    for path, type_desc in sorted(inputs.items()):
        rows.append(
            (render_breakable_code(path, size_override=key_size), render_type(type_desc))
        )
    return render_table(("Input", "Type"), rows)


def flatten_group_items(node, prefix):
    rows = []
    for item_name, type_desc in sorted(node.get("items", {}).items()):
        item_path = f"{prefix}/{item_name}" if prefix else item_name
        rows.append((item_path, type_desc))

    for child_name, child_node in sorted(
        (k, v) for k, v in node.items() if k != "items" and isinstance(v, dict)
    ):
        child_prefix = f"{prefix}/{child_name}" if prefix else child_name
        rows.extend(flatten_group_items(child_node, child_prefix))

    return rows


def render_value_list(values):
    code_size = pick_table_key_size(values)
    return ", ".join(render_breakable_code(v, size_override=code_size) for v in values)


def render_metrics(metrics):
    grouped = metrics.get("grouped", {}) if isinstance(metrics, dict) else {}
    flat_items = metrics.get("flat_items", {}) if isinstance(metrics, dict) else {}

    if not grouped and not flat_items:
        return [r"\emph{No metrics detected.}"]

    root_var_lists = {
        k: v for k, v in grouped.items() if isinstance(v, list) and all(isinstance(x, str) for x in v)
    }
    grouped_nodes = {k: v for k, v in grouped.items() if isinstance(v, dict)}

    rows = []
    group_vars = {}
    if grouped_nodes:
        for group_name, node in sorted(grouped_nodes.items()):
            group_vars[group_name] = {
                k: v
                for k, v in node.items()
                if k != "items" and isinstance(v, list) and all(isinstance(x, str) for x in v)
            }
            rows.extend(flatten_group_items(node, group_name))

    if flat_items:
        for item_name, type_desc in sorted(flat_items.items()):
            rows.append((item_name, type_desc))

    grouped_rows = defaultdict(list)
    for metric_path, type_desc in rows:
        if "/" in metric_path:
            group_path, key = metric_path.rsplit("/", 1)
        else:
            group_path, key = "(root)", metric_path
        grouped_rows[group_path].append((key, type_desc))

    lines = []
    if root_var_lists:
        for var_name, values in sorted(root_var_lists.items()):
            lines.append(
                rf"\textbf{{\textit{{{latex_escape(var_name)}}}}}: {render_value_list(values)}"
            )

    for index, group_path in enumerate(sorted(grouped_rows), start=1):
        lines.append(rf"\subsubsection*{{{index}) {render_breakable_code(group_path, dynamic_size=False)}}}")
        for var_name, values in sorted(group_vars.get(group_path, {}).items()):
            lines.append(
                rf"\textbf{{\textit{{{latex_escape(var_name)}}}}}: {render_value_list(values)}"
            )
        key_size = pick_table_key_size(key for key, _ in grouped_rows[group_path])
        table_rows = [
            (render_breakable_code(key, size_override=key_size), render_type(type_desc))
            for key, type_desc in sorted(grouped_rows[group_path])
        ]
        lines.extend(render_table(("Key", "Type"), table_rows))
        lines.append("")

    return lines


def build_latex(payload):
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{hyperref}",
        r"\usepackage{longtable}",
        r"\setlength{\parindent}{0pt}",
        r"\begin{document}",
        *render_summary(payload),
        r"",
        r"\subsection*{Inputs}",
        *render_inputs(payload.get("inputs", {})),
        r"",
        r"\subsection*{Metrics}",
        *render_metrics(payload.get("metrics", {})),
        r"\end{document}",
        "",
    ]
    return "\n".join(lines)


def load_payload(json_arg=None):
    if json_arg is None:
        text = sys.stdin.read()
    else:
        text = json_arg

    if not text.strip():
        raise ValueError("No JSON input provided")

    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")
    return json.loads(text)


def main():
    if len(sys.argv) not in {2, 3}:
        print("Usage: pipeline_docs_latex_gen.py <output.tex> [json_input]", file=sys.stderr)
        sys.exit(1)

    output_path = Path(sys.argv[1])
    json_input = sys.argv[2] if len(sys.argv) == 3 else None
    payload = load_payload(json_input)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_latex(payload), encoding="utf-8")


if __name__ == "__main__":
    main()