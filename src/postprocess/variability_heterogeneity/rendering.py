from pathlib import Path

import pandas as pd


def dataframe_to_latex_table(df, caption=None, label=None, font_size=r"\scriptsize"):
    latex_tabular = df.to_latex(
        index=False,
        escape=False,
        longtable=False,
        column_format="l" + "c" * (df.shape[1] - 1),
    )
    lines = [r"\begin{table}[H]", r"\centering"]
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.extend((latex_tabular, r"\end{table}"))
    return "\n".join(lines)


def save_table(df, csv_path, tex_path, caption, label, digits=3):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    tex_path.write_text(
        dataframe_to_latex_table(df, caption=caption, label=label),
        encoding="utf-8",
    )
    return [csv_path, tex_path]


def pretty_table_title(csv_file):
    name = csv_file.stem.replace("_", " ")
    if "spatial auc separability ranking all metrics" in name:
        groups = name.split(" spatial auc")[0].replace(" vs ", " and ")
        return f"Spatial variability metrics between {groups}, ranked by AUC separability"
    if "temporal auc separability ranking all metrics" in name:
        groups = name.split(" temporal auc")[0].replace(" vs ", " and ")
        return f"Temporal variability metrics between {groups}, ranked by AUC separability"
    if "strongest spatial variability contrast" in name:
        groups = name.split(" strongest")[0].replace(" vs ", " and ")
        return f"Top 10 strongest spatial variability contrasts between {groups}"
    if "strongest temporal variability contrast" in name:
        groups = name.split(" strongest")[0].replace(" vs ", " and ")
        return f"Top 10 strongest temporal variability contrasts between {groups}"
    title_patterns = {
        "n most spatially variable metrics": "Top 10 most spatially variable metrics",
        "n least spatially variable metrics": "Top 10 least spatially variable metrics",
        "n most temporally variable metrics": "Top 10 most temporally variable metrics",
        "n least temporally variable metrics": "Top 10 least temporally variable metrics",
    }
    for pattern, title in title_patterns.items():
        if pattern in name:
            return f"{title} in group {name.split(' vs ')[0]}"
    if "spatial variability table" in name:
        return f"Raw spatial variability metrics for group {name.replace(' spatial variability table', '')}"
    if "temporal variability table" in name:
        return f"Raw temporal variability metrics for group {name.replace(' temporal variability table', '')}"
    return name


def comparison_order(csv_file):
    name = csv_file.stem.lower()
    patterns = (
        ("most_spatially_variable", 0),
        ("most_temporally_variable", 0),
        ("least_spatially_variable", 1),
        ("least_temporally_variable", 1),
        ("strongest_spatial_variability_contrast", 2),
        ("strongest_temporal_variability_contrast", 2),
        ("spatial_auc_separability", 3),
        ("temporal_auc_separability", 3),
    )
    return next((rank for pattern, rank in patterns if pattern in name), 99)


def card_header(csv_file):
    name = csv_file.stem.lower()
    if "spatial_variability_table" in name:
        group = name.replace("_spatial_variability_table", "")
        return f"Raw - {group.replace('_', ' ').title()}"
    if "temporal_variability_table" in name:
        group = name.replace("_temporal_variability_table", "")
        return f"Raw - {group.replace('_', ' ').title()}"
    patterns = (
        ("most_spatially_variable", "Most spatially variable"),
        ("least_spatially_variable", "Least spatially variable"),
        ("most_temporally_variable", "Most temporally variable"),
        ("least_temporally_variable", "Least temporally variable"),
        ("strongest_spatial_variability_contrast", "Strongest contrast"),
        ("strongest_temporal_variability_contrast", "Strongest contrast"),
        ("spatial_auc_separability", "AUC separability"),
        ("temporal_auc_separability", "AUC separability"),
    )
    return next((title for pattern, title in patterns if pattern in name), "Table")


def _classify_csv_files(output_dir):
    groups = {"spatial_raw": [], "spatial_cmp": [], "temporal_raw": [], "temporal_cmp": []}
    for path in sorted(output_dir.rglob("*.csv")):
        name = path.stem.lower()
        if "spatial" in name:
            key = "spatial_raw" if "variability_table" in name else "spatial_cmp"
        elif "temporal" in name:
            key = "temporal_raw" if "variability_table" in name else "temporal_cmp"
        else:
            continue
        groups[key].append(path)
    groups["spatial_cmp"].sort(key=comparison_order)
    groups["temporal_cmp"].sort(key=comparison_order)
    return groups


def save_html_report(output_dir, title="Variability Report"):
    """Create one navigable HTML page containing all generated CSV tables."""
    output_dir = Path(output_dir)
    html_path = output_dir / "variability_report.html"
    groups = _classify_csv_files(output_dir)
    ordered_groups = (
        ("SPATIAL", "Raw tables", groups["spatial_raw"]),
        ("SPATIAL", "Comparison tables", groups["spatial_cmp"]),
        ("TEMPORAL", "Raw tables", groups["temporal_raw"]),
        ("TEMPORAL", "Comparison tables", groups["temporal_cmp"]),
    )
    ordered_files = [path for _, _, paths in ordered_groups for path in paths]
    indexes = {path: index for index, path in enumerate(ordered_files)}
    sections = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>{title}</title>",
        "<script>window.MathJax={tex:{inlineMath:[['$','$'],['\\\\(','\\\\)']]}};</script>",
        "<script src='https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js'></script>",
        """<style>
body{font-family:Arial,sans-serif;margin:30px;background:#fafafa}h1{text-align:center}
h2{margin-top:45px;border-bottom:2px solid #d9d9d9;padding-bottom:8px}
table{border-collapse:collapse;width:100%;background:white;margin-bottom:40px}
th{background:#e6e6e6;padding:8px;text-align:center;vertical-align:middle}td{border:1px solid #ddd;padding:6px;text-align:center}
tr:nth-child(even){background:#f5f5f5}.container{overflow-x:auto}
.dashboard-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px;margin:20px 0 40px}
.dashboard-card{background:#fff;border:1px solid #d8d8d8;border-radius:10px;padding:18px;text-decoration:none;color:#222}
.card-title{font-size:14px;color:#666;margin-top:5px}.scroll-top{position:fixed;right:25px;bottom:25px;font-size:26px}
</style></head><body id='top'>""",
        f"<h1>{title}</h1>",
    ]
    current_domain = None
    for domain, heading, paths in ordered_groups:
        if domain != current_domain:
            sections.append(f"<h2>{domain}</h2>")
            current_domain = domain
        sections.extend((f"<h3>{heading}</h3>", '<div class="dashboard-grid">'))
        for path in paths:
            sections.append(
                f"<a class='dashboard-card' href='#table{indexes[path]}'>"
                f"<b>{card_header(path)}</b><div class='card-title'>"
                f"{pretty_table_title(path)}</div></a>"
            )
        sections.append("</div>")
    for index, path in enumerate(ordered_files):
        table = pd.read_csv(path).to_html(index=False, escape=False)
        sections.append(
            f"<h2 id='table{index}'>{pretty_table_title(path)}</h2>"
            f"<div class='container'>{table}</div>"
        )
    sections.append("<script>MathJax.typeset();</script><a href='#top' class='scroll-top'>↑</a></body></html>")
    html_path.write_text("\n".join(sections), encoding="utf-8")
    print("HTML dashboard created:", html_path)
    return html_path
