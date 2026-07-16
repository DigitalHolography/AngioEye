import shutil
from pathlib import Path

import matplotlib.pyplot as plt

from input_output.archive_io import replace_folder_in_zip

from .compute import analyze_zip, find_control_group, safe_name
from .constants import (
    DEFAULT_TOP_N,
    INPUT_METRICS,
    PLOT_STYLE,
    SEGMENT_MODE,
    SPATIAL_DESCRIPTOR_MAP,
    SPATIAL_SELECTED_METRICS,
    SPATIAL_VARIABILITY_COLUMNS,
    TEMPORAL_DESCRIPTOR_MAP,
    TEMPORAL_SELECTED_METRICS,
    TEMPORAL_VARIABILITY_COLUMNS,
)
from .formatting import latex_escape_text
from .plots import export_variability_value_plots
from .rendering import save_html_report, save_table
from .tables import (
    build_auc_separability_ranking_table,
    build_contrast_table,
    build_descriptor_pvalue_summary_table,
    build_group_separation_metrics_table,
    build_mannwhitney_ranking_table,
    build_spatial_group_table,
    build_temporal_group_table,
    build_variability_ranking_table,
)


def _write_table(generated, frame, directory, stem, caption, label, digits):
    generated.extend(
        save_table(
            frame,
            directory / f"{stem}.csv",
            directory / f"{stem}.tex",
            caption=caption,
            label=label,
            digits=digits,
        )
    )


def _export_raw_tables(results, spatial_dir, temporal_dir, metrics, digits, generated):
    for group_name in sorted(results):
        safe_group = safe_name(group_name)
        escaped_group = latex_escape_text(group_name)
        _write_table(
            generated,
            build_spatial_group_table(results[group_name], metrics=metrics, digits=digits),
            spatial_dir,
            f"{safe_group}_spatial_variability_table",
            f"Raw spatial variability metrics for group {escaped_group}",
            f"tab:{safe_group}_spatial_variability_raw",
            digits,
        )
        _write_table(
            generated,
            build_temporal_group_table(results[group_name], metrics=metrics, digits=digits),
            temporal_dir,
            f"{safe_group}_temporal_variability_table",
            f"Raw temporal variability metrics for group {escaped_group}",
            f"tab:{safe_group}_temporal_variability_raw",
            digits,
        )


def _export_rankings(
    generated,
    control_results,
    group_results,
    higher_metrics,
    control_name,
    group_name,
    metrics,
    directory,
    pair,
    domain,
    top_n,
    digits,
):
    adjective = "spatially" if domain == "spatial" else "temporally"
    for ascending, qualifier in ((False, "most"), (True, "least")):
        frame = build_variability_ranking_table(
            control_results,
            group_results,
            higher_metrics=higher_metrics,
            control_name=control_name,
            group_name=group_name,
            metrics=metrics,
            n=top_n,
            ascending=ascending,
            digits=digits,
            domain_name=domain,
        )
        _write_table(
            generated,
            frame,
            directory,
            f"{pair}_n_{qualifier}_{adjective}_variable_metrics",
            f"Top {top_n} {qualifier} {adjective} variable metrics in group "
            f"{latex_escape_text(group_name)}",
            f"tab:{pair}_{qualifier}_{adjective}_variable",
            digits,
        )

    frame = build_contrast_table(
        control_results,
        group_results,
        higher_metrics=higher_metrics,
        control_name=control_name,
        group_name=group_name,
        metrics=metrics,
        n=top_n,
        digits=digits,
        domain_name=domain,
    )
    _write_table(
        generated,
        frame,
        directory,
        f"{pair}_strongest_{domain}_variability_contrast",
        f"Top {top_n} strongest {domain} variability contrasts between "
        f"{latex_escape_text(group_name)} and {latex_escape_text(control_name)}",
        f"tab:{pair}_strongest_{domain}_contrast",
        digits,
    )


def _export_statistical_tables(
    generated,
    control_results,
    group_results,
    higher_metrics,
    descriptor_map,
    selected_metrics,
    control_name,
    group_name,
    metrics,
    directory,
    pair,
    domain,
    selected_suffix,
    digits,
    evaluate_both_directions,
):
    escaped_control = latex_escape_text(control_name)
    escaped_group = latex_escape_text(group_name)


    frame = build_auc_separability_ranking_table(
        control_results,
        group_results,
        higher_metrics=higher_metrics,
        control_name=control_name,
        group_name=group_name,
        metrics=metrics,
        digits=digits,
        evaluate_both_directions=evaluate_both_directions,
    )
    _write_table(
        generated,
        frame,
        directory,
        f"{pair}_{domain}_auc_separability_ranking_all_metrics",
        f"{domain.capitalize()} variability metrics between {escaped_control} "
        f"and {escaped_group}, ranked by AUC separability",
        f"tab:{pair}_{domain}_auc_separability_ranking",
        digits,
    )


def export_group_tables_from_results(
    results,
    output_dir,
    metrics=INPUT_METRICS,
    digits=3,
    top_n=DEFAULT_TOP_N,
    idle_callback=None,
    evaluate_both_directions=True,
):
    out_dir = Path(output_dir)
    directories = {
        "spatial_raw": out_dir / "spatial" / "raw",
        "temporal_raw": out_dir / "temporal" / "raw",
        "spatial_cmp": out_dir / "spatial" / "comparisons_vs_control",
        "temporal_cmp": out_dir / "temporal" / "comparisons_vs_control",
        "spatial_fig": out_dir / "spatial" / "figures",
        "temporal_fig": out_dir / "temporal" / "figures",
    }
    plt.style.use(PLOT_STYLE)
    if out_dir.is_dir():
        shutil.rmtree(out_dir)
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    control_name = find_control_group(results)
    safe_control = safe_name(control_name)
    generated = []

    if idle_callback:
        idle_callback()

    _export_raw_tables(
        results,
        directories["spatial_raw"],
        directories["temporal_raw"],
        metrics,
        digits,
        generated,
    )
    if idle_callback:
        idle_callback()

    control_results = results[control_name]
    for group_name in sorted(results):
        if group_name == control_name:
            continue
        group_results = results[group_name]
        pair = f"{safe_name(group_name)}_vs_{safe_control}"
        domains = (
            (
                "spatial",
                SPATIAL_VARIABILITY_COLUMNS,
                SPATIAL_DESCRIPTOR_MAP,
                SPATIAL_SELECTED_METRICS,
                "RI_PI",
            ),
            (
                "temporal",
                TEMPORAL_VARIABILITY_COLUMNS,
                TEMPORAL_DESCRIPTOR_MAP,
                TEMPORAL_SELECTED_METRICS,
                "Nt_Neff",
            ),
        )
        for domain, columns, descriptor_map, selected_metrics, suffix in domains:
            directory = directories[f"{domain}_cmp"]
            _export_rankings(
                generated,
                control_results,
                group_results,
                columns,
                control_name,
                group_name,
                metrics,
                directory,
                pair,
                domain,
                top_n,
                digits,
            )
            _export_statistical_tables(
                generated,
                control_results,
                group_results,
                columns,
                descriptor_map,
                selected_metrics,
                control_name,
                group_name,
                metrics,
                directory,
                pair,
                domain,
                suffix,
                digits,
                evaluate_both_directions,
            )
        if idle_callback:
            idle_callback()

    html_path = save_html_report(out_dir)
    generated.append(html_path)
    return generated


def export_group_tables(
    zip_path,
    metrics=INPUT_METRICS,
    mode=SEGMENT_MODE,
    digits=3,
    top_n=DEFAULT_TOP_N,
    evaluate_both_directions=False,
):
    zip_path = Path(zip_path)
    out_dir = zip_path.parent / "Variability and heterogeneity"
    results = analyze_zip(zip_path, metrics=metrics, mode=mode)
    generated = export_group_tables_from_results(
        results,
        out_dir,
        metrics=metrics,
        digits=digits,
        top_n=top_n,
        evaluate_both_directions=evaluate_both_directions,
    )
    replace_folder_in_zip(
        zip_path,
        out_dir,
        arc_folder="Variability and heterogeneity",
    )
    if out_dir.is_dir():
        shutil.rmtree(out_dir)
    return generated
