from pathlib import Path

import matplotlib
import numpy as np

from math_utils import clean_values

from .compute import safe_name
from .constants import PNG_PIL_KWARGS, SUMMARY_PVALUE_METRICS
from .statistics import get_descriptor_values_for_test

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


def descriptor_axis_label(descriptor_name, high_name):
    if high_name.startswith("CV_"):
        return descriptor_name
    return f"{descriptor_name} / |median metric value|"


def export_variability_value_plots(
    results,
    out_dir,
    descriptor_map,
    domain_name,
    metrics=SUMMARY_PVALUE_METRICS,
    dpi=300,
):
    """Export descriptor values by cohort using the values tested statistically."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    group_names = sorted(results.keys())
    rng = np.random.default_rng(12345)

    for metric_name in metrics:
        for descriptor_name, high_name in descriptor_map.items():
            group_values = []
            non_empty_group_names = []
            for group_name in group_names:
                values = clean_values(
                    get_descriptor_values_for_test(
                        results[group_name], metric_name, high_name
                    )
                )
                if values.size:
                    group_values.append(values)
                    non_empty_group_names.append(group_name)
            if not group_values:
                continue

            fig, axis = plt.subplots(
                figsize=(max(6, 1.2 * len(group_values)), 4.5),
                layout="constrained",
            )
            positions = np.arange(1, len(group_values) + 1)
            axis.boxplot(
                group_values,
                positions=positions,
                widths=0.45,
                showfliers=False,
            )
            for position, values in zip(positions, group_values, strict=True):
                jitter = rng.normal(loc=0.0, scale=0.045, size=len(values))
                axis.scatter(
                    np.full(len(values), position) + jitter,
                    values,
                    s=18,
                    alpha=0.75,
                )
            labels = [
                f"{name}\n(n={len(values)})"
                for name, values in zip(
                    non_empty_group_names, group_values, strict=True
                )
            ]
            axis.set_xticks(positions)
            axis.set_xticklabels(labels, rotation=0)
            axis.set_ylabel(descriptor_axis_label(descriptor_name, high_name))
            axis.set_xlabel("Cohort")
            axis.set_title(
                f"{domain_name.capitalize()} {descriptor_name} variability "
                f"for {metric_name}"
            )
            axis.grid(axis="y", alpha=0.25)

            filename = (
                f"{safe_name(domain_name)}_{safe_name(descriptor_name)}_"
                f"{safe_name(metric_name)}_by_group.png"
            )
            path = out_dir / filename
            fig.savefig(path, dpi=dpi, pil_kwargs=PNG_PIL_KWARGS)
            plt.close(fig)
            generated.append(path)
    return generated
