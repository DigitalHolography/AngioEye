from pathlib import Path

import numpy as np

from input_output.hdf5_schema import safe_h5_key
from postprocess.core.grouped_batch import build_group_order

from .dataclasses import ScoreRecord
from .metrics import REPRESENTATIONS


def write_score_plots(
    records: list[ScoreRecord],
    output_dir: Path,
) -> list[str]:
    if not records:
        return []

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    png_dir = output_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    created_paths: list[str] = []

    for representation in REPRESENTATIONS:
        representation_records = [
            record
            for record in records
            if record.representation == representation
        ]
        if not representation_records:
            continue
        created_paths.append(
            _plot_score_violin_by_cohort(
                representation_records,
                score_name="RWAS",
                value_getter=lambda record: record.rwas,
                representation=representation,
                png_dir=png_dir,
                plt=plt,
            )
        )
        created_paths.append(
            _plot_score_violin_by_cohort(
                representation_records,
                score_name="RWAS4",
                value_getter=lambda record: record.rwas4,
                representation=representation,
                png_dir=png_dir,
                plt=plt,
            )
        )
        created_paths.append(
            _plot_score_histogram(
                representation_records,
                representation=representation,
                png_dir=png_dir,
                plt=plt,
            )
        )

    return created_paths


def _plot_score_violin_by_cohort(
    records: list[ScoreRecord],
    *,
    score_name: str,
    value_getter,
    representation: str,
    png_dir: Path,
    plt,
) -> str:
    cohorts = build_group_order({record.cohort for record in records})

    fig, ax = plt.subplots(figsize=(max(6.4, len(cohorts) * 1.3), 4.9))
    cmap = plt.get_cmap("tab10")
    for cohort_index, cohort in enumerate(cohorts):
        cohort_records = [
            record for record in records if record.cohort == cohort
        ]
        if not cohort_records:
            continue

        position = cohort_index
        color = cmap(cohort_index % 10)
        values = np.asarray(
            [value_getter(record) for record in cohort_records],
            dtype=float,
        )
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue

        if values.size > 1 and float(np.nanmax(values) - np.nanmin(values)) > 0:
            parts = ax.violinplot(
                [values],
                positions=[position],
                widths=0.78,
                showmeans=False,
                showmedians=True,
                showextrema=False,
            )
            for body in parts["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor("black")
                body.set_alpha(0.34)
                body.set_linewidth(0.8)
            if "cmedians" in parts:
                parts["cmedians"].set_color("black")
                parts["cmedians"].set_linewidth(1.4)
        else:
            ax.scatter(
                [position],
                [float(values[0])],
                color=color,
                marker="_",
                s=520,
                linewidths=2.0,
                zorder=3,
            )

        x_values = [
            position + _centered_jitter(index, values.size, width=0.26)
            for index in range(values.size)
        ]
        ax.scatter(
            x_values,
            values,
            color=color,
            alpha=0.42,
            edgecolors="black",
            linewidths=0.25,
            s=24,
            zorder=4,
        )

        q1, median, q3 = np.percentile(values, [25, 50, 75])
        ax.vlines(position, q1, q3, color="black", linewidth=2.0, zorder=5)
        ax.scatter(
            [position],
            [median],
            color="white",
            edgecolors="black",
            linewidths=0.8,
            s=38,
            zorder=6,
        )

    ax.set_title(f"{score_name} concentration by cohort ({representation})")
    ax.set_xlabel("Cohort")
    ax.set_ylabel(score_name)
    ax.set_xticks(range(len(cohorts)))
    ax.set_xticklabels(
        [
            f"{cohort}\nn={sum(1 for record in records if record.cohort == cohort)}"
            for cohort in cohorts
        ],
    )
    ax.grid(axis="y", alpha=0.25)
    if score_name == "RWAS":
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_ylabel("RWAS (symlog)")
    else:
        ax.set_yticks(range(5))
    fig.tight_layout()

    output_path = png_dir / (
        f"composite_scoring_{safe_h5_key(representation)}_"
        f"{safe_h5_key(score_name)}_violin_by_cohort.png"
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)


def _centered_jitter(index: int, count: int, *, width: float) -> float:
    if count <= 1:
        return 0.0
    slots = min(count, 31)
    slot = index % slots
    return ((slot / max(slots - 1, 1)) - 0.5) * width


def _plot_score_histogram(
    records: list[ScoreRecord],
    *,
    representation: str,
    png_dir: Path,
    plt,
) -> str:
    cohort_values: dict[str, list[float]] = {}
    for record in records:
        value = float(record.rwas4)
        if np.isfinite(value):
            cohort_values.setdefault(record.cohort, []).append(value)

    fig, ax = plt.subplots(figsize=(max(6.4, len(cohort_values) * 1.25), 4.9))
    cmap = plt.get_cmap("tab10")

    scores = np.arange(5)
    bar_count = max(len(cohort_values), 1)
    bar_width = min(0.8 / bar_count, 0.35)
    for cohort_index, (cohort, values) in enumerate(cohort_values.items()):
        counts = np.bincount(
            np.asarray(values, dtype=int),
            minlength=5,
        )[:5]
        offset = (cohort_index - (bar_count - 1) / 2) * bar_width
        ax.bar(
            scores + offset,
            counts,
            width=bar_width * 0.92,
            color=cmap(cohort_index % 10),
            edgecolor="black",
            linewidth=0.8,
            label=f"{cohort} (n={len(values)})",
        )

    if not cohort_values:
        ax.text(
            0.5,
            0.5,
            "No finite scores",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

    ax.set_title(f"RWAS4 score distribution by cohort ({representation})")
    ax.set_xlabel("RWAS4")
    ax.set_ylabel("Count")
    ax.set_xticks(range(5))
    ax.grid(axis="y", alpha=0.25)
    if cohort_values:
        ax.legend(frameon=False)
    fig.tight_layout()

    output_path = png_dir / (
        f"composite_scoring_{safe_h5_key(representation)}_"
        "rwas4_histogram_by_cohort.png"
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)
