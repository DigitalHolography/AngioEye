# ruff: noqa: E402

import csv
import sys
import tempfile
import unittest
import zipfile
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from pathlib import Path
from unittest import mock

import h5py
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipelines.core.base import PipelineContext
from pipelines.groups_comparison_dashboard_wfsm import GraphicsDashboardPipeline
from pipelines.groups_comparison_html_wfsm import GroupsComparisonHTMLPipeline
from pipelines.utils import groups_comparison_dashboard, groups_comparison_html
from pipelines.variability_heterogeneity import run as variability_run

scoring_run = import_module("pipelines.composite_scoring.run")


def _write_metrics(path: Path, value: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        root = "/AngioEye/Processing/waveform_shape_metrics"
        for vessel in ("artery", "vein"):
            for representation in ("raw", "bandlimited"):
                h5.create_dataset(
                    f"{root}/{vessel}/global/{representation}/RI",
                    data=[value, value],
                )
        h5.create_dataset(
            f"{root}/artery/by_segment/bandlimited_segment/RI",
            data=np.arange(8, dtype=float).reshape(2, 2, 2) + value,
        )
    return path


class PipelineOutputRootTests(unittest.TestCase):
    def test_context_preserves_default_batch_root(self) -> None:
        root = Path("outputs")
        context = PipelineContext(root, (), (), Path("inputs"), False)
        self.assertEqual(root, context.batch_root)

    def test_group_reports_archive_acquisitions_and_write_separate_reports(
        self,
    ) -> None:
        cases = (
            (GroupsComparisonHTMLPipeline, groups_comparison_html, "HTML"),
            (GraphicsDashboardPipeline, groups_comparison_dashboard, "Dashboard"),
        )
        for pipeline_type, utility, report_kind in cases:
            with (
                self.subTest(report_kind=report_kind),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                report_dir = root / "cohort-results"
                report_dir.mkdir()
                paths = tuple(
                    _write_metrics(
                        root / cohort / stem / f"{stem}_AE" / "h5" / f"{stem}_AE.h5",
                        value,
                    )
                    for cohort, stem, value in (
                        ("control", "a", 1.0),
                        ("patient", "b", 2.0),
                    )
                )
                folder_name = (
                    f"group comparison ({report_kind}) - Waveform Shape Metrics"
                )
                source_marker = root / folder_name / "source.txt"
                source_marker.parent.mkdir()
                source_marker.write_text("preserve", encoding="utf-8")
                context = PipelineContext(
                    report_dir, paths, (), root, False, analysis_root=root
                )

                def save_dashboard(
                    *args,
                    report_kind=report_kind,
                    root=root,
                    paths=paths,
                    folder_name=folder_name,
                    **kwargs,
                ):
                    archive_path = args[0] if report_kind == "HTML" else args[1]
                    with zipfile.ZipFile(archive_path, "a") as archive:
                        self.assertEqual(
                            {path.relative_to(root).as_posix() for path in paths},
                            set(archive.namelist()),
                        )
                        archive.writestr(f"{folder_name}/metric.png", b"png")
                        if report_kind == "HTML":
                            archive.writestr(
                                "waveform_shape_metrics_group_comparison.html", "report"
                            )

                with mock.patch.object(
                    utility, "save_dashboard", side_effect=save_dashboard
                ):
                    result = pipeline_type().run(context)

                self.assertEqual("preserve", source_marker.read_text(encoding="utf-8"))
                self.assertTrue((report_dir / folder_name / "metric.png").is_file())
                self.assertTrue(
                    all(
                        Path(path).is_relative_to(report_dir)
                        for path in result.generated_paths
                    )
                )

    def test_composite_scoring_keeps_cohorts_when_reports_are_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_dir = root / "cohort-results"
            report_dir.mkdir()
            paths = tuple(
                _write_metrics(
                    root / cohort / stem / f"{stem}_AE" / "h5" / f"{stem}_AE.h5",
                    value,
                )
                for cohort, stem, value in (
                    ("control", "a", 0.2),
                    ("control", "b", 0.3),
                    ("patient", "c", 0.7),
                    ("patient", "d", 0.8),
                )
            )
            context = PipelineContext(
                report_dir, paths, (), root, False, analysis_root=root
            )

            with mock.patch.object(
                scoring_run, "write_score_plots", return_value=[]
            ) as plots:
                result = scoring_run.run_composite_scoring(context)

            calibration = (
                report_dir / "composite_scoring" / "optimal_split_calibration.csv"
            )
            with calibration.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(
                {("control", "patient")},
                {(row["control_group"], row["pathology_group"]) for row in rows},
            )
            contributions = (
                report_dir / "composite_scoring" / "metric_contributions_by_subject.csv"
            )
            with contributions.open(encoding="utf-8", newline="") as stream:
                cohorts = {row["cohort"] for row in csv.DictReader(stream)}
            self.assertEqual({"control", "patient"}, cohorts)
            self.assertEqual(report_dir, plots.call_args.args[1])
            self.assertTrue(
                all(Path(path).is_file() for path in result.generated_paths)
            )

    def test_variability_reports_preserve_source_folder_and_cohorts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_dir = root / "cohort-results"
            report_dir.mkdir()
            paths = tuple(
                _write_metrics(
                    root / cohort / stem / f"{stem}_AE" / "h5" / f"{stem}_AE.h5",
                    value,
                )
                for cohort, stem, value in (
                    ("control", "a", 1.0),
                    ("patient", "b", 2.0),
                )
            )
            source_marker = root / "Variability and heterogeneity" / "source.txt"
            source_marker.parent.mkdir()
            source_marker.write_text("preserve", encoding="utf-8")
            context = PipelineContext(
                report_dir, paths, (), root, False, analysis_root=root
            )

            def export_tables(results, table_dir, **kwargs):
                self.assertEqual({"control", "patient"}, set(results))
                self.assertEqual(
                    report_dir / "Variability and heterogeneity", table_dir
                )
                table_dir.mkdir()
                output = table_dir / "cohorts.txt"
                output.write_text("\n".join(sorted(results)), encoding="utf-8")
                return [output]

            with (
                mock.patch.object(
                    variability_run, "ProcessPoolExecutor", ThreadPoolExecutor
                ),
                mock.patch.object(variability_run, "AppSettingsStore") as settings,
                mock.patch.object(
                    variability_run,
                    "export_group_tables_from_results",
                    side_effect=export_tables,
                ),
            ):
                settings.return_value.load_batch_execution.return_value = {
                    "process_workers": 1
                }
                settings.return_value.load_pipeline_batching.return_value = {}
                result = variability_run.VariabilityHeterogeneityPipeline().run(context)

            self.assertEqual("preserve", source_marker.read_text(encoding="utf-8"))
            self.assertEqual(
                [str(report_dir / "Variability and heterogeneity" / "cohorts.txt")],
                result.generated_paths,
            )
