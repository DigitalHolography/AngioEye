import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cli  # noqa: E402
from pipelines import PipelineDescriptor  # noqa: E402


class StatisticalCliTests(unittest.TestCase):
    def setUp(self):
        self.descriptor = PipelineDescriptor(
            name="HTML summary", description="", available=True
        )

    def test_selection_supports_list_literals_and_names_files(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "names.txt"
            path.write_text(
                "# comment\nHTML summary\nPipeline Metrics Manifest\n", encoding="utf-8"
            )
            names = cli._selected_names([str(path), '["HTML summary", "Other"]'])
            self.assertEqual(
                ["HTML summary", "Pipeline Metrics Manifest", "Other"], names
            )
            self.assertEqual(
                ["HTML summary", "Other"],
                cli._selected_names(['["HTML summary",', '"Other"]']),
            )

    def test_cli_passes_multiple_inputs_and_processing_branch(self):
        with (
            mock.patch.object(
                cli, "load_pipeline_catalog", return_value=([self.descriptor], [])
            ),
            mock.patch.object(
                cli, "run_batch", return_value=SimpleNamespace(failures=[])
            ) as run,
        ):
            status = cli.main(
                [
                    "--data",
                    "a.holo",
                    "b.holo",
                    "--pipelines",
                    "HTML summary",
                    "--processing-branch",
                    "ProcessingAlt",
                    "--output",
                    "results",
                    "--zip",
                ]
            )
        self.assertEqual(0, status)
        self.assertEqual(["a.holo", "b.holo"], run.call_args.args[0])
        self.assertEqual([self.descriptor], run.call_args.args[1])
        self.assertEqual(Path("results"), run.call_args.args[2])
        self.assertEqual("ProcessingAlt", run.call_args.kwargs["processing_branch"])
        self.assertTrue(run.call_args.kwargs["zip_outputs"])

    def test_cli_returns_failure_on_unknown_tool_or_failed_batch(self):
        with (
            mock.patch.object(
                cli, "load_pipeline_catalog", return_value=([self.descriptor], [])
            ),
            mock.patch.object(cli, "_log_cli"),
            mock.patch.object(
                cli, "run_batch", return_value=SimpleNamespace(failures=["failed"])
            ) as run,
        ):
            self.assertEqual(1, cli.run_cli("sample.holo", ["old metric calculator"]))
            run.assert_not_called()
            self.assertEqual(1, cli.run_cli("sample.holo", ["HTML summary"]))

    def test_default_selection_uses_migrated_settings(self):
        with (
            mock.patch.object(
                cli, "load_pipeline_catalog", return_value=([self.descriptor], [])
            ),
            mock.patch.object(cli, "AppSettingsStore") as store,
            mock.patch.object(
                cli, "run_batch", return_value=SimpleNamespace(failures=[])
            ) as run,
        ):
            store.return_value.load_pipeline_visibility.return_value = {
                "HTML summary": True
            }
            self.assertEqual(0, cli.run_cli("sample.holo"))
            self.assertEqual([self.descriptor], run.call_args.args[1])
            self.assertIsNone(run.call_args.args[2])

    def test_list_pipelines_runs_without_input(self):
        with mock.patch.object(cli, "print") as printer:
            self.assertEqual(0, cli.main(["--list-pipelines"]))
        self.assertTrue(printer.called)

    def test_cli_rejects_direct_exports_and_archives(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with (
                mock.patch.object(
                    cli, "load_pipeline_catalog", return_value=([self.descriptor], [])
                ),
                mock.patch.object(cli, "_log_cli") as log,
            ):
                for name in ("sample_EF.h5", "sample_EF.hdf5", "cohort.zip"):
                    with self.subTest(name=name):
                        source = root / name
                        source.touch()
                        self.assertEqual(
                            1, cli.run_cli(source, ["HTML summary"], root / "results")
                        )
                        self.assertIn(".holo", log.call_args.args[0])
                        self.assertFalse((root / "results").exists())

    def test_cli_reports_expected_result_for_unprocessed_hologram(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            holo = root / "sample.holo"
            holo.touch()
            with (
                mock.patch.object(
                    cli, "load_pipeline_catalog", return_value=([self.descriptor], [])
                ),
                mock.patch.object(cli, "_log_cli") as log,
            ):
                self.assertEqual(
                    1, cli.run_cli(holo, ["HTML summary"], root / "results")
                )
                self.assertIn("sample_EF.h5", log.call_args.args[0])
                self.assertIn("EyeFlow", log.call_args.args[0])
                self.assertFalse((root / "results").exists())


if __name__ == "__main__":
    unittest.main()
