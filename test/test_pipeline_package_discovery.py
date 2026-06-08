import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class PipelinePackageDiscoveryTests(unittest.TestCase):
    def tearDown(self) -> None:
        for module_name in list(sys.modules):
            if module_name.startswith("pipelines.package_demo"):
                del sys.modules[module_name]

    def test_pipeline_catalog_discovers_package_entrypoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package_demo"
            package_dir.mkdir()
            (package_dir / "__init__.py").write_text(
                "\n".join(
                    [
                        "import numpy as np",
                        "from pipelines.core.base import (",
                        "    ProcessPipeline,",
                        "    ProcessResult,",
                        "    registerPipeline,",
                        ")",
                        "",
                        '@registerPipeline(name="package_demo")',
                        "class PackageDemoPipeline(ProcessPipeline):",
                        '    description = "Package-style test pipeline."',
                        "",
                        "    def run(self, h5file):",
                        '        return ProcessResult(metrics={"value": np.asarray(1)})',
                    ]
                ),
                encoding="utf-8",
            )

            with mock.patch.dict(os.environ, {"ANGIOEYE_PIPELINES_DIR": tmp}):
                import pipelines

                pipelines = importlib.reload(pipelines)
                available, missing = pipelines.load_pipeline_catalog()

        self.assertNotIn("package_demo", {pipeline.name for pipeline in missing})
        package_pipeline = next(
            pipeline for pipeline in available if pipeline.name == "package_demo"
        )
        self.assertEqual(
            "Package-style test pipeline.", package_pipeline.description
        )
        self.assertEqual(
            "PackageDemoPipeline", package_pipeline.pipeline_cls.__name__
        )


class OptionalRequirementsScannerTests(unittest.TestCase):
    def test_scanner_includes_package_entrypoint_required_deps(self):
        from scripts import gen_optional_reqs

        with tempfile.TemporaryDirectory() as tmp:
            pipelines_dir = Path(tmp)
            (pipelines_dir / "file_pipeline.py").write_text(
                "\n".join(
                    [
                        "from pipelines.core.base import registerPipeline",
                        "",
                        "@registerPipeline(",
                        '    name="file_pipeline",',
                        '    required_deps=["numpy>=1.0"],',
                        ")",
                        "class FilePipeline:",
                        "    pass",
                    ]
                ),
                encoding="utf-8",
            )

            package_dir = pipelines_dir / "package_pipeline"
            package_dir.mkdir()
            (package_dir / "__init__.py").write_text(
                "\n".join(
                    [
                        "from pipelines.core.base import registerPipeline",
                        "",
                        "@registerPipeline(",
                        '    name="package_pipeline",',
                        '    required_deps=["scipy>=1.10"],',
                        ")",
                        "class PackagePipeline:",
                        "    pass",
                    ]
                ),
                encoding="utf-8",
            )

            (package_dir / "helpers.py").write_text(
                'REQUIRED = "not-scanned"\n',
                encoding="utf-8",
            )

            entrypoints = list(
                gen_optional_reqs.iter_pipeline_entrypoints(pipelines_dir)
            )
            requirements = {
                req
                for path in entrypoints
                for req in gen_optional_reqs.parse_required_deps(path)
            }

        self.assertEqual(
            {"numpy>=1.0", "scipy>=1.10"},
            requirements,
        )


if __name__ == "__main__":
    unittest.main()
