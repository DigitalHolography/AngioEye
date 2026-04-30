import importlib
import sys
import tempfile
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

build_installer = importlib.import_module("scripts.build_installer")


class BuildInstallerTests(unittest.TestCase):
    def _patch_module_globals(self, **replacements: object) -> None:
        originals = {
            name: getattr(build_installer, name) for name in replacements
        }

        for name, value in replacements.items():
            setattr(build_installer, name, value)

        def _restore() -> None:
            for name, value in originals.items():
                setattr(build_installer, name, value)

        self.addCleanup(_restore)

    def test_prepare_payload_copies_both_onefile_binaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            dist_dir = repo_root / "dist"
            payload_dir = repo_root / "build" / "installer_payload"
            onedir_build = dist_dir / "AngioEye"
            src_dir = repo_root / "src"
            pipelines_dir = src_dir / "pipelines"
            postprocess_dir = src_dir / "postprocess"
            extra_files = (
                repo_root / "LICENSE",
                repo_root / "README.md",
                repo_root / "THIRD_PARTY_NOTICES",
                repo_root / "AngioEye.ico",
                repo_root / "default_settings.json",
            )

            dist_dir.mkdir(parents=True)
            src_dir.mkdir()
            pipelines_dir.mkdir()
            postprocess_dir.mkdir()

            (dist_dir / "AngioEye.exe").write_text("gui", encoding="utf-8")
            (dist_dir / "AngioEyeCLI.exe").write_text("cli", encoding="utf-8")
            (pipelines_dir / "__init__.py").write_text("", encoding="utf-8")
            (pipelines_dir / "custom_pipeline.py").write_text(
                "# pipeline\n",
                encoding="utf-8",
            )
            (postprocess_dir / "__init__.py").write_text("", encoding="utf-8")
            (postprocess_dir / "custom_postprocess.py").write_text(
                "# postprocess\n",
                encoding="utf-8",
            )

            for extra_file in extra_files:
                extra_file.write_text(extra_file.name, encoding="utf-8")

            self._patch_module_globals(
                PROJECT_ROOT=repo_root,
                DIST_DIR=dist_dir,
                ONEDIR_BUILD=onedir_build,
                PAYLOAD_DIR=payload_dir,
                PAYLOAD_EXTRA_FILES=extra_files,
                EDITABLE_PACKAGE_DIRS=("pipelines", "postprocess"),
            )

            build_installer._prepare_payload()

            self.assertTrue((payload_dir / "AngioEye.exe").is_file())
            self.assertTrue((payload_dir / "AngioEyeCLI.exe").is_file())
            self.assertTrue((payload_dir / "pipelines" / "custom_pipeline.py").is_file())
            self.assertTrue(
                (payload_dir / "postprocess" / "custom_postprocess.py").is_file()
            )
            self.assertFalse((payload_dir / "pipelines" / "__init__.py").exists())
            self.assertTrue((payload_dir / "README.md").is_file())

    def test_prepare_payload_rejects_partial_onefile_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            dist_dir = repo_root / "dist"
            payload_dir = repo_root / "build" / "installer_payload"

            dist_dir.mkdir(parents=True)
            (dist_dir / "AngioEye.exe").write_text("gui", encoding="utf-8")

            self._patch_module_globals(
                PROJECT_ROOT=repo_root,
                DIST_DIR=dist_dir,
                ONEDIR_BUILD=dist_dir / "AngioEye",
                PAYLOAD_DIR=payload_dir,
                PAYLOAD_EXTRA_FILES=(),
                EDITABLE_PACKAGE_DIRS=(),
            )

            with self.assertRaises(FileNotFoundError) as ctx:
                build_installer._prepare_payload()

            self.assertIn("AngioEyeCLI.exe", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
