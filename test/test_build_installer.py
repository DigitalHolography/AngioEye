from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BuildInstallerScriptTests(unittest.TestCase):
    def test_powershell_build_is_single_console_executable(self) -> None:
        script = (PROJECT_ROOT / "build_installer.ps1").read_text(encoding="utf-8")
        project = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn('hiddenimports += ["angio_eye", "cli", "launcher"]', script)
        self.assertIn("console=True", script)
        self.assertNotIn("AngioEyeCLI.exe", script)
        self.assertNotIn("angioeye-cli", project)

    def test_legacy_two_executable_build_files_are_removed(self) -> None:
        self.assertFalse((PROJECT_ROOT / "AngioEye.spec").exists())
        self.assertFalse((PROJECT_ROOT / "installer" / "AngioEye.iss").exists())
        self.assertFalse(
            (PROJECT_ROOT / "src" / "scripts" / "build_installer.py").exists()
        )


if __name__ == "__main__":
    unittest.main()
