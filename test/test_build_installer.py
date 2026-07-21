from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BuildInstallerScriptTests(unittest.TestCase):
    def test_powershell_build_defaults_to_gui_and_supports_console_variant(self) -> None:
        script = (PROJECT_ROOT / "build_installer.ps1").read_text(encoding="utf-8")
        project = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn('hiddenimports += ["angio_eye", "cli", "launcher"]', script)
        self.assertIn("[switch]$Console", script)
        self.assertIn('$pyInstallerConsole = "False"', script)
        self.assertIn('$pyInstallerConsole = "True"', script)
        self.assertIn("console=$pyInstallerConsole", script)
        self.assertNotIn("AngioEyeCLI.exe", script)
        self.assertIn('angioeye-cli = "launcher:cli_main"', project)

    def test_legacy_two_executable_build_files_are_removed(self) -> None:
        self.assertFalse((PROJECT_ROOT / "AngioEye.spec").exists())
        self.assertFalse((PROJECT_ROOT / "installer" / "AngioEye.iss").exists())
        self.assertFalse(
            (PROJECT_ROOT / "src" / "scripts" / "build_installer.py").exists()
        )


if __name__ == "__main__":
    unittest.main()
