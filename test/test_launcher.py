import sys
import tempfile
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import launcher


class LauncherTests(unittest.TestCase):
    def test_find_checkout_src_from_nested_repo_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            src_dir = repo_root / "src"
            nested_dir = repo_root / "sub" / "dir"
            src_dir.mkdir()
            nested_dir.mkdir(parents=True)
            (repo_root / "pyproject.toml").write_text("[project]\nname='AngioEye'\n")
            (src_dir / "angio_eye.py").write_text("def main():\n    return 'ok'\n")

            result = launcher._find_checkout_src(
                "angio_eye.py",
                start_dir=nested_dir,
            )

            self.assertEqual(src_dir, result)

    def test_call_entry_prefers_checkout_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            src_dir = repo_root / "src"
            src_dir.mkdir()
            (repo_root / "pyproject.toml").write_text("[project]\nname='AngioEye'\n")
            (src_dir / "fake_entry.py").write_text(
                "def main(value=None):\n"
                "    return ('checkout', value)\n",
                encoding="utf-8",
            )

            result = launcher._call_entry(
                "fake_entry",
                "fake_entry.py",
                "main",
                "value",
                start_dir=repo_root,
            )

            self.assertEqual(("checkout", "value"), result)

    def test_call_entry_falls_back_to_imported_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            module_dir = Path(tmp_dir)
            outside_dir = module_dir / "outside"
            outside_dir.mkdir()
            (module_dir / "fallback_entry.py").write_text(
                "def main():\n"
                "    return 'installed'\n",
                encoding="utf-8",
            )
            sys.path.insert(0, str(module_dir))
            sys.modules.pop("fallback_entry", None)
            try:
                result = launcher._call_entry(
                    "fallback_entry",
                    "fallback_entry.py",
                    "main",
                    start_dir=outside_dir,
                )
            finally:
                sys.path.remove(str(module_dir))
                sys.modules.pop("fallback_entry", None)

            self.assertEqual("installed", result)


if __name__ == "__main__":
    unittest.main()
