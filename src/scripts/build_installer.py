from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEC_FILE = PROJECT_ROOT / "AngioEye.spec"
ISS_FILE = PROJECT_ROOT / "installer" / "AngioEye.iss"
DIST_DIR = PROJECT_ROOT / "dist"
GUI_EXE_NAME = "AngioEye.exe"
CLI_EXE_NAME = "AngioEyeCLI.exe"
ONEDIR_BUILD = DIST_DIR / "AngioEye"
PAYLOAD_DIR = PROJECT_ROOT / "build" / "installer_payload"
INSTALLER_OUTPUT_DIR = DIST_DIR
VERSION_PATTERN = re.compile(r'^version\s*=\s*"([^"]+)"\s*$')
INNO_SETUP_CANDIDATES = (
    Path.home() / "AppData" / "Local" / "Programs" / "Inno Setup 6" / "ISCC.exe",
    Path(r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe"),
    Path(r"C:\Program Files\Inno Setup 6\ISCC.exe"),
)
PAYLOAD_EXTRA_FILES = (
    PROJECT_ROOT / "LICENSE",
    PROJECT_ROOT / "THIRD_PARTY_NOTICES",
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "AngioEye.ico",
    PROJECT_ROOT / "default_settings.json",
    PROJECT_ROOT / "pyproject.toml",
)
EDITABLE_PACKAGE_DIRS = ("pipelines", "postprocess")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the AngioEye Windows installer with PyInstaller and Inno Setup."
    )
    parser.add_argument(
        "--skip-pyinstaller",
        action="store_true",
        help="Reuse the current dist output instead of rebuilding it with PyInstaller.",
    )
    parser.add_argument(
        "--iscc",
        type=Path,
        help="Optional full path to ISCC.exe.",
    )
    return parser.parse_args()


def _ensure_supported_python() -> None:
    if sys.version_info < (3, 10):  # noqa: UP036
        version = ".".join(str(part) for part in sys.version_info[:3])
        raise SystemExit(
            "build-installer must run with Python 3.10 or newer. "
            f"Current interpreter: {sys.executable} ({version})."
        )


def _read_version() -> str:
    pyproject_path = PROJECT_ROOT / "pyproject.toml"
    for line in pyproject_path.read_text(encoding="utf-8").splitlines():
        match = VERSION_PATTERN.match(line)
        if match:
            return match.group(1)
    raise RuntimeError(f"Could not read version from {pyproject_path}")


def _find_iscc(explicit_path: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path)

    env_override = os.environ.get("INNO_SETUP_COMPILER")
    if env_override:
        candidates.append(Path(env_override))

    for command_name in ("iscc.exe", "iscc"):
        resolved = shutil.which(command_name)
        if resolved:
            candidates.append(Path(resolved))

    candidates.extend(INNO_SETUP_CANDIDATES)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join(str(path) for path in candidates if path)
    raise FileNotFoundError(
        "Could not find ISCC.exe. Set INNO_SETUP_COMPILER, pass --iscc, "
        "or add Inno Setup to PATH.\n"
        f"Searched:\n{searched}"
    )


def _run_command(command: list[str | Path]) -> None:
    cmd = [str(part) for part in command]
    print(f"> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _onedir_build_targets() -> tuple[Path, Path]:
    return (ONEDIR_BUILD / GUI_EXE_NAME, ONEDIR_BUILD / CLI_EXE_NAME)


def _onefile_build_targets() -> tuple[Path, Path]:
    return (DIST_DIR / GUI_EXE_NAME, DIST_DIR / CLI_EXE_NAME)


def _clean_pyinstaller_outputs() -> None:
    for path in (ONEDIR_BUILD, *_onefile_build_targets()):
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def _run_pyinstaller() -> None:
    _clean_pyinstaller_outputs()
    _run_command([sys.executable, "-m", "PyInstaller", "--noconfirm", SPEC_FILE])


def _copy_tree_contents(source_dir: Path, destination_dir: Path) -> None:
    for child in source_dir.iterdir():
        target = destination_dir / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)


def _copy_editable_package_modules(package_name: str) -> None:
    source_dir = PROJECT_ROOT / "src" / package_name
    destination_dir = PAYLOAD_DIR / package_name
    destination_dir.mkdir(parents=True, exist_ok=True)
    for source_file in source_dir.glob("*.py"):
        if source_file.name == "__init__.py":
            continue
        shutil.copy2(source_file, destination_dir / source_file.name)


def _select_release_mode() -> str:
    build_modes = {
        "onedir": _onedir_build_targets(),
        "onefile": _onefile_build_targets(),
    }
    complete_modes: list[tuple[str, float]] = []
    partial_modes: list[tuple[str, list[Path]]] = []

    for mode_name, targets in build_modes.items():
        existing = [path for path in targets if path.is_file()]
        if len(existing) == len(targets):
            newest_target = max(path.stat().st_mtime for path in targets)
            complete_modes.append((mode_name, newest_target))
        elif existing:
            missing = [path for path in targets if not path.is_file()]
            partial_modes.append((mode_name, missing))

    if complete_modes:
        return max(complete_modes, key=lambda item: item[1])[0]

    if partial_modes:
        details = "; ".join(
            f"{mode_name} missing: {', '.join(str(path) for path in missing)}"
            for mode_name, missing in partial_modes
        )
        raise FileNotFoundError(f"Incomplete PyInstaller output. {details}")

    expected_modes = (
        ", ".join(str(path) for path in _onedir_build_targets()),
        ", ".join(str(path) for path in _onefile_build_targets()),
    )
    raise FileNotFoundError(
        "PyInstaller output not found. Expected either "
        f"one-dir files ({expected_modes[0]}) or one-file binaries "
        f"({expected_modes[1]})."
    )


def _prepare_payload() -> None:
    if PAYLOAD_DIR.exists():
        shutil.rmtree(PAYLOAD_DIR)
    PAYLOAD_DIR.mkdir(parents=True, exist_ok=True)

    release_mode = _select_release_mode()
    if release_mode == "onefile":
        for build_target in _onefile_build_targets():
            shutil.copy2(build_target, PAYLOAD_DIR / build_target.name)
    else:
        _copy_tree_contents(ONEDIR_BUILD, PAYLOAD_DIR)

    for package_name in EDITABLE_PACKAGE_DIRS:
        _copy_editable_package_modules(package_name)

    for extra_file in PAYLOAD_EXTRA_FILES:
        if extra_file.exists():
            shutil.copy2(extra_file, PAYLOAD_DIR / extra_file.name)


def _run_inno_setup(iscc_path: Path, app_version: str) -> None:
    INSTALLER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _run_command(
        [
            iscc_path,
            f"/DAppVersion={app_version}",
            f"/DPayloadDir={PAYLOAD_DIR}",
            f"/DOutputDir={INSTALLER_OUTPUT_DIR}",
            ISS_FILE,
        ]
    )


def main() -> None:
    args = _parse_args()
    _ensure_supported_python()

    if not SPEC_FILE.exists():
        raise SystemExit(f"PyInstaller spec file not found: {SPEC_FILE}")
    if not ISS_FILE.exists():
        raise SystemExit(f"Inno Setup script not found: {ISS_FILE}")

    iscc_path = _find_iscc(args.iscc)
    app_version = _read_version()

    if not args.skip_pyinstaller:
        _run_pyinstaller()

    _prepare_payload()
    _run_inno_setup(iscc_path, app_version)

    installer_name = INSTALLER_OUTPUT_DIR / f"AngioEye-setup-{app_version}.exe"
    print(f"Installer created at {installer_name}")


if __name__ == "__main__":
    main()
