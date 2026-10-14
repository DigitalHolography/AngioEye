from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Iterable, Mapping
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

APP_NAME = "AngioEye"
SETTINGS_FILENAME = "settings.json"
DEFAULT_SETTINGS_FILENAME = "default_settings.json"
LAST_BATCH_LOG_FILENAME = "last_batch_log.txt"
VERSION_PATTERN = re.compile(r'^version\s*=\s*"([^"]+)"\s*$')
INVALID_PATH_CHARS_PATTERN = re.compile(r'[<>:"/\\|?*]+')
DEFAULT_SELECTED_PIPELINES = {"HTML summary", "Pipeline Metrics Manifest"}
LEGACY_PIPELINE_NAMES = {
    "Variability and heterogeneity tables": "Variability and heterogeneity",
    "groups comparison dashboard": "Group comparison (Dashboard) - Waveform Shape Metrics",
    "groups comparison html": "Group comparison (HTML) - Waveform Shape Metrics",
}


def _read_version_from_pyproject(pyproject_path: Path) -> str | None:
    try:
        lines = pyproject_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    for line in lines:
        match = VERSION_PATTERN.match(line)
        if match:
            return match.group(1)
    return None


def app_version() -> str | None:
    env_version = os.getenv("ANGIOEYE_VERSION", "").strip()
    if env_version:
        return env_version

    try:
        return importlib_metadata.version(APP_NAME)
    except importlib_metadata.PackageNotFoundError:
        pass

    for root in _resource_roots():
        version = _read_version_from_pyproject(root / "pyproject.toml")
        if version:
            return version
    return None


def app_display_name() -> str:
    version = app_version()
    if not version:
        return APP_NAME
    display_version = version if version.lower().startswith("v") else f"v{version}"
    return f"{APP_NAME} {display_version}"


def _settings_subdir_name() -> str:
    version = app_version()
    if not version:
        return APP_NAME

    safe_version = INVALID_PATH_CHARS_PATTERN.sub("-", version).rstrip(" .")
    return safe_version or APP_NAME


def default_settings_path() -> Path:
    appdata = os.getenv("APPDATA") or os.getenv("LOCALAPPDATA")
    if appdata:
        base_dir = Path(appdata)
    else:
        xdg_config = os.getenv("XDG_CONFIG_HOME")
        base_dir = Path(xdg_config) if xdg_config else Path.home() / ".config"
    return base_dir / APP_NAME / _settings_subdir_name() / SETTINGS_FILENAME


def default_batch_log_path() -> Path:
    return default_settings_path().with_name(LAST_BATCH_LOG_FILENAME)


def _resource_roots() -> list[Path]:
    roots: list[Path] = []
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        roots.append(Path(frozen_root))
    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).resolve().parent)
    roots.append(Path(__file__).resolve().parents[1])
    roots.append(Path.cwd())
    return roots


def default_settings_template_path() -> Path | None:
    env_path = os.getenv("ANGIOEYE_DEFAULT_SETTINGS")
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.is_file():
            return candidate

    for root in _resource_roots():
        candidate = root / DEFAULT_SETTINGS_FILENAME
        if candidate.is_file():
            return candidate
    return None


def _load_settings_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def normalize_named_visibility(
    item_names: Iterable[str], stored_visibility: Mapping[str, bool] | None
) -> tuple[dict[str, bool], bool]:
    ordered_names = list(dict.fromkeys(item_names))
    clean_stored = {
        name: value
        for name, value in (stored_visibility or {}).items()
        if isinstance(name, str) and isinstance(value, bool)
    }

    if not clean_stored:
        return {name: True for name in ordered_names}, bool(ordered_names)

    visibility: dict[str, bool] = {}
    changed = False
    for name in ordered_names:
        if name in clean_stored:
            visibility[name] = clean_stored[name]
        else:
            visibility[name] = False
            changed = True

    if set(clean_stored) != set(visibility):
        changed = True
    return visibility, changed


def normalize_pipeline_visibility(
    pipeline_names: Iterable[str], stored_visibility: Mapping[str, bool] | None
) -> tuple[dict[str, bool], bool]:
    if not stored_visibility:
        ordered_names = list(dict.fromkeys(pipeline_names))
        return (
            {name: name in DEFAULT_SELECTED_PIPELINES for name in ordered_names},
            bool(ordered_names),
        )
    return normalize_named_visibility(pipeline_names, stored_visibility)


class AppSettingsStore:
    def __init__(
        self,
        path: Path | None = None,
        default_template_path: Path | None = None,
    ) -> None:
        self.path = path or default_settings_path()
        self.default_template_path = (
            default_template_path
            if default_template_path is not None
            else (default_settings_template_path() if path is None else None)
        )

    def load_defaults(self) -> dict[str, Any]:
        if self.default_template_path is None:
            return {}
        return _load_settings_file(self.default_template_path)

    def initialize_from_defaults(self) -> bool:
        if self.path.exists():
            return False
        defaults = self.load_defaults()
        if not defaults:
            return False
        self.save(defaults)
        return True

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            settings = self.load_defaults()
        else:
            settings = _load_settings_file(self.path)
        # The former per-file pipelines moved to EyeFlow; preserve the user's
        # batch analysis selection when loading preferences from that release.
        legacy_visibility = settings.pop("postprocess_visibility", None)
        if isinstance(legacy_visibility, dict):
            settings["pipeline_visibility"] = legacy_visibility
        visibility = settings.get("pipeline_visibility")
        if isinstance(visibility, dict):
            for legacy_name, current_name in LEGACY_PIPELINE_NAMES.items():
                if legacy_name in visibility:
                    value = visibility.pop(legacy_name)
                    visibility.setdefault(current_name, value)
        legacy_batching = settings.pop("postprocess_batching", None)
        if isinstance(legacy_batching, dict):
            settings.setdefault("pipeline_batching", legacy_batching)
        settings.pop("persist_source", None)
        settings.pop("trim_h5source", None)
        return settings

    def save(self, settings: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(dict(settings), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(self.path)

    def load_named_visibility(self, key: str) -> dict[str, bool]:
        raw_visibility = self.load().get(key, {})
        if not isinstance(raw_visibility, dict):
            return {}
        return {
            name: value
            for name, value in raw_visibility.items()
            if isinstance(name, str) and isinstance(value, bool)
        }

    def save_named_visibility(self, key: str, visibility: Mapping[str, bool]) -> None:
        settings = self.load()
        settings[key] = {
            name: bool(visibility[name]) for name in sorted(visibility, key=str.lower)
        }
        self.save(settings)

    def load_pipeline_visibility(self) -> dict[str, bool]:
        return self.load_named_visibility("pipeline_visibility")

    def save_pipeline_visibility(self, visibility: Mapping[str, bool]) -> None:
        self.save_named_visibility("pipeline_visibility", visibility)

    def load_ui_mode(self) -> str:
        mode = self.load().get("ui_mode")
        if isinstance(mode, str) and mode in {"advanced", "sandbox_advanced"}:
            return "advanced"
        return "minimal"

    def save_ui_mode(self, mode: str) -> None:
        settings = self.load()
        settings["ui_mode"] = (
            "advanced" if mode in {"advanced", "sandbox_advanced"} else "minimal"
        )
        self.save(settings)

    def load_processing_branch(self) -> str:
        value = self.load().get("processing_branch", "Processing")
        return (
            value
            if isinstance(value, str) and value in {"Processing", "ProcessingAlt"}
            else "Processing"
        )

    def save_processing_branch(self, processing_branch: str) -> None:
        if processing_branch not in {"Processing", "ProcessingAlt"}:
            raise ValueError("Choose Processing or ProcessingAlt.")
        settings = self.load()
        settings["processing_branch"] = processing_branch
        self.save(settings)

    def load_batch_execution(self) -> dict[str, Any]:
        batch_execution = self.load().get("batch_execution")
        if isinstance(batch_execution, dict):
            return batch_execution

        default_batch_execution = self.load_defaults().get("batch_execution")
        return (
            default_batch_execution if isinstance(default_batch_execution, dict) else {}
        )

    def load_pipeline_batching(self) -> dict[str, Any]:
        pipeline_batching = self.load().get("pipeline_batching")
        if isinstance(pipeline_batching, dict):
            return pipeline_batching

        default_pipeline_batching = self.load_defaults().get("pipeline_batching")
        return (
            default_pipeline_batching
            if isinstance(default_pipeline_batching, dict)
            else {}
        )
