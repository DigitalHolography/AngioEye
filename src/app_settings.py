from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

APP_NAME = "AngioEye"
SETTINGS_FILENAME = "settings.json"


def default_settings_path() -> Path:
    appdata = os.getenv("APPDATA") or os.getenv("LOCALAPPDATA")
    if appdata:
        base_dir = Path(appdata)
    else:
        xdg_config = os.getenv("XDG_CONFIG_HOME")
        base_dir = Path(xdg_config) if xdg_config else Path.home() / ".config"
    return base_dir / APP_NAME / SETTINGS_FILENAME


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
    return normalize_named_visibility(pipeline_names, stored_visibility)


def normalize_postprocess_visibility(
    postprocess_names: Iterable[str], stored_visibility: Mapping[str, bool] | None
) -> tuple[dict[str, bool], bool]:
    return normalize_named_visibility(postprocess_names, stored_visibility)


class AppSettingsStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_settings_path()

    def load(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

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

    def load_postprocess_visibility(self) -> dict[str, bool]:
        return self.load_named_visibility("postprocess_visibility")

    def save_postprocess_visibility(self, visibility: Mapping[str, bool]) -> None:
        self.save_named_visibility("postprocess_visibility", visibility)
