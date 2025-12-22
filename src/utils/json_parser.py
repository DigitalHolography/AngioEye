import json
from pathlib import Path
from typing import Any

from logger.logger_class import Logger
from utils.path_utils import conv_to_path


class JsonParser:
    def __init__(self, file_path: str | Path) -> None:
        self.file_path = file_path

    def get_value(self, element_path: str, default_value=None):
        return JsonParser.get_value_file(
            self.file_path, element_path, default_value
        )

    def set_value(self, element_path: str, value):
        return JsonParser.set_value_file(self.file_path, element_path, value)

    def validate(self) -> bool:
        """See `JsonParser.validate_file()`

        Returns:
            bool: True if the json is valid, False otherwise
        """
        return JsonParser.validate_file(self.file_path)

    # +======================================================================+ #
    # |                            STATIC METHODS                            | #
    # +======================================================================+ #

    @staticmethod
    def get_value_file(
        file_path: str | Path,
        element_path: str,
        default_value: Any = None,
        silence_logs: bool = False,
    ) -> Any:
        """Get the value inside a JSON file

        Args:
            file_path (Path): JSON file path
            element_path (str): The Path inisde of the JSON (separated with ".")
            default_value (_type_, optional): Default return value.
                                              Defaults to None.
            silence_logs (bool, optional): Silence all possible warns/errors.
                                           Defaults to False.

        Returns:
            any: Returns the value linked to the JSON path
        """
        file_path = conv_to_path(file_path)

        if not file_path.exists():
            if not silence_logs:
                Logger.error(f"File not found ! {file_path}")

            return default_value

        data = JsonParser.load_json_file(file_path)
        if not data:
            return default_value

        try:
            keys = element_path.split(".")
            current = data

            for key in keys:
                if isinstance(current, dict):
                    current = current.get(key)
                elif isinstance(current, list):
                    if not key.isdigit():
                        if not silence_logs:
                            Logger.error(
                                f"Trying to access array with key '{key}'",
                                "JSON",
                            )

                        return default_value
                    if int(key) >= len(current):
                        if not silence_logs:
                            Logger.error(
                                "Trying to access array of length "
                                f"{len(current)} with idx: '{key}'",
                                "JSON",
                            )

                        return default_value

                    current = current[int(key)]
                else:
                    return default_value

            return current
        except Exception as e:
            if not silence_logs:
                Logger.error(
                    f"Something went wrong while accessing data !\n{e}"
                )
            return default_value

    @staticmethod
    def set_value_file(
        file_path: str | Path,
        element_path: str,
        value: Any,
        create_file: bool = False,
        create_missing_entry: bool = True,
    ) -> bool:
        """Write inside the JSON file a data

        Args:
            file_path (Path): The Path of the file
            element_path (str): The Path inisde of the JSON (separated with ".")
            value (_type_): The value to set
            create_file (bool, optional): Whether creates the file or not.
                                          Defaults to False.
            create_missing_entry (bool, optional): Whether creates entry.
                                                   Defaults to True.

        Returns:
            bool: True if successful
        """
        file_path = conv_to_path(file_path)

        if not file_path.exists():
            if create_file:
                Logger.info(
                    f"File was created (as it was not found): '{file_path}'",
                    ["JSON", "FILESYSTEM"],
                )
                file_path.parent.mkdir(parents=True, exist_ok=True)
                current_data = {}
            else:
                Logger.error(
                    "File was not found !\nUse 'create_file' = True "
                    f"for file auto creation.\n'{file_path}'",
                    ["JSON", "FILESYSTEM"],
                )
                return False
        else:
            current_data = JsonParser.load_json_file(file_path)
            if not current_data:
                return False

        keys = element_path.split(".")
        current = current_data

        for i, key in enumerate(keys[:-1]):
            if key not in current or not isinstance(current[key], dict):
                if not create_missing_entry:
                    Logger.error(
                        f"Entry '{key}' in '{element_path}' "
                        f"was not found in {file_path}"
                    )
                    return False

                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(current_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            Logger.error(
                f"Error while Writting JSON info !\n{e}", ["JSON", "FILESYSTEM"]
            )
            return False

        return True

    @staticmethod
    def validate_file(file_path: str | Path) -> bool:
        """Will return wether the json is valid inside file_path

        Will only log Error on `FileNotFoundError` and `PermissionError`

        Args:
            file_path (Path): The Path to the file

        Returns:
            bool: True if the json is valid, False otherwise
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                json.load(f)
                return True
        except (FileNotFoundError, PermissionError) as e:
            Logger.error(
                f"File not found (or missing permissions): {file_path}\n{e}",
                ["JSON", "FILESYSTEM"],
            )
            return False
        except Exception:
            return False

    @staticmethod
    def load_json_file(file_path: str | Path, encoding: str = "utf-8") -> Any:
        file_path = conv_to_path(file_path)

        if not file_path.exists():
            Logger.error(
                f"File '{file_path}' not found !", ["JSON", "FILESYSTEM"]
            )
            return None

        try:
            with open(file_path, "r", encoding=encoding) as f:
                return json.load(f)
        except (FileNotFoundError, PermissionError) as e:
            Logger.error(
                f"File not found (or missing permissions): {file_path}\n{e}",
                ["JSON", "FILESYSTEM"],
            )
            return None
        except Exception as e:
            Logger.error(f"Failed to load the JSON !\n{e}", "JSON")
            return None
