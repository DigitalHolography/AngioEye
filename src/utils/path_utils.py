from pathlib import Path

from logger.logger_class import Logger


def conv_to_path(path: str | Path) -> Path:
    try:
        return Path(path) if isinstance(path, str) else path
    except Exception as e:
        Logger.error(f"Path '{path}' could not be converted !\n{e}")
        return Path()
