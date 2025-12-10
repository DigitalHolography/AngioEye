import os
import sys

from src.colors.color_class import col

# ┌───────────────────────────────────┐
# │            TAGS_COLORS            │
# └───────────────────────────────────┘

tags_color = {
    "INFO": [col.bg.CYA, col.WHI, col.BOLD],
    "WARN": [col.bg.YEL, col.WHI, col.BOLD],
    "INFO ": [col.bg.CYA, col.WHI, col.BOLD],  # For alignment
    "WARN ": [col.bg.YEL, col.BLA, col.BOLD],  # For alignment
    "ERROR": [col.bg.RED, col.WHI, col.BOLD],
    "FATAL": [col.bg.RED, col.WHI, col.BOLD],
    "DEBUG": [col.bg.BLU, col.WHI, col.BOLD],
    "DOWNLOAD": [col.bg.PUR, col.WHI, col.BOLD],
    "DONE": [col.bg.GRE, col.BLA, col.BOLD],
    "TIME": [col.bg.BLU, col.WHI, col.BOLD],
    "FILESYSTEM": [col.bg.PUR, col.WHI, col.BOLD],
    "DATABASE": [col.bg.WHI, col.BLA, col.BOLD],
    "SETTINGS": [col.bg.BLU, col.WHI, col.BOLD],
    "SKIP": [col.bg.GRE, col.WHI, col.BOLD],
    "H5": [col.bg.PUR, col.WHI, col.PUR],
}

################################################################################

# Check for Python version
if sys.version_info < (3, 10):
    print(f"{col.BOLD}{col.RED}You are using a Python version before 3.10!")
    print("This could result in failure to load")
    print(f"{col.YEL}Current version {sys.version}")

# Allows for ANSI escape character processing on Windows
if os.name == "nt":
    os.system("")

################################################################################


def log_to_file(msg: str, file_path: str) -> None:
    """This will write the plain log message to a file

    Args:
        msg (str): the message to log
        file_path (str): the path to the log file
    """
    try:
        with open(file_path, "a") as f:
            f.write(msg + "\n")
    except Exception as e:
        print(
            f"{col.bg}{col.RED} ERROR {col.BLU} LOGGER {col.RES}{col.RED}Failed to log to file {file_path}: {e}"
        )


def log(msg: str, colors: list[str], end: str = "\n") -> None:
    for x in colors:
        print(x, end="")
    print(f"{msg}{col.RES}", end=end)


def log_tags(msg: str, tags: list[tuple[str, list[str]]]) -> None:
    for tag in tags:
        log(f" {tag[0]} ", tag[1], "")
    print("", msg)


def log_t(msg: str, tags: list[str] | str) -> None:
    """
    This function will print tags before the message with color codes defined in
    the tags_color global var.
    You should maybe use the Logger class.
    """
    if isinstance(tags, str):
        tags = [tags]
    for t in tags:
        # print(tags_color.get(t, []))
        if t in tags_color:
            log(f" {t} ", tags_color.get(t, []), "")
            # log(f" {t:<8} ", tags_color.get(t, []), "")
        else:
            log(f"[ {t} ]", [], "")
    print("", msg)


# Logger class
# @param    msg     The message to be printed
# @param    tags    Take the tag (or list of tags) to be printed before
class Logger:
    @staticmethod
    def info(msg: str, tags: list[str] | str = []) -> None:
        if isinstance(tags, str):
            tags = [tags]
        log_t(msg, ["INFO "] + tags)

    @staticmethod
    def warn(msg: str, tags: list[str] | str = []) -> None:
        if isinstance(tags, str):
            tags = [tags]
        log_t(msg, ["WARN "] + tags)

    @staticmethod
    def error(msg: str, tags: list[str] | str = []) -> None:
        if isinstance(tags, str):
            tags = [tags]
        log_t(msg, ["ERROR"] + tags)

    @staticmethod
    def debug(msg: str, tags: list[str] | str = []) -> None:
        if isinstance(tags, str):
            tags = [tags]
        log_t(msg, ["DEBUG"] + tags)

    @staticmethod
    def fatal(msg: str, tags: list[str] | str = [], raiseExeption: bool = True) -> None:
        if isinstance(tags, str):
            tags = [tags]
        log_t(msg, ["FATAL"] + tags)

        if raiseExeption:
            raise Exception(f"[FATAL] {tags} | {msg}")
