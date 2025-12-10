import os
import sys

from src.colors.color_class import col
from src.logger.logger_class import Logger


def main():
    Logger.info("Hello World!", "H5")


def _check_py_ver():
    # Allows for ANSI escape character processing on Windows
    if os.name == "nt":
        os.system("")

    # Check for Python version
    if sys.version_info < (3, 10):
        print(f"{col.BOLD}{col.RED}You are using a Python version before 3.10!")
        print("This could result in failure to load")
        print(f"{col.YEL}Current version {sys.version}{col.RES}")


if __name__ == "__main__":
    _check_py_ver()
    main()
