import os
import sys

from analysis.analysis_class import AnalysisClass
from colors.color_class import col
from logger.logger_class import Logger
from utils.export_results import save_womersley_results


def main():
    # Logger.info("Hello World!", "H5")
    # file_name = input("File path: ")
    # if len(file_name) == 0 or not os.path.exists(os.path.abspath(file_name)):
    #     Logger.error(f"File invalid: '{file_name}'")
    #     return

    # ranges = [(10, 90), (25, 75)]

    # df = AnalysisClass.generate_df(
    #     file_name,
    #     "/Array3D_FrequencyAnalysis/Womersley",
    #     ranges,
    # )
    # save_womersley_results(df, "3D", "output")
    # df = AnalysisClass.generate_df(
    #     file_name,
    #     "/Array2D_FrequencyAnalysis/Womersley",
    #     ranges,
    # )
    # save_womersley_results(df, "2D", "output")
    analyser = AnalysisClass()

    analyser.execute_from_file("./src/analysis/analysis_settings.json")


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
