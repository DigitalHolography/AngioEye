import importlib
import json
from pathlib import Path
from typing import Any, Callable

from colors.color_class import col
from logger.logger_class import Logger
from utils.json_parser import JsonParser
from utils.path_utils import conv_to_path


class AnalysisClass:
    def __init__(
        self,
        analysers_module: str = "analysis.analysers",
        auto_reload: bool = False,
    ):
        self.analysers_module = analysers_module
        self.auto_reload = auto_reload
        self.loaded_modules = {}

        # Add analysers directory to Python path
        # analysers_path = Path(analysers_dir).parent
        # if str(analysers_path) not in sys.path:
        #     sys.path.insert(0, str(analysers_path))

    def _import_module(self, module_name: str):
        """Import a module from the analysers directory"""
        full_module_path = f"{self.analysers_module}.{module_name}"
        return importlib.import_module(full_module_path)

    def get_function(self, func_name: str) -> Callable | None:
        """Get a function from a module, trying various naming conventions

        Args:
            func_name (str): The name of the function

        Returns:
            Callable | None: The function
        """
        try:
            if self.auto_reload or func_name not in self.loaded_modules:
                module = self._import_module(func_name)
                self.loaded_modules[func_name] = module
            else:
                module = self.loaded_modules[func_name]

            possible_names = [func_name]

            for name in possible_names:
                if hasattr(module, name):
                    func = getattr(module, name)
                    if callable(func):
                        return func

            Logger.error(f"No callable function found in module '{func_name}'")
            return None

        except ImportError as e:
            Logger.error(f"Failed to import module '{func_name}': {e}")
            return None
        except Exception as e:
            Logger.error(f"Error loading function '{func_name}': {e}")
            return None

    def execute_single(self, func_name: str, args: Any | list | dict) -> Any:
        """Execute a single function call

        Args:
            func_name (str): _description_
            args (Any | list | dict): The args to use (It can be Any, but
                                      usually, use list or dict)

        Returns:
            Any: _description_
        """
        func = self.get_function(func_name)
        if not func:
            return {
                "error": f"Function '{func_name}' not found or not callable"
            }

        try:
            if isinstance(args, list):
                return func(*args)
            elif isinstance(args, dict):
                return func(**args)
            else:
                return func(args)
        except TypeError as e:
            Logger.error(f"Argument mismatch for {func_name}: {e}")
            return {"error": f"Argument mismatch: {e}"}
        except Exception as e:
            Logger.error(f"Error executing {func_name}: {e}")
            return {"error": str(e)}

    def execute_batch(
        self,
        config: dict[str, dict],
    ) -> dict[str, Any]:
        """Execute batch of function calls from configuration"""
        results = {}

        for section_name, params in config.items():
            if "input_data" not in params:
                Logger.error(
                    f"'input_data' is not inisde '{section_name}'. Skipping...",
                    "JSON",
                )
                continue

            if "analyser" not in params:
                Logger.error(
                    f"'analyser' is not inisde '{section_name}'. Skipping...",
                    "JSON",
                )
                continue

            args_list = params["input_data"]
            analyser_name = params["analyser"]
            Logger.info(
                f"{section_name}: Executing '{analyser_name}'"
                f"with {len(args_list)} argument(s)"
            )

            func_results = []
            for i, args in enumerate(args_list):
                Logger.debug(f"  Call {i + 1}: {args}")
                result = self.execute_single(analyser_name, args)
                func_results.append(result)

            results[section_name] = func_results

        return results

    def execute_from_file(
        self, file_path: str | Path, h5_path: str | Path
    ) -> dict[str, list] | None:
        """Load configuration from JSON file and execute

        Args:
            file_path (Path): _description_

        Returns:
            dict[str, list] | None: _description_
        """

        file_path = conv_to_path(file_path)

        config = JsonParser.load_json_file(file_path)
        if not config:
            Logger.error(f"Failed to execute Json file: '{file_path}'")
            return None

        return self.execute_batch(config)
