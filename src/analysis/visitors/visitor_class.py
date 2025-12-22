from abc import ABC, abstractmethod
from pathlib import Path


# Abstract class
class VisitorClass(ABC):
    @abstractmethod
    @staticmethod
    def visit(file_path: str | Path, h5_path: str) -> dict:
        pass
