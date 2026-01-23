import csv
from dataclasses import dataclass
from typing import Any, Dict, Optional

import h5py


@dataclass
class ProcessResult:
    metrics: Dict[str, Any]
    artifacts: Optional[Dict[str, Any]] = None
    output_h5_path: Optional[str] = None


class ProcessPipeline:
    name: str = "Pipeline"
    description: str = ""

    def run(self, h5file: h5py.File) -> ProcessResult:
        raise NotImplementedError

    def export(self, result: ProcessResult, output_path: str) -> str:
        """Default CSV export for metrics."""
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["metric", "value"])
            for key, value in result.metrics.items():
                writer.writerow([key, value])
        return output_path
