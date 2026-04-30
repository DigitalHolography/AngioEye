from __future__ import annotations

import json
from pathlib import Path

import h5py

from angioeye_io.hdf5_schema import get_processing_root, iter_metric_datasets

from .core.base import (
    BatchPostprocess,
    PostprocessContext,
    PostprocessResult,
    registerPostprocess,
)


@registerPostprocess(
    name="Pipeline Metrics Manifest",
    description=(
        "Scan the generated HDF5 outputs and write a JSON manifest of pipeline "
        "groups and metric dataset paths for the batch."
    ),
)
class PipelineMetricsManifestPostprocess(BatchPostprocess):
    def run(self, context: PostprocessContext) -> PostprocessResult:
        manifest = {
            "input_path": str(context.input_path),
            "selected_pipelines": list(context.selected_pipelines),
            "zip_outputs": context.zip_outputs,
            "files": [],
        }

        for file_path in context.processed_files:
            manifest["files"].append(self._describe_file(file_path, context.output_dir))

        output_path = context.output_dir / "pipeline_metrics_manifest.json"
        output_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return PostprocessResult(
            summary="Generated pipeline_metrics_manifest.json.",
            generated_paths=[str(output_path)],
        )

    def _describe_file(self, file_path: Path, output_dir: Path) -> dict[str, object]:
        try:
            relative_path = str(file_path.resolve().relative_to(output_dir.resolve()))
        except ValueError:
            relative_path = str(file_path)

        pipelines: list[dict[str, object]] = []
        with h5py.File(file_path, "r") as h5file:
            pipelines_group = get_processing_root(h5file)
            if pipelines_group is not None:
                for group_name, group in pipelines_group.items():
                    if not isinstance(group, h5py.Group):
                        continue
                    metric_paths = self._collect_dataset_paths(group)
                    pipelines.append(
                        {
                            "group_name": group_name,
                            "pipeline_name": group.attrs.get("pipeline", group_name),
                            "metric_count": len(metric_paths),
                            "metrics": metric_paths,
                        }
                    )

        return {
            "path": relative_path,
            "pipeline_count": len(pipelines),
            "pipelines": pipelines,
        }

    def _collect_dataset_paths(self, group: h5py.Group) -> list[str]:
        metric_paths = [name for name, _dataset in iter_metric_datasets(group)]
        metric_paths.sort()
        return metric_paths
