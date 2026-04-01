from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from .core.base import (
    BatchPostprocess,
    PostprocessContext,
    PostprocessResult,
    registerPostprocess,
)


@registerPostprocess(
    name="Variability and heterogeneity tables",
    description=(
        "Build group-level LaTeX and CSV tables for variability and heterogeneity "
        "metrics computed from by-segment arterial waveform shape metrics."
    ),
    required_deps=["pandas>=2.1"],
    required_pipelines=["waveform_shape_metrics"],
)
class GraphicsDashboardPostprocess(BatchPostprocess):
    def run(self, context: PostprocessContext) -> PostprocessResult:
        if not context.processed_files:
            raise ValueError(
                "No processed HDF5 outputs are available for postprocessing."
            )

        output_dir = context.output_dir.expanduser().resolve()
        if not output_dir.exists() or not output_dir.is_dir():
            raise FileNotFoundError(f"Output folder does not exist: {output_dir}")

        from .core import variability_heterogeneity_dashboard

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            temp_zip = temp_root / "batch_outputs.zip"
            self._zip_folder(output_dir, temp_zip)

            cwd = Path.cwd()
            try:
                os.chdir(temp_root)

                results = variability_heterogeneity_dashboard.analyze_zip(
                    str(temp_zip),
                    mode="bandlimited_segment",
                )
                if not results:
                    raise ValueError(
                        "No compatible by-segment metrics were found for the variability/heterogeneity tables."
                    )

                variability_heterogeneity_dashboard.export_group_tables(
                    str(temp_zip),
                    mode="bandlimited_segment",
                )
            finally:
                os.chdir(cwd)

            table_paths = self._extract_prefix(
                zip_path=temp_zip,
                member_prefix="latex_tables/",
                output_dir=output_dir,
            )

        created_paths = [str(path) for path in table_paths]
        summary = (
            f"Generated {len(table_paths)} variability/heterogeneity table file(s)."
        )
        return PostprocessResult(summary=summary, generated_paths=created_paths)

    def _zip_folder(self, folder: Path, zip_path: Path) -> None:
        files = sorted(
            (path for path in folder.rglob("*") if path.is_file()),
            key=lambda path: str(path.relative_to(folder)),
        )
        with zipfile.ZipFile(
            zip_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=1,
        ) as archive:
            for file_path in files:
                archive.write(file_path, file_path.relative_to(folder))

    def _extract_prefix(
        self,
        zip_path: Path,
        member_prefix: str,
        output_dir: Path,
    ) -> list[Path]:
        target_dir = output_dir / member_prefix.rstrip("/")
        if target_dir.exists():
            shutil.rmtree(target_dir)

        extracted: list[Path] = []
        with zipfile.ZipFile(zip_path, "r") as archive:
            for member in archive.namelist():
                if not member.startswith(member_prefix) or member.endswith("/"):
                    continue
                rel_path = Path(member).relative_to(member_prefix.rstrip("/"))
                target = target_dir / rel_path
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src, target.open("wb") as dest:
                    shutil.copyfileobj(src, dest)
                extracted.append(target)
        return extracted
