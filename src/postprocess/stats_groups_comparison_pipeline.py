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
    name="groups comparison stats",
    description=(
        "Build the cohort HTML dashboard and PNG metric exports from arterial "
        "waveform shape metrics."
    ),
    required_deps=["matplotlib>=3.8", "pandas>=2.1", "plotly>=5.18"],
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

        from .core import stats_groups_comparison

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            temp_zip = temp_root / "batch_outputs.zip"
            self._zip_folder(output_dir, temp_zip)

            cwd = Path.cwd()
            try:
                os.chdir(temp_root)
                all_results, single_group = stats_groups_comparison.analyze_zip(
                    str(temp_zip)
                )
                if not all_results:
                    raise ValueError(
                        "No compatible pipeline metrics were found for the dashboard."
                    )
                stats_groups_comparison.save_dashboard(str(temp_zip))
            finally:
                os.chdir(cwd)

            png_paths = self._extract_prefix(
                zip_path=temp_zip,
                member_prefix="export_png/",
                output_dir=output_dir,
            )
            eps_paths = self._extract_prefix(
                zip_path=temp_zip,
                member_prefix="export_eps/",
                output_dir=output_dir,
            )
        created_paths = [
            *[str(path) for path in png_paths],
            *[str(path) for path in eps_paths],
        ]
        summary = f"Generated {len(png_paths)} PNG illustration(s)."
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

    def _extract_member(
        self,
        zip_path: Path,
        member_name: str,
        output_dir: Path,
    ) -> Path:
        target = output_dir / member_name
        with zipfile.ZipFile(zip_path, "r") as archive:
            with archive.open(member_name) as src, target.open("wb") as dest:
                shutil.copyfileobj(src, dest)
        return target

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
