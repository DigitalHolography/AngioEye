from __future__ import annotations

import shutil
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from batch_engine import BatchExecutionSettings, batch_count, env_int
from input_output import (
    ZipH5Member,
    extract_h5_member,
    extract_h5_members,
    iter_h5_member_batches,
)


@dataclass(frozen=True)
class ZipBatchSettings(BatchExecutionSettings):
    @property
    def extract_workers(self) -> int:
        return self.staging_workers

    @property
    def pipeline_workers(self) -> int:
        return self.task_workers

    @classmethod
    def from_env(cls) -> ZipBatchSettings:
        return cls(
            batch_size=env_int(
                "ANGIOEYE_BATCH_SIZE",
                env_int("ANGIOEYE_ZIP_BATCH_SIZE", cls.batch_size),
            ),
            staging_workers=env_int(
                "ANGIOEYE_BATCH_STAGING_WORKERS",
                env_int(
                    "ANGIOEYE_ZIP_EXTRACT_WORKERS",
                    cls.staging_workers,
                ),
            ),
            task_workers=env_int(
                "ANGIOEYE_BATCH_TASK_WORKERS",
                env_int(
                    "ANGIOEYE_ZIP_PIPELINE_WORKERS",
                    cls.task_workers,
                ),
            ),
        )

    def __init__(
        self,
        batch_size: int = 4,
        staging_workers: int | None = None,
        task_workers: int | None = None,
        *,
        extract_workers: int | None = None,
        pipeline_workers: int | None = None,
    ) -> None:
        if staging_workers is None:
            staging_workers = extract_workers if extract_workers is not None else 2
        if task_workers is None:
            task_workers = (
                pipeline_workers
                if pipeline_workers is not None
                else BatchExecutionSettings.task_workers
            )
        super().__init__(
            batch_size=batch_size,
            staging_workers=staging_workers,
            task_workers=task_workers,
        )


@dataclass(frozen=True)
class ExtractedZipBatch:
    index: int
    count: int
    members: tuple[ZipH5Member, ...]
    h5_paths: tuple[Path, ...]
    root: Path
    error: Exception | None = None


def extract_h5_members_parallel(
    zip_path: str | Path,
    members: list[ZipH5Member],
    output_root: str | Path,
    *,
    max_workers: int,
) -> list[Path]:
    if max_workers <= 1 or len(members) <= 1:
        return extract_h5_members(zip_path, members, output_root)

    worker_count = min(max_workers, len(members))

    def extract_member(member: ZipH5Member) -> Path:
        return extract_h5_member(zip_path, member, output_root)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(extract_member, members))


def iter_extracted_zip_batches(
    zip_path: str | Path,
    members: Iterable[ZipH5Member],
    *,
    member_count: int,
    settings: ZipBatchSettings,
) -> Iterator[ExtractedZipBatch]:
    batch_count_value = batch_count(member_count, settings.batch_size)
    with TemporaryDirectory() as tmp_dir:
        extraction_root = Path(tmp_dir)
        for batch_idx, member_batch in iter_h5_member_batches(
            members,
            settings.batch_size,
        ):
            batch_root = extraction_root / f"batch_{batch_idx:05d}"
            try:
                h5_paths, error = _extract_zip_batch(
                    zip_path=zip_path,
                    members=member_batch,
                    output_root=batch_root,
                    max_workers=settings.extract_workers,
                )
                yield ExtractedZipBatch(
                    index=batch_idx,
                    count=batch_count_value,
                    members=tuple(member_batch),
                    h5_paths=tuple(h5_paths),
                    root=batch_root,
                    error=error,
                )
            finally:
                shutil.rmtree(batch_root, ignore_errors=True)

def _extract_zip_batch(
    *,
    zip_path: str | Path,
    members: list[ZipH5Member],
    output_root: Path,
    max_workers: int,
) -> tuple[list[Path], Exception | None]:
    try:
        h5_paths = extract_h5_members_parallel(
            zip_path,
            members,
            output_root,
            max_workers=max_workers,
        )
    except Exception as exc:  # noqa: BLE001
        return [], exc
    return h5_paths, None
