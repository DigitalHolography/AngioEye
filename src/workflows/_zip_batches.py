from __future__ import annotations

import shutil
import threading
import time
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from queue import Full, Queue
from tempfile import TemporaryDirectory

from batch_engine import BatchExecutionSettings, batch_count, env_int
from input_output import ZipH5Member, iter_h5_member_batches

from .timing import TimingRecorder


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
                    cls.default_staging_workers(),
                ),
            ),
            task_workers=env_int(
                "ANGIOEYE_BATCH_TASK_WORKERS",
                env_int(
                    "ANGIOEYE_ZIP_PIPELINE_WORKERS",
                    cls.default_task_workers(),
                ),
            ),
        )

    def __init__(
        self,
        batch_size: int = 8,
        staging_workers: int | None = None,
        task_workers: int | None = None,
        *,
        extract_workers: int | None = None,
        pipeline_workers: int | None = None,
    ) -> None:
        if staging_workers is None:
            staging_workers = (
                extract_workers
                if extract_workers is not None
                else self.default_staging_workers()
            )
        if task_workers is None:
            task_workers = (
                pipeline_workers
                if pipeline_workers is not None
                else self.default_task_workers()
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


def iter_extracted_zip_batches(
    zip_path: str | Path,
    members: Iterable[ZipH5Member],
    *,
    member_count: int,
    settings: ZipBatchSettings,
    timings: TimingRecorder | None = None,
) -> Iterator[ExtractedZipBatch]:
    batch_count_value = batch_count(member_count, settings.batch_size)
    batch_queue: Queue[ExtractedZipBatch | None] = Queue(maxsize=1)
    stop_event = threading.Event()

    with TemporaryDirectory() as tmp_dir:
        extraction_root = Path(tmp_dir)
        producer = threading.Thread(
            target=_produce_extracted_zip_batches,
            kwargs={
                "zip_path": zip_path,
                "members": members,
                "batch_count_value": batch_count_value,
                "batch_size": settings.batch_size,
                "extraction_root": extraction_root,
                "batch_queue": batch_queue,
                "stop_event": stop_event,
                "timings": timings,
            },
            daemon=True,
        )
        producer.start()
        try:
            while True:
                extracted_batch = batch_queue.get()
                if extracted_batch is None:
                    break
                try:
                    yield extracted_batch
                finally:
                    shutil.rmtree(extracted_batch.root, ignore_errors=True)
        finally:
            stop_event.set()
            producer.join()


def _produce_extracted_zip_batches(
    *,
    zip_path: str | Path,
    members: Iterable[ZipH5Member],
    batch_count_value: int,
    batch_size: int,
    extraction_root: Path,
    batch_queue: Queue[ExtractedZipBatch | None],
    stop_event: threading.Event,
    timings: TimingRecorder | None,
) -> None:
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            for batch_idx, member_batch in iter_h5_member_batches(members, batch_size):
                if stop_event.is_set():
                    break

                batch_root = extraction_root / f"batch_{batch_idx:05d}"
                extraction_started_at = time.monotonic()
                h5_paths, error = _extract_zip_batch(
                    archive=archive,
                    members=member_batch,
                    output_root=batch_root,
                )
                if timings is not None:
                    timings.add(
                        "source ZIP extraction batch",
                        time.monotonic() - extraction_started_at,
                    )
                if not _put_batch(
                    batch_queue,
                    ExtractedZipBatch(
                        index=batch_idx,
                        count=batch_count_value,
                        members=tuple(member_batch),
                        h5_paths=tuple(h5_paths),
                        root=batch_root,
                        error=error,
                    ),
                    stop_event,
                ):
                    break
    except Exception as exc:  # noqa: BLE001
        _put_batch(
            batch_queue,
            ExtractedZipBatch(
                index=1,
                count=batch_count_value,
                members=(),
                h5_paths=(),
                root=extraction_root / "batch_00001",
                error=exc,
            ),
            stop_event,
        )
    finally:
        _put_batch(batch_queue, None, stop_event)


def _put_batch(
    batch_queue: Queue[ExtractedZipBatch | None],
    item: ExtractedZipBatch | None,
    stop_event: threading.Event,
) -> bool:
    while not stop_event.is_set():
        try:
            batch_queue.put(item, timeout=0.05)
            return True
        except Full:
            continue
    return False


def _extract_zip_batch(
    *,
    archive: zipfile.ZipFile,
    members: list[ZipH5Member],
    output_root: Path,
) -> tuple[list[Path], Exception | None]:
    try:
        h5_paths = [
            _extract_h5_member_from_open_archive(archive, member, output_root)
            for member in members
        ]
    except Exception as exc:  # noqa: BLE001
        return [], exc
    return h5_paths, None


def _extract_h5_member_from_open_archive(
    archive: zipfile.ZipFile,
    member: ZipH5Member,
    output_root: Path,
) -> Path:
    target_path = output_root / member.relative_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member.name) as src, target_path.open("wb") as dest:
        shutil.copyfileobj(src, dest, length=1024 * 1024)
    return target_path
