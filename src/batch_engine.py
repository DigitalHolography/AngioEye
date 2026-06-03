from __future__ import annotations

import os
import pickle
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from app_settings import AppSettingsStore

T = TypeVar("T")
R = TypeVar("R")

IdleCallback = Callable[[], None]
DEFAULT_BATCH_SIZE = 16
DEFAULT_PROCESS_WORKERS = 8


def default_batch_size() -> int:
    return DEFAULT_BATCH_SIZE


def default_staging_workers() -> int:
    cpu_count = os.cpu_count() or 2
    return max(1, cpu_count // 2)


def _positive_int(value: object, default: int) -> int:
    try:
        if isinstance(value, bool) or value in (None, ""):
            raise ValueError
        return max(1, int(value))
    except (TypeError, ValueError):
        return max(1, int(default))


def settings_int(
    settings: Mapping[str, Any],
    key: str,
    default: int,
) -> int:
    return _positive_int(settings.get(key), default)


@dataclass(frozen=True)
class BatchExecutionSettings:
    batch_size: int = field(default_factory=default_batch_size)
    staging_workers: int = field(default_factory=default_staging_workers)
    process_workers: int = DEFAULT_PROCESS_WORKERS

    @classmethod
    def from_settings(
        cls,
        settings: Mapping[str, Any],
    ) -> BatchExecutionSettings:
        batch_size = settings_int(
            settings,
            "batch_size",
            default_batch_size(),
        )
        return cls(
            batch_size=batch_size,
            staging_workers=settings_int(
                settings,
                "staging_workers",
                default_staging_workers(),
            ),
            process_workers=settings_int(
                settings,
                "process_workers",
                DEFAULT_PROCESS_WORKERS,
            ),
        )

    @classmethod
    def from_app_settings(cls) -> BatchExecutionSettings:
        return cls.from_settings(AppSettingsStore().load_batch_execution())


@dataclass(frozen=True)
class BatchTaskResult(Generic[T, R]):
    item: T
    value: R | None = None
    error: Exception | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class BatchGroupResult(Generic[T, R]):
    index: int
    count: int
    results: tuple[BatchTaskResult[T, R], ...] = ()
    elapsed_seconds: float = 0.0
    process_id: int | None = None
    error: Exception | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


def batch_count(item_count: int, batch_size: int) -> int:
    return (item_count + batch_size - 1) // batch_size


def iter_batches(items: Iterable[T], batch_size: int) -> Iterator[list[T]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def can_pickle(*values: object) -> bool:
    try:
        pickle.dumps(values)
    except Exception:  # noqa: BLE001
        return False
    return True


def run_task_batch(
    items: Sequence[T],
    *,
    run_item: Callable[[T], R],
    max_workers: int,
    idle_callback: IdleCallback | None = None,
) -> Iterator[BatchTaskResult[T, R]]:
    worker_count = min(len(items), max(1, max_workers))
    if worker_count <= 1:
        yield from _run_task_batch_sequential(items, run_item=run_item)
        return

    yield from _run_task_batch_parallel(
        items,
        run_item=run_item,
        max_workers=worker_count,
        idle_callback=idle_callback,
    )


def run_threaded_batches_in_process_pool(
    batches: Sequence[Sequence[T]],
    *,
    run_item: Callable[[T], R],
    process_workers: int,
    thread_workers: int,
    idle_callback: IdleCallback | None = None,
) -> Iterator[BatchGroupResult[T, R]]:
    process_count = min(len(batches), max(1, process_workers))
    if process_count <= 0:
        return

    group_count = len(batches)
    with ProcessPoolExecutor(max_workers=process_count) as executor:
        futures = {}
        for group_index, batch in enumerate(batches, start=1):
            worker_count = min(len(batch), max(1, thread_workers))
            future = executor.submit(
                _run_task_group_in_process,
                group_index,
                group_count,
                tuple(batch),
                run_item,
                worker_count,
            )
            futures[future] = group_index

        pending = set(futures)
        while pending:
            done, pending = wait(
                pending,
                timeout=0.05,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                if idle_callback is not None:
                    idle_callback()
                continue

            for future in done:
                group_index = futures[future]
                try:
                    yield future.result()
                except Exception as exc:  # noqa: BLE001
                    yield BatchGroupResult(
                        index=group_index,
                        count=group_count,
                        error=exc,
                    )

            if idle_callback is not None:
                idle_callback()


def run_threaded_batches_in_process_pool_bounded(
    batches: Iterable[Sequence[T]],
    *,
    group_count: int,
    run_item: Callable[[T], R],
    process_workers: int,
    thread_workers: int,
    max_pending_batches: int,
    idle_callback: IdleCallback | None = None,
) -> Iterator[BatchGroupResult[T, R]]:
    yield from run_indexed_threaded_batches_in_process_pool_bounded(
        enumerate(batches, start=1),
        group_count=group_count,
        run_item=run_item,
        process_workers=process_workers,
        thread_workers=thread_workers,
        max_pending_batches=max_pending_batches,
        idle_callback=idle_callback,
    )


def run_indexed_threaded_batches_in_process_pool_bounded(
    indexed_batches: Iterable[tuple[int, Sequence[T]]],
    *,
    group_count: int,
    run_item: Callable[[T], R],
    process_workers: int,
    thread_workers: int,
    max_pending_batches: int,
    idle_callback: IdleCallback | None = None,
) -> Iterator[BatchGroupResult[T, R]]:
    process_count = max(1, process_workers)
    pending_limit = max(1, max_pending_batches)
    batch_iter = iter(indexed_batches)

    with ProcessPoolExecutor(max_workers=process_count) as executor:
        futures = {}
        exhausted = False

        def submit_ready() -> None:
            nonlocal exhausted
            while not exhausted and len(futures) < pending_limit:
                try:
                    group_index, batch = next(batch_iter)
                except StopIteration:
                    exhausted = True
                    return
                worker_count = min(len(batch), max(1, thread_workers))
                future = executor.submit(
                    _run_task_group_in_process,
                    group_index,
                    group_count,
                    tuple(batch),
                    run_item,
                    worker_count,
                )
                futures[future] = group_index

        submit_ready()
        while futures:
            done, _pending = wait(
                set(futures),
                timeout=0.05,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                if idle_callback is not None:
                    idle_callback()
                continue

            for future in done:
                group_index = futures.pop(future)
                try:
                    yield future.result()
                except Exception as exc:  # noqa: BLE001
                    yield BatchGroupResult(
                        index=group_index,
                        count=group_count,
                        error=exc,
                    )
            submit_ready()
            if idle_callback is not None:
                idle_callback()


def _run_task_group_in_process(
    group_index: int,
    group_count: int,
    batch: tuple[T, ...],
    run_item: Callable[[T], R],
    thread_workers: int,
) -> BatchGroupResult[T, R]:
    started_at = time.monotonic()
    return BatchGroupResult(
        index=group_index,
        count=group_count,
        results=tuple(
            run_task_batch(
                batch,
                run_item=run_item,
                max_workers=thread_workers,
            )
        ),
        elapsed_seconds=time.monotonic() - started_at,
        process_id=os.getpid(),
    )


def _run_task_batch_sequential(
    items: Sequence[T],
    *,
    run_item: Callable[[T], R],
) -> Iterator[BatchTaskResult[T, R]]:
    for item in items:
        try:
            value = run_item(item)
        except Exception as exc:  # noqa: BLE001
            yield BatchTaskResult(item=item, error=exc)
            continue
        yield BatchTaskResult(item=item, value=value)


def _run_task_batch_parallel(
    items: Sequence[T],
    *,
    run_item: Callable[[T], R],
    max_workers: int,
    idle_callback: IdleCallback | None,
) -> Iterator[BatchTaskResult[T, R]]:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_item, item): item for item in items}
        pending = set(futures)
        while pending:
            done, pending = wait(
                pending,
                timeout=0.05,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                if idle_callback is not None:
                    idle_callback()
                continue

            for future in done:
                item = futures[future]
                try:
                    value = future.result()
                except Exception as exc:  # noqa: BLE001
                    yield BatchTaskResult(item=item, error=exc)
                    continue
                yield BatchTaskResult(item=item, value=value)

            if idle_callback is not None:
                idle_callback()
