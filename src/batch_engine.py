from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")

IdleCallback = Callable[[], None]
MAX_TASK_WORKERS = 32


def default_task_workers() -> int:
    return min(max(1, (os.cpu_count() or 2) - 1), MAX_TASK_WORKERS)


def default_staging_workers() -> int:
    cpu_count = os.cpu_count() or 2
    return max(1, cpu_count // 2)


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    try:
        return max(1, int(value if value else default))
    except ValueError:
        return max(1, int(default))


@dataclass(frozen=True)
class BatchExecutionSettings:
    batch_size: int = field(default_factory=default_task_workers)
    staging_workers: int = field(default_factory=default_staging_workers)
    task_workers: int = field(default_factory=default_task_workers)

    @classmethod
    def from_env(cls) -> BatchExecutionSettings:
        task_workers = env_int("ANGIOEYE_BATCH_TASK_WORKERS", default_task_workers())
        return cls(
            batch_size=env_int("ANGIOEYE_BATCH_SIZE", task_workers),
            staging_workers=env_int(
                "ANGIOEYE_BATCH_STAGING_WORKERS",
                default_staging_workers(),
            ),
            task_workers=task_workers,
        )


@dataclass(frozen=True)
class BatchTaskResult(Generic[T, R]):
    item: T
    value: R | None = None
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
