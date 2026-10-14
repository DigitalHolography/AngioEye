"""Small batching helpers shared by the statistical tools."""

from collections.abc import Iterable, Iterator, Mapping
from typing import Any, TypeVar

T = TypeVar("T")
DEFAULT_PROCESS_WORKERS = 8


def settings_int(settings: Mapping[str, Any], key: str, default: int) -> int:
    value = settings.get(key)
    try:
        if isinstance(value, bool) or value in (None, ""):
            raise ValueError
        return max(1, int(value))
    except (TypeError, ValueError):
        return max(1, int(default))


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
