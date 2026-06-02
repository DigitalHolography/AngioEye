from __future__ import annotations

import functools
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import ParamSpec, TypeVar


TimingCallback = Callable[[str, float], None]
P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class TimingRecorder:
    _samples: dict[str, list[float]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, label: str, seconds: float) -> None:
        with self._lock:
            self._samples.setdefault(label, []).append(seconds)

    def snapshot(self) -> dict[str, list[float]]:
        with self._lock:
            return {label: list(samples) for label, samples in self._samples.items()}

    def __bool__(self) -> bool:
        with self._lock:
            return bool(self._samples)


TimingSamples = TimingRecorder | Mapping[str, list[float]]


@contextmanager
def timed_section(record_timing: TimingCallback | None, label: str):
    started_at = time.monotonic()
    try:
        yield
    finally:
        if record_timing is not None:
            record_timing(label, time.monotonic() - started_at)


def timed_call(
    label: str,
    record_timing: TimingCallback | None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def _decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def _wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with timed_section(record_timing, label):
                return func(*args, **kwargs)

        return _wrapper

    return _decorator
