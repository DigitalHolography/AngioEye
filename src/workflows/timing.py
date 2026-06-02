from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, field


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
