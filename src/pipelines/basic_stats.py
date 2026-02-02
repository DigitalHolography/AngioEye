import h5py
import numpy as np

from .core.base import ProcessPipeline, ProcessResult, registerPipeline


@registerPipeline(name="Basic Stats")
class BasicStats(ProcessPipeline):
    description = "Min / Max / Mean / Std over the first dataset found in the file."

    def _first_dataset(self, h5file: h5py.File) -> h5py.Dataset | None:
        found: h5py.Dataset | None = None

        def visitor(_name: str, obj: h5py.Dataset) -> None:
            nonlocal found
            if found is None and isinstance(obj, h5py.Dataset):
                found = obj

        h5file.visititems(visitor)
        return found

    def run(self, h5file: h5py.File) -> ProcessResult:
        dataset = self._first_dataset(h5file)
        if dataset is None:
            raise ValueError("No dataset found in the file.")
        data = np.asarray(dataset[...]).ravel()
        finite = data[np.isfinite(data)]
        arr = finite if finite.size > 0 else data
        if arr.size == 0:
            metrics = {
                "count": 0,
                "min": float("nan"),
                "max": float("nan"),
                "mean": float("nan"),
                "std": float("nan"),
            }
        else:
            metrics = {
                "count": float(arr.size),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
            }
        return ProcessResult(metrics=metrics, artifacts={"dataset": dataset.name})
