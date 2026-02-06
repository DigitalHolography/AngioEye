import inspect
import linecache
import traceback
from pathlib import Path

from .base import ProcessPipeline


def _resolve_path(path: str | None) -> Path | None:
    if not path:
        return None
    try:
        return Path(path).resolve()
    except (OSError, RuntimeError, ValueError):
        return None


def _shorten_path(path: str) -> str:
    try:
        resolved = Path(path).resolve()
        return str(resolved.relative_to(Path.cwd()))
    except (OSError, RuntimeError, ValueError):
        return path


def _pick_relevant_frame(
    frames: list[traceback.FrameSummary],
    pipeline_path: Path | None,
) -> traceback.FrameSummary:
    if pipeline_path is not None:
        for frame in reversed(frames):
            frame_path = _resolve_path(frame.filename)
            if frame_path and frame_path == pipeline_path:
                return frame
    for frame in reversed(frames):
        if "pipelines" in Path(frame.filename).parts:
            return frame
    return frames[-1]


def format_pipeline_exception(
    exc: BaseException, pipeline: ProcessPipeline | None = None
) -> str:
    """
    Format an exception raised while running a pipeline, highlighting the
    most relevant line in the pipeline code when possible.
    """
    summary = f"{type(exc).__name__}: {exc}"
    label = f"Pipeline '{pipeline.name}'" if pipeline is not None else "Pipeline"

    frames = traceback.extract_tb(exc.__traceback__)
    if not frames:
        return f"{label} failed: {summary}"

    pipeline_path = None
    if pipeline is not None:
        pipeline_path = _resolve_path(inspect.getsourcefile(pipeline.__class__))

    target = _pick_relevant_frame(frames, pipeline_path)
    location = f"{_shorten_path(target.filename)}:{target.lineno} in {target.name}()"
    line = (target.line or linecache.getline(target.filename, target.lineno)).strip()

    if line:
        return (
            f"{label} failed: {summary}\n"
            f"  at {location}\n"
            f"    {line}\n"
            f"    ^"
        )
    return f"{label} failed: {summary}\n  at {location}"
