"""Shared Matplotlib figure export: PNG plus mirrored EPS companion.

EPS is best-effort but aggressive: translucent artists are flattened onto
white (vector-preserving), PS fonts use TrueType, and failed saves retry
with selective then broad rasterization before giving up.
"""

from __future__ import annotations

import contextlib
import logging
import warnings
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from input_output.output_paths import eps_path_for_png

POSTSCRIPT_BACKEND_MODULE = "matplotlib.backends.backend_ps"
_EPS_BACKEND_AVAILABLE: bool | None = None
_EPS_BACKEND_WARNED = False
_WHITE = (1.0, 1.0, 1.0)

# PostScript-friendly rc overrides applied only around EPS saves.
_EPS_RC = {
    "ps.fonttype": 42,  # TrueType; better Unicode / mathtext embedding
    "pdf.fonttype": 42,
    "ps.papersize": "figure",
}


def postscript_backend_available() -> bool:
    """True when Matplotlib can write Encapsulated PostScript."""
    global _EPS_BACKEND_AVAILABLE
    if _EPS_BACKEND_AVAILABLE is None:
        try:
            __import__(POSTSCRIPT_BACKEND_MODULE)
        except ModuleNotFoundError:
            _EPS_BACKEND_AVAILABLE = False
        else:
            _EPS_BACKEND_AVAILABLE = True
    return bool(_EPS_BACKEND_AVAILABLE)


def save_figure(fig, out_path: str | Path, **savefig_kwargs) -> Path:
    """Save ``fig`` as PNG and, when possible, a mirrored EPS companion.

    ``out_path`` must be the PNG path. Savefig kwargs (``bbox_inches``,
    ``pad_inches``, ``dpi``, …) are applied unchanged to the PNG so layout
    and design stay identical. The EPS copy reuses the same layout kwargs
    (``dpi`` / ``format`` / ``backend`` are dropped) and applies PS-safe
    overrides that do not mutate the on-screen / PNG appearance.
    """
    out_path = Path(out_path)
    if out_path.suffix.lower() != ".png":
        raise ValueError(f"save_figure expects a .png path, got {out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, **savefig_kwargs)
    _save_eps_companion(fig, out_path, savefig_kwargs)
    return out_path


def _save_eps_companion(fig, png_path: Path, savefig_kwargs: dict) -> Path | None:
    """Write the sibling EPS for ``png_path``, or skip with a warning."""
    ps_ok = postscript_backend_available()
    cairo_ok = _cairo_backend_available()
    if not ps_ok and not cairo_ok:
        _warn_backend_missing()
        return None

    eps_path = eps_path_for_png(png_path)
    eps_kwargs = {
        key: value
        for key, value in savefig_kwargs.items()
        if key
        not in {"dpi", "format", "backend", "transparent", "facecolor", "edgecolor"}
    }
    # Opaque white page: PS does not composite alpha the way Agg/PNG does.
    eps_kwargs["transparent"] = False
    eps_kwargs["facecolor"] = "white"
    eps_kwargs["edgecolor"] = "none"

    try:
        eps_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _warn_eps_failure(png_path, exc)
        return None

    attempts: list[tuple[str, contextlib.AbstractContextManager]] = []
    if ps_ok:
        attempts.extend(
            [
                ("vector", _eps_prepare_vector(fig)),
                ("rasterize_alpha", _eps_prepare_rasterize_alpha(fig)),
                ("rasterize_artists", _eps_prepare_rasterize_non_text(fig)),
            ]
        )
    if cairo_ok:
        # Cairo often tolerates constructs the classic PS backend rejects.
        attempts.append(("cairo", _eps_prepare_vector(fig)))

    last_error: BaseException | None = None
    for name, prepare in attempts:
        try:
            with _eps_rc_context(), prepare, _silence_ps_transparency_log():
                save_kw = dict(eps_kwargs)
                if name == "cairo":
                    save_kw["backend"] = "cairo"
                fig.savefig(eps_path, format="eps", **save_kw)
            return eps_path
        except ModuleNotFoundError as exc:
            missing = getattr(exc, "name", None) or ""
            if missing == POSTSCRIPT_BACKEND_MODULE or "cairo" in missing.lower():
                last_error = exc
                continue
            raise
        except Exception as exc:  # try the next hardening strategy
            last_error = exc
            if eps_path.exists():
                with contextlib.suppress(OSError):
                    eps_path.unlink()
            continue

    if last_error is not None:
        _warn_eps_failure(png_path, last_error)
    elif not ps_ok:
        _warn_backend_missing()
    return None


def _warn_backend_missing() -> None:
    global _EPS_BACKEND_WARNED
    if _EPS_BACKEND_WARNED:
        return
    warnings.warn(
        "EPS export skipped because the Matplotlib PostScript backend "
        f"'{POSTSCRIPT_BACKEND_MODULE}' is unavailable in this build.",
        RuntimeWarning,
        stacklevel=4,
    )
    _EPS_BACKEND_WARNED = True


def _warn_eps_failure(png_path: Path, exc: BaseException) -> None:
    kind = type(exc).__name__
    detail = " ".join(str(exc).split())[:200] or kind
    warnings.warn(
        f"EPS export skipped for {png_path.name}: {kind}: {detail}",
        RuntimeWarning,
        stacklevel=4,
    )


def _cairo_backend_available() -> bool:
    try:
        __import__("matplotlib.backends.backend_cairo")
    except Exception:
        return False
    return True


@contextlib.contextmanager
def _eps_rc_context() -> Iterator[None]:
    import matplotlib as mpl

    with mpl.rc_context(_EPS_RC):
        yield


@contextlib.contextmanager
def _silence_ps_transparency_log() -> Iterator[None]:
    """Quiet matplotlib's PS transparency *log* line (not a warnings.warn)."""
    logger = logging.getLogger("matplotlib.backends.backend_ps")
    previous = logger.level
    logger.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*PostScript backend does not support transparency.*",
            )
            yield
    finally:
        logger.setLevel(previous)


def _blend_rgba_on_white(rgba) -> tuple[float, float, float, float] | None:
    """Return opaque RGB blended onto white, or None if already opaque / invalid."""
    try:
        values = np.asarray(rgba, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if values.size < 4:
        return None
    alpha = float(values[3])
    if not np.isfinite(alpha) or alpha >= 1.0 - 1e-9:
        return None
    if alpha <= 0.0:
        return (*_WHITE, 1.0)
    rgb = values[:3] * alpha + np.asarray(_WHITE, dtype=float) * (1.0 - alpha)
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)


def _effective_rgba_array(color, artist_alpha: float | None):
    """RGBA array for ``color``, with artist-level alpha applied when needed.

    ``Collection`` / ``Patch`` face colors often already carry the artist alpha
    in the RGBA A channel. Only multiply when the color is fully opaque but the
    artist itself is translucent (common for Line2D).
    """
    from matplotlib.colors import to_rgba_array

    arr = np.asarray(to_rgba_array(color), dtype=float)
    if arr.size == 0:
        return arr
    if artist_alpha is None or not np.isfinite(artist_alpha):
        return arr
    artist_alpha = float(artist_alpha)
    if artist_alpha >= 1.0 - 1e-9:
        return arr
    out = arr.copy()
    # If every A channel is already < 1, the artist alpha is already baked in
    # (fill_between stores alpha on the face color). Do not multiply again.
    if np.all(out[:, 3] < 1.0 - 1e-9):
        return out
    out[:, 3] = np.clip(out[:, 3] * artist_alpha, 0.0, 1.0)
    return out


def _flatten_color(color, *, artist_alpha: float | None = None):
    """Flatten one color or a sequence of colors onto white."""
    if color is None:
        return None
    # Sentinel / named values the PS path should leave alone.
    if isinstance(color, str) and color.lower() in {"none", "auto"}:
        return None
    try:
        arr = _effective_rgba_array(color, artist_alpha)
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    if arr.shape[0] == 1:
        return _blend_rgba_on_white(arr[0])
    out = []
    changed = False
    for row in arr:
        blended = _blend_rgba_on_white(row)
        if blended is None:
            out.append(tuple(float(x) for x in row))
        else:
            out.append(blended)
            changed = True
    return out if changed else None


@contextlib.contextmanager
def _eps_prepare_vector(fig) -> Iterator[None]:
    """Flatten translucent face/edge/line colors onto white; keep vectors."""
    restores: list[tuple] = []
    try:
        for artist in list(fig.findobj()):
            _flatten_artist_alpha(artist, restores)
        yield
    finally:
        _run_restores(restores)


@contextlib.contextmanager
def _eps_prepare_rasterize_alpha(fig) -> Iterator[None]:
    """Rasterize artists that still carry transparency; keep the rest vector."""
    restores: list[tuple] = []
    try:
        for artist in list(fig.findobj()):
            _flatten_artist_alpha(artist, restores)
            if _artist_has_transparency(artist) and hasattr(artist, "set_rasterized"):
                restores.append((artist.set_rasterized, artist.get_rasterized()))
                artist.set_rasterized(True)
        yield
    finally:
        _run_restores(restores)


@contextlib.contextmanager
def _eps_prepare_rasterize_non_text(fig) -> Iterator[None]:
    """Last resort: rasterize every non-text artist so EPS can still write."""
    from matplotlib.text import Text

    restores: list[tuple] = []
    try:
        for artist in list(fig.findobj()):
            _flatten_artist_alpha(artist, restores)
            if isinstance(artist, Text):
                continue
            if hasattr(artist, "set_rasterized"):
                restores.append((artist.set_rasterized, artist.get_rasterized()))
                artist.set_rasterized(True)
        yield
    finally:
        _run_restores(restores)


def _run_restores(restores: list[tuple]) -> None:
    for item in reversed(restores):
        setter, value = item[0], item[1]
        with contextlib.suppress(Exception):
            if value is None and getattr(setter, "__name__", "").startswith(
                "_restore_"
            ):
                setter()
            else:
                setter(value)


def _artist_has_transparency(artist) -> bool:
    alpha = getattr(artist, "get_alpha", lambda: None)()
    try:
        if alpha is not None and np.isfinite(alpha) and float(alpha) < 1.0 - 1e-9:
            return True
    except (TypeError, ValueError):
        pass
    try:
        from matplotlib.colors import to_rgba_array
    except Exception:
        return False
    for getter in ("get_facecolor", "get_edgecolor", "get_color"):
        if not hasattr(artist, getter):
            continue
        try:
            color = getattr(artist, getter)()
            if isinstance(color, str) and color.lower() in {"none", "auto"}:
                continue
            arr = to_rgba_array(color)
        except Exception:
            continue
        if arr.size == 0:
            continue
        if np.any(arr[:, 3] < 1.0 - 1e-9):
            return True
    return False


def _flatten_artist_alpha(artist, restores: list[tuple]) -> None:
    """Bake translucent colors into opaque RGB blends for the PS backend.

    ``Collection.set_alpha`` reloads ``_original_facecolor`` and can wipe a
    prior ``set_facecolor`` of the baked gray. Compute blends first, clear
    alpha, then re-apply the baked colors (and originals) so EPS keeps the
    light-gray band instead of solid black.
    """
    artist_alpha = None
    if hasattr(artist, "get_alpha"):
        try:
            artist_alpha = artist.get_alpha()
        except Exception:
            artist_alpha = None

    baked: list[tuple] = []
    for get_name, set_name in (
        ("get_facecolor", "set_facecolor"),
        ("get_edgecolor", "set_edgecolor"),
        ("get_color", "set_color"),
        ("get_markerfacecolor", "set_markerfacecolor"),
        ("get_markeredgecolor", "set_markeredgecolor"),
    ):
        result = _prepare_flattened_color(
            artist, get_name, set_name, artist_alpha=artist_alpha
        )
        if result is not None:
            baked.append(result)

    if (
        hasattr(artist, "set_alpha")
        and artist_alpha is not None
        and np.isfinite(artist_alpha)
        and float(artist_alpha) < 1.0 - 1e-9
    ):
        restores.append((artist.set_alpha, artist_alpha))
        artist.set_alpha(1.0)

    for setter, original, flattened in baked:
        restores.append((setter, original))
        with contextlib.suppress(Exception):
            setter(flattened)
        # Keep Collection originals in sync so a later alpha touch cannot
        # resurrect the translucent source color.
        if getattr(setter, "__name__", "") == "set_facecolor" and hasattr(
            artist, "_original_facecolor"
        ):
            previous_original = artist._original_facecolor

            def _restore_face_original(value=previous_original, target=artist) -> None:
                target._original_facecolor = value

            restores.append((_restore_face_original, None))
            artist._original_facecolor = flattened
        if getattr(setter, "__name__", "") == "set_edgecolor" and hasattr(
            artist, "_original_edgecolor"
        ):
            previous_original = artist._original_edgecolor

            def _restore_edge_original(value=previous_original, target=artist) -> None:
                target._original_edgecolor = value

            restores.append((_restore_edge_original, None))
            artist._original_edgecolor = flattened


def _prepare_flattened_color(
    artist,
    get_name: str,
    set_name: str,
    *,
    artist_alpha: float | None = None,
) -> tuple | None:
    if not (hasattr(artist, get_name) and hasattr(artist, set_name)):
        return None
    getter = getattr(artist, get_name)
    setter = getattr(artist, set_name)
    try:
        original = getter()
    except Exception:
        return None
    flattened = _flatten_color(original, artist_alpha=artist_alpha)
    if flattened is None:
        return None
    return (setter, original, flattened)
