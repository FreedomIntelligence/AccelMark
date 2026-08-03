"""
Backend registry with auto-discovery.

Each module in ``profiling.backends`` that does **not** start with
``_`` is imported.  Modules that export a :class:`ProfilerBackend`
subclass are collected, sorted by ``PRIORITY``, and made available
through :func:`get_active_backend`.

The first backend whose :meth:`~ProfilerBackend.is_available` returns
``True`` is cached at module level — the active platform does not
change during a process lifetime.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Optional

from profiling.base import ProfilerBackend

# ── Module-level cache ──────────────────────────────────────────────────────

_active_backend_cache: Optional[ProfilerBackend] = None
"""Cache for the resolved backend instance.  ``None`` means not yet resolved;
a sentinel of ``False`` means resolution ran but no backend is available."""

_SENTINEL = False


# ── Discovery ────────────────────────────────────────────────────────────────


def _iter_backend_classes() -> list[type[ProfilerBackend]]:
    """Import every ``profiling.backends.<name>`` module and collect
    concrete :class:`ProfilerBackend` subclasses.

    Sorted by ``(PRIORITY, ID)`` — lower priority tried first.
    Import errors are silently swallowed.
    """
    classes: list[type[ProfilerBackend]] = []
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"{__name__}.{info.name}")
        except Exception:
            continue
        # Collect any ProfilerBackend subclass defined in the module
        for attr in dir(mod):
            obj = getattr(mod, attr, None)
            if (
                isinstance(obj, type)
                and issubclass(obj, ProfilerBackend)
                and obj is not ProfilerBackend
            ):
                classes.append(obj)
    classes.sort(key=lambda cls: (getattr(cls, "PRIORITY", 50),
                                   getattr(cls, "ID", "")))
    return classes


def discover_backends() -> list[ProfilerBackend]:
    """Return instantiated backends for every detected subclass.

    Each backend is a fresh instance.  Callers should call
    :meth:`ProfilerBackend.is_available` to filter.
    """
    return [cls() for cls in _iter_backend_classes()]


# ── Resolution ───────────────────────────────────────────────────────────────


def get_active_backend() -> Optional[ProfilerBackend]:
    """Return the first available backend instance, or ``None``.

    Resolution runs at most once per process; the result is cached.
    """
    global _active_backend_cache
    if _active_backend_cache is not None:
        if _active_backend_cache is _SENTINEL:
            return None
        return _active_backend_cache

    for backend in discover_backends():
        try:
            if backend.is_available():
                _active_backend_cache = backend
                return backend
        except Exception:
            continue

    _active_backend_cache = _SENTINEL
    return None


def get_backend_by_id(backend_id: str) -> Optional[ProfilerBackend]:
    """Return a backend instance by its :attr:`~ProfilerBackend.ID`.

    Returns ``None`` when no backend with the given *backend_id* exists.
    Does **not** check :meth:`~ProfilerBackend.is_available`.
    """
    for cls in _iter_backend_classes():
        if getattr(cls, "ID", "") == backend_id:
            return cls()
    return None
