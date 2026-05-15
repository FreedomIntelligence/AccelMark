"""
Platform plug-ins for the AccelMark environment collector.

Each file in this package (other than the ones starting with ``_``) is a
self-contained detector for one accelerator family. ``collect_env.py``
imports them all at runtime and asks them, in priority order, whether
their accelerator is present on the host. The first plug-in that returns
a non-empty ``collect()`` result owns the environment report.

To add support for a new platform you only need to drop a single file
``runners/platforms/<my_platform>.py`` exporting one or more of the
following module-level attributes:

    ID:               str    — short identifier, lowercase, matches the
                              ``platform`` field used in meta.json
    DISPLAY_NAME:     str    — human-readable label used in warnings
    PRIORITY:         int    — lower numbers are tried first; default 50
    VENDOR_LABEL:     str    — string written into accelerator["vendor"]
                              by ``collect()`` (informational)

    def collect() -> list[dict]: ...
    def detect_runtime_version() -> str | None: ...
    def detect_pcie_gen() -> str | None: ...
    def detect_topology() -> str | None: ...
    def detect_intra_node_interconnect() -> str | None: ...
    def diagnostics(env, accelerators) -> list[str]: ...

All functions are optional. The collector skips any that are missing
and silently swallows any plug-in import errors so a broken third-party
plug-in cannot block detection of the platforms that ship with the
repository.
"""
from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType
from typing import List


def _plugin_sort_key(mod: ModuleType) -> tuple[int, str]:
    return (int(getattr(mod, "PRIORITY", 50)), str(getattr(mod, "ID", mod.__name__)))


def discover_plugins() -> List[ModuleType]:
    """Import every ``runners.platforms.<name>`` module and return them
    sorted by ``PRIORITY`` (lower first). Modules whose name begins with
    an underscore are treated as internal/helper and skipped.

    Plug-ins that fail to import (for example due to a missing optional
    dependency at import time) are silently ignored — detection of
    other platforms must not be blocked by one broken plug-in.
    """
    plugins: list[ModuleType] = []
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"{__name__}.{info.name}")
        except Exception:
            continue
        plugins.append(mod)
    plugins.sort(key=_plugin_sort_key)
    return plugins


__all__ = ["discover_plugins"]
