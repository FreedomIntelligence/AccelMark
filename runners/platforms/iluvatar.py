"""Iluvatar CoreX accelerator platform plug-in."""
from __future__ import annotations

import glob
import re


ID = "iluvatar"
DISPLAY_NAME = "Iluvatar CoreX"
VENDOR_LABEL = "Iluvatar"
PRIORITY = 55  # Before the generic torch fallback in moorethreads.py.


def _is_iluvatar(name: str) -> bool:
    lowered = name.lower()
    return "iluvatar" in lowered or "mr-v" in lowered


def collect() -> list[dict]:
    try:
        import torch
    except ImportError:
        return []

    try:
        count = int(torch.cuda.device_count())
    except Exception:
        return []

    accelerators: list[dict] = []
    for idx in range(count):
        try:
            props = torch.cuda.get_device_properties(idx)
            name = str(getattr(props, "name", ""))
            if not _is_iluvatar(name):
                continue
            total = getattr(props, "total_memory", None)
            memory_gb = round(float(total) / (1024 ** 3), 1) if total else None
            try:
                supports_bf16 = bool(torch.cuda.is_bf16_supported())
            except Exception:
                supports_bf16 = True
            accelerators.append(
                {
                    "index": idx,
                    "name": name,
                    "vendor": VENDOR_LABEL,
                    "memory_gb": memory_gb,
                    "driver_version": "unknown",
                    "firmware_version": None,
                    "supports_bf16": supports_bf16,
                }
            )
        except Exception:
            continue
    return accelerators


def detect_runtime_version() -> str | None:
    versions: list[tuple[tuple[int, ...], str]] = []
    for path in glob.glob("/usr/local/corex-*"):
        match = re.search(r"corex-([0-9]+(?:\.[0-9]+)*)$", path)
        if match:
            raw = match.group(1)
            versions.append((tuple(int(x) for x in raw.split(".")), raw))
    corex = max(versions)[1] if versions else None

    cuda = None
    try:
        import torch
        cuda = getattr(torch.version, "cuda", None)
    except ImportError:
        pass

    if corex and cuda:
        return f"CoreX {corex} (CUDA shim {cuda})"
    if corex:
        return f"CoreX {corex}"
    if cuda:
        return f"Iluvatar CUDA shim {cuda}"
    return None


def detect_pcie_gen() -> str | None:
    return None


def detect_topology() -> str | None:
    return None


def detect_intra_node_interconnect() -> str | None:
    return None


def sample_power_watts() -> float | None:
    return None


def diagnostics(env: dict, accelerators: list[dict]) -> list[str]:
    if not accelerators:
        return ["No Iluvatar CoreX accelerator detected via torch.cuda."]
    return []
