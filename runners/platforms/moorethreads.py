"""Moore Threads MUSA GPU platform plug-in.

Moore Threads ships its own driver and management tooling:

* ``mthreads-gmi`` — the moral equivalent of ``nvidia-smi`` / ``rocm-smi``.
* ``pymtml`` — Python bindings analogous to NVML / pynvml.
* ``torchada`` — a CUDA→MUSA compatibility shim that exposes the standard
  ``torch.cuda`` API, with the real backend version available via
  ``torch.version.musa``.

This plug-in first tries the Python bindings (best machine-readable
output) and falls back to scraping ``mthreads-gmi`` text output. Both
paths are best-effort: when none of the tools are installed the plug-in
silently reports zero accelerators and the collector moves on.
"""
from __future__ import annotations

import re
import subprocess

ID = "moorethreads"
DISPLAY_NAME = "Moore Threads"
VENDOR_LABEL = "Moore Threads"
PRIORITY = 60

# S5000 / S4000 datacenter SKUs ship with native BF16 support; the older
# consumer-class MTT S80/S70 cards are FP16-only.
_BF16_SUPPORTED_HINTS = ("s5000", "s4000", "s3000")
_NO_BF16_HINTS = ("s80", "s70", "s60", "s50")


def _supports_bf16(chip_name: str) -> bool:
    if not chip_name:
        return True
    name_lower = chip_name.lower()
    if any(k in name_lower for k in _BF16_SUPPORTED_HINTS):
        return True
    if any(k in name_lower for k in _NO_BF16_HINTS):
        return False
    return True


def _collect_via_pymtml() -> list[dict]:
    try:
        import pymtml as mtml  # type: ignore[import-not-found]
    except ImportError:
        return []

    try:
        mtml.mtmlInit()
    except Exception:
        return []

    accelerators: list[dict] = []
    try:
        count = mtml.mtmlDeviceGetCount()
    except Exception:
        try:
            mtml.mtmlShutdown()
        except Exception:
            pass
        return []

    for idx in range(int(count)):
        try:
            handle = mtml.mtmlDeviceGetHandleByIndex(idx)
            name = mtml.mtmlDeviceGetName(handle)
            mem = mtml.mtmlDeviceGetMemoryInfo(handle)
            total_mb = getattr(mem, "total", None) or mem.get("total", 0)
            driver = mtml.mtmlSystemGetDriverVersion()
        except Exception:
            continue
        accelerators.append(
            {
                "index": idx,
                "name": name if isinstance(name, str) else name.decode("utf-8", "ignore"),
                "vendor": VENDOR_LABEL,
                "memory_gb": round(int(total_mb) / 1024, 1) if total_mb else None,
                "driver_version": driver if isinstance(driver, str) else driver.decode("utf-8", "ignore"),
                "firmware_version": None,
                "supports_bf16": _supports_bf16(str(name)),
            }
        )

    try:
        mtml.mtmlShutdown()
    except Exception:
        pass

    return accelerators


def _collect_via_smi() -> list[dict]:
    """Fallback parser for ``mthreads-gmi`` text output.

    The output format mirrors nvidia-smi: a header with the driver / MUSA
    versions followed by per-device blocks listing the product name and
    memory usage. We only need the device name and total memory.
    """
    try:
        out = subprocess.check_output(
            ["mthreads-gmi"], text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return []

    driver = "unknown"
    m = re.search(r"Driver\s+Version\s*:\s*(\S+)", out, re.IGNORECASE)
    if m:
        driver = m.group(1)

    accelerators: list[dict] = []
    # Per-device rows look like:
    #   |   0  MTT S4000                  ...     | 0000:65:00.0  Off |   ... |
    # followed by:
    #   |   0%   45C    P0    ... /   ... |    234MiB / 49152MiB |    ... |
    for match in re.finditer(
        r"\|\s*(\d+)\s+(MTT\s+\S+(?:\s+\S+)?)\s*", out
    ):
        idx = int(match.group(1))
        name = match.group(2).strip()
        # Search downstream of this match for the memory line
        tail = out[match.end():]
        mem_match = re.search(r"(\d+)MiB\s*/\s*(\d+)MiB", tail)
        memory_gb = None
        if mem_match:
            memory_gb = round(int(mem_match.group(2)) / 1024, 1)
        accelerators.append(
            {
                "index": idx,
                "name": name,
                "vendor": VENDOR_LABEL,
                "memory_gb": memory_gb,
                "driver_version": driver,
                "firmware_version": None,
                "supports_bf16": _supports_bf16(name),
            }
        )
    return accelerators


def collect() -> list[dict]:
    accelerators = _collect_via_pymtml()
    if accelerators:
        return accelerators
    return _collect_via_smi()


def detect_runtime_version() -> str | None:
    """Prefer torch.version.musa (most reliable when torchada is installed),
    fall back to scraping ``mthreads-gmi`` header.
    """
    try:
        import torch

        ver = getattr(torch.version, "musa", None)
        if ver:
            return f"MUSA {ver}"
    except ImportError:
        pass

    try:
        out = subprocess.check_output(
            ["mthreads-gmi"], text=True, stderr=subprocess.DEVNULL
        )
        m = re.search(r"MUSA\s+Version\s*:\s*(\S+)", out, re.IGNORECASE)
        if m:
            return f"MUSA {m.group(1)}"
        m = re.search(r"Driver\s+Version\s*:\s*(\S+)", out, re.IGNORECASE)
        if m:
            return f"Moore Threads Driver {m.group(1)}"
    except Exception:
        pass
    return None


def diagnostics(env: dict, accelerators: list[dict]) -> list[str]:
    notes: list[str] = []
    if accelerators and (env.get("pytorch_version") or "") == "unknown":
        notes.append(
            "PyTorch (with the torchada MUSA shim) is not installed — "
            "pytorch_version is unknown."
        )
    if accelerators and (env.get("runtime_version") or "") == "unknown":
        notes.append(
            "Could not detect MUSA runtime (tried torch.version.musa and "
            "mthreads-gmi). runtime_version is unknown — install torchada "
            "or the Moore Threads MUSA toolkit."
        )
    return notes
