"""Moore Threads MUSA GPU platform plug-in.

Used by ``runners/collect_env.py`` to populate ``env_info.json``.

Detection order (first non-empty wins):

  1. ``pymtml`` (mthreads-ml-py) — same API as used in the vllm-musa runner
  2. ``mthreads-gmi`` text output
  3. ``torch`` device properties (``torch.cuda`` aliased to MUSA via torchada,
     or native ``torch.musa`` when available)
"""
from __future__ import annotations

import re
import subprocess

ID = "moorethreads"
DISPLAY_NAME = "Moore Threads"
VENDOR_LABEL = "Moore Threads"
PRIORITY = 60

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


def _driver_version_from_smi() -> str | None:
    try:
        out = subprocess.check_output(
            ["mthreads-gmi"], text=True, stderr=subprocess.DEVNULL
        )
        m = re.search(r"Driver\s+Version\s*:\s*(\S+)", out, re.IGNORECASE)
        if m:
            return m.group(1)
    except Exception:
        pass
    return None


def _collect_via_pymtml() -> list[dict]:
    try:
        import pymtml
    except ImportError:
        return []

    try:
        pymtml.mtmlInit()
    except Exception:
        return []

    driver = _driver_version_from_smi() or "unknown"
    accelerators: list[dict] = []
    try:
        count = pymtml.mtmlDeviceGetCount()
    except Exception:
        try:
            pymtml.mtmlShutdown()
        except Exception:
            pass
        return []

    for idx in range(int(count)):
        try:
            dev = pymtml.mtmlDeviceGetByIndex(idx)
            name = pymtml.mtmlDeviceGetName(dev)
            mem = pymtml.mtmlDeviceGetMemoryInfo(dev)
            total_bytes = getattr(mem, "total", None)
            if total_bytes is None and isinstance(mem, dict):
                total_bytes = mem.get("total")
        except Exception:
            continue
        if not isinstance(name, str):
            name = name.decode("utf-8", "ignore")
        memory_gb = round(int(total_bytes) / (1024 ** 3), 1) if total_bytes else None
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

    try:
        pymtml.mtmlShutdown()
    except Exception:
        pass

    return accelerators


def _collect_via_smi() -> list[dict]:
    """Parse ``mthreads-gmi`` text output (mthreads-gmi 1.14+ tabular format)."""
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
    # Example row:
    #   0    MTT S4000      |00000000:28:00.0    |0%    4MiB(49152MiB)
    for match in re.finditer(
        r"^(\d+)\s+(MTT\s+\S+)\s+\|",
        out,
        re.MULTILINE,
    ):
        idx = int(match.group(1))
        name = match.group(2).strip()
        tail = out[match.end(): match.end() + 256]
        mem_match = re.search(r"\d+MiB\((\d+)MiB\)", tail)
        memory_gb = round(int(mem_match.group(1)) / 1024, 1) if mem_match else None
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


def _collect_via_torch() -> list[dict]:
    """Fallback when management libraries are missing but torch MUSA is loaded."""
    try:
        import torch
    except ImportError:
        return []

    driver = _driver_version_from_smi() or "unknown"
    accelerators: list[dict] = []

    if hasattr(torch, "musa"):
        try:
            count = torch.musa.device_count()
            get_props = torch.musa.get_device_properties
        except Exception:
            count = 0
            get_props = None
    else:
        try:
            count = torch.cuda.device_count()
            get_props = torch.cuda.get_device_properties
        except Exception:
            return []

    for idx in range(int(count)):
        try:
            props = get_props(idx)
            name = getattr(props, "name", None) or f"MTT GPU {idx}"
            total = getattr(props, "total_memory", None)
            memory_gb = round(total / (1024 ** 3), 1) if total else None
        except Exception:
            continue
        accelerators.append(
            {
                "index": idx,
                "name": name if isinstance(name, str) else str(name),
                "vendor": VENDOR_LABEL,
                "memory_gb": memory_gb,
                "driver_version": driver,
                "firmware_version": None,
                "supports_bf16": _supports_bf16(str(name)),
            }
        )
    return accelerators


def collect() -> list[dict]:
    for fn in (_collect_via_pymtml, _collect_via_smi, _collect_via_torch):
        accelerators = fn()
        if accelerators:
            return accelerators
    return []


def detect_runtime_version() -> str | None:
    try:
        import torch

        ver = getattr(torch.version, "musa", None)
        if ver:
            return f"MUSA {ver}"
        if getattr(torch.version, "cuda", None):
            return f"MUSA (torch.cuda shim) {torch.version.cuda}"
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


def detect_pcie_gen() -> str | None:
    try:
        out = subprocess.check_output(
            ["mthreads-gmi"], text=True, stderr=subprocess.DEVNULL
        )
        m = re.search(r"\|\s*(\d+)x\((\d+)x\)\s*\|", out)
        if m:
            return f"PCIe {m.group(1)}x/{m.group(2)}x"
    except Exception:
        pass
    return None


def detect_intra_node_interconnect() -> str | None:
    """Moore Threads multi-GPU hosts typically use MCCL over PCIe."""
    accels = collect()
    if len(accels) > 1:
        return "MCCL/PCIe"
    return None


def sample_power_watts() -> float | None:
    """Return instantaneous total board power (watts) summed across all
    visible Moore Threads GPUs, or None if unavailable.
    Attempts pymtml first, then mthreads-gmi. Respects MUSA_VISIBLE_DEVICES."""
    import os

    # Try pymtml first
    try:
        import pymtml
        pymtml.mtmlInit()
        count = pymtml.mtmlDeviceGetCount()
        visible = os.environ.get("MUSA_VISIBLE_DEVICES", "")
        visible_indices: set[int] | None = None
        if visible:
            try:
                visible_indices = {int(x.strip()) for x in visible.split(",") if x.strip()}
            except ValueError:
                pass

        total = 0.0
        found = 0
        for idx in range(int(count)):
            if visible_indices is not None and idx not in visible_indices:
                continue
            try:
                dev = pymtml.mtmlDeviceGetByIndex(idx)
                power = pymtml.mtmlDeviceGetPowerUsage(dev)
                if power and power > 0:
                    total += float(power)
                    found += 1
            except Exception:
                continue
        try:
            pymtml.mtmlShutdown()
        except Exception:
            pass
        return round(total, 1) if found > 0 else None
    except Exception:
        pass

    # Fallback: try mthreads-gmi
    try:
        out = subprocess.check_output(
            ["mthreads-gmi"], text=True, stderr=subprocess.DEVNULL, timeout=5
        )
    except Exception:
        return None

    visible = os.environ.get("MUSA_VISIBLE_DEVICES", "")
    visible_indices = None
    if visible:
        try:
            visible_indices = {int(x.strip()) for x in visible.split(",") if x.strip()}
        except ValueError:
            pass

    total = 0.0
    found = 0
    for match in re.finditer(
        r"^(\d+)\s+(MTT\s+\S+)\s+\|",
        out,
        re.MULTILINE,
    ):
        idx = int(match.group(1))
        if visible_indices is not None and idx not in visible_indices:
            continue
        # Look for power in the row's tail: e.g. "150W"
        tail = out[match.end(): match.end() + 256]
        power_match = re.search(r"(\d+\.?\d*)\s*W", tail)
        if power_match:
            total += float(power_match.group(1))
            found += 1

    return round(total, 1) if found > 0 else None


def diagnostics(env: dict, accelerators: list[dict]) -> list[str]:
    notes: list[str] = []
    if not accelerators:
        notes.append(
            "No Moore Threads MUSA GPUs detected (tried pymtml, mthreads-gmi, "
            "and torch). Install the MUSA driver/toolkit per "
            "https://github.com/MooreThreads/vllm-musa ."
        )
        return notes
    if (env.get("pytorch_version") or "") == "unknown":
        notes.append(
            "PyTorch with MUSA support is not installed — pytorch_version is unknown."
        )
    if (env.get("runtime_version") or "") == "unknown":
        notes.append(
            "Could not detect MUSA runtime (tried torch.version.musa and "
            "mthreads-gmi). runtime_version is unknown."
        )
    if accelerators and sample_power_watts() is None:
        notes.append(
            "Power sampling returned None — pymtml / mthreads-gmi power is unavailable. "
            "tokens_per_sec_per_watt will not be computed."
        )
    return notes
