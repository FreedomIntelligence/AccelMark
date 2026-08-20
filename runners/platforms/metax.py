"""MetaX MXMACA GPU platform plug-in.

Used by ``runners/collect_env.py`` to populate ``env_info.json``.

Detection order (first non-empty wins):

  1. Vendor torch extension (``mxmaca`` / ``torch_mx`` / ``mx``) — mirrors the
     torch.cuda API (``device_count`` / ``get_device_properties``) and is the
     most reliable signal, since the vllm-metax runner relies on the same
     cu-bridge layer.
  2. ``mx-smi`` / ``maca-smi`` CLI (best-effort).

Deliberately does **not** fall back to generic ``torch.cuda``: MetaX presents
MACA as a CUDA-compatible device, so a bare ``torch.cuda.device_count()``
would also count NVIDIA GPUs and misidentify an NVIDIA host as MetaX.
"""
from __future__ import annotations

import importlib
import os
import re
import subprocess

ID = "metax"
DISPLAY_NAME = "MetaX MXMACA"
VENDOR_LABEL = "MetaX"
PRIORITY = 70

_VENDOR_TORCH_MODULES = ("mxmaca", "torch_mx", "mx")
_SMI_CANDIDATES = ("mx-smi", "maca-smi")


def _supports_bf16(chip_name: str) -> bool:
    # All current MetaX MXC-series GPUs (N260/C500) support BF16 natively.
    return True


def _driver_from_smi() -> str | None:
    for tool in _SMI_CANDIDATES:
        try:
            out = subprocess.check_output(
                [tool], text=True, stderr=subprocess.DEVNULL, timeout=10
            )
        except Exception:
            continue
        m = re.search(r"(?:Driver|MXMACA)\s+[Vv]ersion\s*[:=]\s*(\S+)", out)
        if m:
            return m.group(1)
    return None


def _collect_via_torch() -> list[dict]:
    for modname in _VENDOR_TORCH_MODULES:
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue

        try:
            count = mod.device_count()
        except Exception:
            continue
        if not count:
            continue

        get_props = getattr(mod, "get_device_properties", None)
        driver = _driver_from_smi() or "unknown"
        accelerators: list[dict] = []
        for idx in range(int(count)):
            name = f"MetaX MXMACA GPU {idx}"
            memory_gb = None
            if get_props is not None:
                try:
                    props = get_props(idx)
                    raw = getattr(props, "name", None)
                    if raw:
                        name = raw if isinstance(raw, str) else str(raw)
                    total = getattr(props, "total_memory", None)
                    if total:
                        memory_gb = round(int(total) / (1024 ** 3), 1)
                except Exception:
                    pass
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
        if accelerators:
            return accelerators
    return []


def _collect_via_smi() -> list[dict]:
    for tool in _SMI_CANDIDATES:
        try:
            out = subprocess.check_output(
                [tool], text=True, stderr=subprocess.DEVNULL, timeout=10
            )
        except Exception:
            continue
        if not out.strip():
            continue

        driver = "unknown"
        m = re.search(r"(?:Driver|MXMACA)\s+[Vv]ersion\s*[:=]\s*(\S+)", out)
        if m:
            driver = m.group(1)

        # Best-effort device rows: "| 0  N260  ...  |" or "<name>" tokens.
        accelerators: list[dict] = []
        seen = set()
        for match in re.finditer(r"(?im)(N\d{3}|C\d{3}|MXC\s*\S*|MetaX\s+\S+)", out):
            name = match.group(1).strip()
            if name in seen:
                continue
            seen.add(name)
            accelerators.append(
                {
                    "index": len(accelerators),
                    "name": name,
                    "vendor": VENDOR_LABEL,
                    "memory_gb": None,
                    "driver_version": driver,
                    "firmware_version": None,
                    "supports_bf16": _supports_bf16(name),
                }
            )

        if not accelerators:
            # CLI present but format unknown — still report one device so the
            # collector can populate env_info (name/memory are best-effort).
            accelerators.append(
                {
                    "index": 0,
                    "name": "MetaX MXMACA GPU",
                    "vendor": VENDOR_LABEL,
                    "memory_gb": None,
                    "driver_version": driver,
                    "firmware_version": None,
                    "supports_bf16": True,
                }
            )
        return accelerators
    return []


def collect() -> list[dict]:
    for fn in (_collect_via_torch, _collect_via_smi):
        accelerators = fn()
        if accelerators:
            return accelerators
    return []


def detect_runtime_version() -> str | None:
    for modname in _VENDOR_TORCH_MODULES:
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        ver = getattr(getattr(mod, "version", None), "mx", None)
        if ver:
            return f"MXMACA {ver}"
    try:
        import torch

        ver = getattr(torch.version, "mx", None)
        if ver:
            return f"MXMACA {ver}"
    except Exception:
        pass
    d = _driver_from_smi()
    return f"MetaX MXMACA driver {d}" if d else None


def sample_power_watts() -> float | None:
    for tool in _SMI_CANDIDATES:
        try:
            out = subprocess.check_output(
                [tool], text=True, stderr=subprocess.DEVNULL, timeout=5
            )
        except Exception:
            continue
        total = 0.0
        found = 0
        for m in re.finditer(r"(?i)(\d+\.?\d*)\s*W", out):
            total += float(m.group(1))
            found += 1
        return round(total, 1) if found > 0 else None
    return None


def diagnostics(env: dict, accelerators: list[dict]) -> list[str]:
    notes: list[str] = []
    if not accelerators:
        notes.append(
            "No MetaX MXMACA GPUs detected (tried mx-smi/maca-smi and the "
            "MXMACA torch extension). Install the MXMACA SDK per "
            "https://developer.metax-tech.com ."
        )
        return notes
    if (env.get("runtime_version") or "") in ("unknown", None):
        notes.append(
            "Could not detect the MXMACA runtime version. runtime_version is unknown."
        )
    if accelerators and sample_power_watts() is None:
        notes.append(
            "Power sampling returned None — mx-smi power field is unavailable. "
            "tokens_per_sec_per_watt will not be computed."
        )
    return notes
