"""Hygon DCU platform plug-in.

Used by ``runners/collect_env.py`` to populate ``env_info.json``.

Hygon DCU is presented through a ROCm-compatible software stack (DTK), so
detection is ROCm/HIP-based rather than a CUDA shim. Detection order (first
non-empty wins):

  1. ``hy-smi`` CLI (Hygon's management tool).
  2. ``rocm-smi`` CLI — only accepted when the device name looks like a DCU
     (``DCU`` / ``Hygon`` / ``深算``), so an AMD ROCm host is not misattributed.
  3. ``torch`` with a HIP backend (``torch.version.hip`` set) whose device name
     looks like a DCU — again gated on the DCU name to avoid AMD false positives.
"""
from __future__ import annotations

import re
import subprocess

ID = "hygon"
DISPLAY_NAME = "Hygon DCU"
VENDOR_LABEL = "Hygon"
PRIORITY = 110

_SMI_CANDIDATES = ("hy-smi", "rocm-smi")
_DCU_NAME_HINTS = ("dcu", "hygon", "深算", "z100", "k100")


def _supports_bf16(chip_name: str) -> bool:
    # Hygon DCU Z100/K100 support BF16 for LLM workloads.
    return True


def _looks_like_dcu(name: str) -> bool:
    n = (name or "").lower()
    return any(h in n for h in _DCU_NAME_HINTS)


def _driver_from_smi() -> str | None:
    for tool in _SMI_CANDIDATES:
        try:
            out = subprocess.check_output(
                [tool], text=True, stderr=subprocess.DEVNULL, timeout=10
            )
        except Exception:
            continue
        m = re.search(r"(?:Driver|ROCm|DTK)\s+[Vv]ersion\s*[:=]\s*(\S+)", out)
        if m:
            return m.group(1)
    return None


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
        m = re.search(r"(?:Driver|ROCm|DTK)\s+[Vv]ersion\s*[:=]\s*(\S+)", out)
        if m:
            driver = m.group(1)

        accelerators: list[dict] = []
        seen = set()
        for match in re.finditer(r"(?im)(DCU\s*\S*|深算\s*\S*|Z100|K100|Hygon\s+\S+)", out):
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

        if not accelerators and _looks_like_dcu(out):
            # CLI present and mentions DCU but device names weren't parseable —
            # report one device so the collector can populate env_info.
            accelerators.append(
                {
                    "index": 0,
                    "name": "Hygon DCU",
                    "vendor": VENDOR_LABEL,
                    "memory_gb": None,
                    "driver_version": driver,
                    "firmware_version": None,
                    "supports_bf16": True,
                }
            )
        if accelerators:
            return accelerators
    return []


def _collect_via_torch() -> list[dict]:
    try:
        import torch
    except Exception:
        return []

    # ROCm/HIP builds set torch.version.hip; CUDA builds do not.
    if not getattr(getattr(torch, "version", None), "hip", None):
        return []

    try:
        count = torch.cuda.device_count()
        get_props = torch.cuda.get_device_properties
    except Exception:
        return []

    driver = _driver_from_smi() or "unknown"
    accelerators: list[dict] = []
    for idx in range(int(count)):
        try:
            props = get_props(idx)
            name = getattr(props, "name", None) or f"Hygon DCU {idx}"
            name = name if isinstance(name, str) else str(name)
        except Exception:
            continue
        if not _looks_like_dcu(name):
            continue  # AMD/other ROCm device — not Hygon
        total = getattr(props, "total_memory", None)
        memory_gb = round(int(total) / (1024 ** 3), 1) if total else None
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
    for fn in (_collect_via_smi, _collect_via_torch):
        accelerators = fn()
        if accelerators:
            return accelerators
    return []


def detect_runtime_version() -> str | None:
    try:
        import torch

        ver = getattr(torch.version, "hip", None)
        if ver:
            return f"ROCm/HIP {ver} (Hygon DTK)"
    except Exception:
        pass
    d = _driver_from_smi()
    return f"Hygon DTK driver {d}" if d else None


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
            "No Hygon DCU accelerators detected (tried hy-smi/rocm-smi and a "
            "ROCm/HIP torch backend with a DCU device name). Obtain and install "
            "the Hygon DTK stack from Hygon/Sugon."
        )
        return notes
    if (env.get("runtime_version") or "") in ("unknown", None):
        notes.append(
            "Could not detect the Hygon DTK runtime version. runtime_version is unknown."
        )
    if accelerators and sample_power_watts() is None:
        notes.append(
            "Power sampling returned None — hy-smi power field is unavailable. "
            "tokens_per_sec_per_watt will not be computed."
        )
    return notes
