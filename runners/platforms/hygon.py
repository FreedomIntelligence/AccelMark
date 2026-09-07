"""Hygon DCU platform plug-in.

Used by ``runners/collect_env.py`` to populate ``env_info.json``.

Hygon DCU is presented through a ROCm-compatible software stack (DTK), so
detection is ROCm/HIP-based rather than a CUDA shim. Detection order (first
non-empty wins):

  1. ``hy-smi`` CLI (Hygon's management tool).
  2. ``rocm-smi`` CLI — only accepted when the output carries a Hygon marker
     (``HCU`` / ``DCU`` / ``Hygon`` / ``深算``), so an AMD ROCm host is not
     misattributed.
  3. ``torch`` with a HIP backend (``torch.version.hip`` set) whose device name
     looks like a DCU — again gated on the name to avoid AMD false positives.

The plug-in is prioritised ahead of ``amd`` because Hygon shares the ROCm
toolchain: it must claim the card before the generic AMD detector does.
"""
from __future__ import annotations

import re
import subprocess

ID = "hygon"
DISPLAY_NAME = "Hygon DCU"
VENDOR_LABEL = "Hygon"
# Ahead of amd (PRIORITY 20): Hygon shares the ROCm stack, so this plug-in
# must positively claim the card (via the "HCU" marker) first.
PRIORITY = 15

_SMI_CANDIDATES = ("hy-smi", "rocm-smi")
# Hygon's ROCm fork reports "HCU" (Hygon Compute Unit) instead of AMD's "GPU".
_DCU_NAME_HINTS = ("dcu", "hygon", "深算", "z100", "k100", "hcu")


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


def _run_smi(tool: str, *args: str) -> str | None:
    """Run a management CLI defensively and return its stdout, or None."""
    try:
        return subprocess.check_output(
            [tool, *args], text=True, stderr=subprocess.DEVNULL, timeout=10
        )
    except Exception:
        return None


def _card_series(tool: str) -> str | None:
    """Parse the card series (e.g. ``BW``) from ``--showproductname``."""
    out = _run_smi(tool, "--showproductname") or ""
    m = re.search(r"(?i)card\s+series\s*:\s*(\S+)", out)
    return m.group(1).strip() if m else None


def _vram_gb_by_index(tool: str) -> dict[int, float]:
    """Parse ``--showmeminfo vram`` into ``{hcu_index: total_gb}``.

    Hygon's ROCm fork labels devices inconsistently across driver builds:
    ``HCU[0]`` / ``DCU[0]`` / ``GPU[0]``. Accept all three (brackets optional)
    so the VRAM total is captured regardless of which label this DTK build
    emits (e.g. ``DCU[0]  : vram Total Memory (MiB): 16368``).
    """
    out = _run_smi(tool, "--showmeminfo", "vram") or ""
    mem: dict[int, float] = {}
    for m in re.finditer(
        r"(?i)(?:hcu|dcu|gpu)\s*\[?\s*(\d+)\s*\]?\s*:\s*vram\s+total\s+memory\s*\(\s*mib\s*\)\s*:\s*(\d+)",
        out,
    ):
        mem[int(m.group(1))] = round(int(m.group(2)) / 1024.0, 1)
    return mem


def _collect_via_smi() -> list[dict]:
    for tool in _SMI_CANDIDATES:
        out = _run_smi(tool)
        if not out or not out.strip():
            continue

        # Hygon's ROCm fork reports "HCU" (Hygon Compute Unit) rather than
        # AMD's "GPU", so require a Hygon marker — an AMD ROCm host must be
        # left for the amd plug-in to claim.
        if not _looks_like_dcu(out):
            continue

        driver = "unknown"
        m = re.search(r"(?:Driver|ROCm|DTK)\s+[Vv]ersion\s*[:=]\s*(\S+)", out)
        if m:
            driver = m.group(1)

        series = _card_series(tool)
        if not series:
            nm = re.search(r"(?im)(DCU\s*\S+|深算\s*\S+|Z100|K100|Hygon\s+\S+)", out)
            series = nm.group(1).strip() if nm else "Hygon DCU"

        mem_by_idx = _vram_gb_by_index(tool)
        indices = sorted(mem_by_idx) or [0]

        accelerators: list[dict] = []
        for idx in indices:
            accelerators.append(
                {
                    "index": idx,
                    "name": series,
                    "vendor": VENDOR_LABEL,
                    "memory_gb": mem_by_idx.get(idx),
                    "driver_version": driver,
                    "firmware_version": None,
                    "supports_bf16": _supports_bf16(series),
                }
            )
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
        out = _run_smi(tool)
        if not out:
            continue
        total = 0.0
        found = 0
        for line in out.splitlines():
            # Each device row's first "<n>W" is AvgPwr; the later value is the
            # power *cap*, which must not be summed into the draw.
            m = re.search(r"(?i)(\d+\.?\d*)\s*W", line)
            if m:
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
