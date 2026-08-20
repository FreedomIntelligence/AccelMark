"""TecoOrigin SDAA platform plug-in.

Used by ``runners/collect_env.py`` to populate ``env_info.json``.

Detection order (first non-empty wins):

  1. Vendor torch extension (``torch_sdaa`` / ``torch-sdaa`` / ``sdaa``) —
     mirrors the torch.cuda API.
  2. ``sdaa-smi`` / ``tco-smi`` CLI (best-effort).

Deliberately does **not** fall back to generic ``torch.cuda``: Teco-vLLM
presents SDAA as a CUDA-compatible device, so a bare
``torch.cuda.device_count()`` would also count NVIDIA GPUs and misidentify an
NVIDIA host as TecoOrigin.
"""
from __future__ import annotations

import importlib
import re
import subprocess

ID = "tecorigin"
DISPLAY_NAME = "TecoOrigin SDAA"
VENDOR_LABEL = "TecoOrigin"
PRIORITY = 100

_VENDOR_TORCH_MODULES = ("torch_sdaa", "torch_sdaa_core", "sdaa")
_SMI_CANDIDATES = ("sdaa-smi", "tco-smi")


def _supports_bf16(chip_name: str) -> bool:
    # SDAA-200/400 support BF16 for LLM workloads.
    return True


def _driver_from_smi() -> str | None:
    for tool in _SMI_CANDIDATES:
        try:
            out = subprocess.check_output(
                [tool], text=True, stderr=subprocess.DEVNULL, timeout=10
            )
        except Exception:
            continue
        m = re.search(r"(?:Driver|SDAA)\s+[Vv]ersion\s*[:=]\s*(\S+)", out)
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
            name = f"TecoOrigin SDAA {idx}"
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
        m = re.search(r"(?:Driver|SDAA)\s+[Vv]ersion\s*[:=]\s*(\S+)", out)
        if m:
            driver = m.group(1)

        accelerators: list[dict] = []
        seen = set()
        for match in re.finditer(r"(?im)(SDAA[- ]?\d{2,4}|SDAA[- ]?\S*|TecoOrigin\s+\S+)", out):
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
            accelerators.append(
                {
                    "index": 0,
                    "name": "TecoOrigin SDAA",
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
        ver = getattr(getattr(mod, "version", None), "sdaa", None)
        if ver:
            return f"SDAA {ver}"
    d = _driver_from_smi()
    return f"TecoOrigin SDAA driver {d}" if d else None


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
            "No TecoOrigin SDAA accelerators detected (tried sdaa-smi/tco-smi "
            "and the SDAA torch extension). Obtain and install the SDAA SDK from "
            "TecoOrigin."
        )
        return notes
    if (env.get("runtime_version") or "") in ("unknown", None):
        notes.append(
            "Could not detect the SDAA runtime version. runtime_version is unknown."
        )
    if accelerators and sample_power_watts() is None:
        notes.append(
            "Power sampling returned None — sdaa-smi power field is unavailable. "
            "tokens_per_sec_per_watt will not be computed."
        )
    return notes
