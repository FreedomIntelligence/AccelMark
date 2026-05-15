"""AMD GPU (ROCm) platform plug-in."""
from __future__ import annotations

import json
import re
import subprocess

ID = "amd"
DISPLAY_NAME = "AMD"
VENDOR_LABEL = "AMD"
PRIORITY = 20

# Architectures that natively support BF16.
_BF16_SUPPORTED = {
    "cdna2", "cdna3",                          # MI200, MI300 series
    "rdna3", "rdna4",                          # RX 7000+ series
    "gfx90a",                                  # MI250X arch code
    "gfx940", "gfx941", "gfx942",             # MI300 arch codes
    "gfx1100", "gfx1101", "gfx1102",          # RDNA3 arch codes
}

# Architectures known to lack hardware BF16.
_NO_BF16 = {
    "cdna1",                                   # MI100
    "rdna", "rdna1", "rdna2",                 # RX 5000, RX 6000 series
    "gfx908",                                  # MI100 arch code
    "gfx1030", "gfx1031",                     # RDNA2 arch codes
}


def _supports_bf16(arch_str: str) -> bool:
    if not arch_str:
        return True
    arch_lower = arch_str.lower()
    if any(k in arch_lower for k in _BF16_SUPPORTED):
        return True
    if any(k in arch_lower for k in _NO_BF16):
        return False
    return True


def collect() -> list[dict]:
    try:
        out = subprocess.check_output(
            [
                "rocm-smi",
                "--showproductname",
                "--showmeminfo", "vram",
                "--showdriverversion",
                "--json",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []

    try:
        data = json.loads(out)
    except Exception:
        return []

    arch_str = ""
    try:
        arch_out = subprocess.check_output(
            ["rocm-smi", "--showallinfo"], text=True, stderr=subprocess.DEVNULL
        )
        gfx_matches = re.findall(r"gfx\d+[a-z]?", arch_out.lower())
        arch_str = gfx_matches[0] if gfx_matches else ""
    except Exception:
        pass

    accelerators: list[dict] = []
    for idx, (_card_id, info) in enumerate(data.items()):
        if not isinstance(info, dict):
            continue
        name = (
            info.get("Card Series")
            or info.get("Card series")
            or info.get("Product Name")
            or info.get("product_name")
            or "AMD GPU"
        )
        mem_bytes = int(
            info.get("VRAM Total Memory (B)")
            or info.get("vram_total_memory_b")
            or info.get("VRAM Total Memory")
            or 0
        )
        driver = (
            info.get("Driver version")
            or info.get("driver_version")
            or info.get("Driver Version")
            or "unknown"
        )
        accelerators.append(
            {
                "index": idx,
                "name": name,
                "vendor": VENDOR_LABEL,
                "memory_gb": round(mem_bytes / (1024 ** 3), 1),
                "driver_version": driver,
                "firmware_version": None,
                "supports_bf16": _supports_bf16(arch_str),
            }
        )
    return accelerators


def detect_runtime_version() -> str | None:
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--version"], text=True, stderr=subprocess.STDOUT
        )
        return f"ROCm {out.strip().splitlines()[-1]}"
    except Exception:
        return None


def detect_topology() -> str | None:
    try:
        return subprocess.check_output(["rocm-smi", "--showtopo"], text=True)
    except Exception:
        return None


def diagnostics(env: dict, accelerators: list[dict]) -> list[str]:
    notes: list[str] = []
    if (env.get("pytorch_version") or "") == "unknown":
        notes.append(
            "PyTorch is not installed — pytorch_version is unknown. For GPU stack "
            "metadata: pip install torch (match your ROCm environment)."
        )
    if (env.get("runtime_version") or "") == "unknown":
        notes.append(
            "Could not detect ROCm runtime (rocm-smi / PyTorch ROCm). "
            "runtime_version is unknown."
        )
    if env.get("accelerator_topology") is None and accelerators:
        notes.append(
            "accelerator_topology is null — rocm-smi --showtopo did not return data."
        )
    return notes
