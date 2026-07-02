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


def sample_power_watts() -> float | None:
    """Return instantaneous total board power (watts) summed across all
    visible AMD GPUs, or None if unavailable. Respects ROCR_VISIBLE_DEVICES."""
    import os

    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showpower", "--json"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except Exception:
        return None

    # Respect ROCR_VISIBLE_DEVICES — only sum power for visible GPUs
    visible = os.environ.get("ROCR_VISIBLE_DEVICES", os.environ.get("HIP_VISIBLE_DEVICES", ""))
    visible_indices: set[int] | None = None
    if visible:
        try:
            visible_indices = {int(x.strip()) for x in visible.split(",") if x.strip()}
        except ValueError:
            pass

    try:
        data = json.loads(out)
    except Exception:
        # Fallback: try text parsing (some ROCm versions have different output)
        return _sample_power_amd_text(out)

    total = 0.0
    found = 0
    for card_id, info in data.items():
        if not isinstance(info, dict):
            continue
        # Extract index from card_id like "card0"
        try:
            idx = int(re.sub(r"[^0-9]", "", card_id))
        except ValueError:
            idx = found  # fallback
        if visible_indices is not None and idx not in visible_indices:
            continue
        power = None
        for key in ("Average Graphics Package Power (W)", "Current Socket Graphics Package Power (W)",
                     "GPU Power Draw (W)", "average_graphics_package_power_w",
                     "current_socket_power_w", "gpu_power_draw_w"):
            val = info.get(key)
            if isinstance(val, (int, float)) and val > 0:
                power = float(val)
                break
            if isinstance(val, str):
                try:
                    power = float(val.replace(" W", "").strip())
                    if power > 0:
                        break
                except ValueError:
                    continue
        if power is not None:
            total += power
            found += 1

    return round(total, 1) if found > 0 else None


def _sample_power_amd_text(out: str) -> float | None:
    """Fallback parser for rocm-smi --showpower plain-text output."""
    total = 0.0
    found = 0
    for line in out.splitlines():
        m = re.search(r"(\d+\.?\d*)\s*W", line)
        if m:
            total += float(m.group(1))
            found += 1
    return round(total, 1) if found > 0 else None


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
    if accelerators and sample_power_watts() is None:
        notes.append(
            "Power sampling returned None — rocm-smi --showpower is unavailable. "
            "tokens_per_sec_per_watt will not be computed."
        )
    return notes
