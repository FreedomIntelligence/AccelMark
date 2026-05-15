"""Apple Silicon (M-series SoC) platform plug-in."""
from __future__ import annotations

import json
import platform
import subprocess

ID = "apple"
DISPLAY_NAME = "Apple Silicon"
VENDOR_LABEL = "Apple"
PRIORITY = 40


def _silicon_brand() -> str | None:
    """Return SoC marketing name (e.g. 'Apple M3 Pro') on Apple Silicon, else None."""
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return chip if "Apple" in chip else None
    except Exception:
        return None


def _supports_bf16(chip_name: str) -> bool:
    """M1 has limited/slow BF16. M2+ has full hardware BF16."""
    if not chip_name:
        return True
    name_lower = chip_name.lower()
    if "m1" in name_lower and "m10" not in name_lower:
        return False
    return True


def _macos_build_string() -> str:
    """Product + build for reproducibility on local Macs."""
    try:
        ver = subprocess.check_output(
            ["sw_vers", "-productVersion"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        build = subprocess.check_output(
            ["sw_vers", "-buildVersion"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        return f"macOS {ver} (build {build})"
    except Exception:
        v = platform.mac_ver()[0]
        return f"macOS {v}" if v else "macOS"


def _metal_summary() -> str | None:
    """Best-effort Metal support line from system_profiler (may take a few seconds)."""
    try:
        proc = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0 or not proc.stdout:
            return None
        data = json.loads(proc.stdout)
        displays = data.get("SPDisplaysDataType") or []
        for disp in displays:
            if not isinstance(disp, dict):
                continue
            for key, val in disp.items():
                kl = key.lower()
                if ("metal" in kl or "mtl" in kl) and val:
                    return f"GPU runtime ({key}): {val}"
        return None
    except Exception:
        return None


def collect() -> list[dict]:
    chip = _silicon_brand()
    if not chip:
        return []
    try:
        mem_bytes = int(
            subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except Exception:
        return []
    return [
        {
            "index": 0,
            "name": chip,
            "vendor": VENDOR_LABEL,
            "memory_gb": round(mem_bytes / (1024 ** 3), 1),
            "driver_version": _macos_build_string(),
            "firmware_version": None,
            "compute_capability": None,
            "supports_bf16": _supports_bf16(chip),
        }
    ]


def detect_runtime_version() -> str | None:
    if not _silicon_brand():
        return None
    try:
        import mlx

        ver = getattr(mlx, "__version__", None)
        if ver is None:
            try:
                from importlib.metadata import version as _pkg_version

                ver = _pkg_version("mlx")
            except Exception:
                ver = None
        return f"MLX {ver}" if ver else "MLX"
    except ImportError:
        pass
    try:
        import torch

        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return f"Metal MPS (PyTorch {torch.__version__})"
    except ImportError:
        pass
    return _macos_build_string()


def detect_pcie_gen() -> str | None:
    if _silicon_brand():
        return "SoC integrated (no discrete PCIe GPU)"
    return None


def detect_topology() -> str | None:
    brand = _silicon_brand()
    if not brand:
        return None
    lines = [
        f"Apple Silicon — integrated GPU in {brand} (unified system memory).",
        f"Machine: {platform.machine()}",
    ]
    metal = _metal_summary()
    if metal:
        lines.append(metal)
    return "\n".join(lines)


def detect_intra_node_interconnect() -> str | None:
    if _silicon_brand():
        return "SoC unified memory (no GPU-GPU interconnect)"
    return None


def diagnostics(env: dict, accelerators: list[dict]) -> list[str]:
    notes: list[str] = []
    runtime = env.get("runtime_version") or ""
    if runtime.startswith("macOS ") and "MLX" not in runtime and "Metal MPS" not in runtime:
        notes.append(
            "Neither MLX nor PyTorch with MPS is available — runtime_version only reflects "
            "macOS build. For ML stack: pip install mlx  or  pip install torch (with MPS)."
        )
    return notes
