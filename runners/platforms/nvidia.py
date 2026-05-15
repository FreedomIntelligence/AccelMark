"""NVIDIA GPU platform plug-in."""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

ID = "nvidia"
DISPLAY_NAME = "NVIDIA"
VENDOR_LABEL = "NVIDIA"
PRIORITY = 10


def collect() -> list[dict]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return []

    accelerators: list[dict] = []
    for line in out.strip().splitlines():
        idx, name, mem, driver, compute_cap = [x.strip() for x in line.split(",")]
        try:
            cc_float = float(compute_cap) if compute_cap else 0.0
            supports_bf16 = cc_float >= 8.0
        except (ValueError, TypeError):
            supports_bf16 = True
        accelerators.append(
            {
                "index": int(idx),
                "name": name,
                "vendor": VENDOR_LABEL,
                "memory_gb": round(float(mem) / 1024, 1),
                "driver_version": driver,
                "firmware_version": None,
                "compute_capability": compute_cap,
                "supports_bf16": supports_bf16,
            }
        )
    return accelerators


def detect_runtime_version() -> str | None:
    try:
        import torch

        if torch.version.cuda:
            return f"CUDA {torch.version.cuda}"
    except ImportError:
        pass

    try:
        out = subprocess.check_output(
            ["nvcc", "--version"], text=True, stderr=subprocess.STDOUT
        )
        for line in out.splitlines():
            if "release" in line.lower():
                parts = line.split("release")
                if len(parts) > 1:
                    version = parts[1].split(",")[0].strip()
                    return f"CUDA {version}"
    except Exception:
        pass

    for env_var in ("CUDA_HOME", "CUDA_PATH"):
        cuda_home = os.environ.get(env_var)
        if not cuda_home:
            continue
        version_file = Path(cuda_home) / "version.txt"
        if version_file.exists():
            return version_file.read_text().strip()
        version_json = Path(cuda_home) / "version.json"
        if version_json.exists():
            try:
                data = json.loads(version_json.read_text())
                cuda = data.get("cuda", {}).get("version", "")
                if cuda:
                    return f"CUDA {cuda}"
            except Exception:
                pass

    return None


def detect_pcie_gen() -> str | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=pcie.link.gen.current",
                "--format=csv,noheader",
            ],
            text=True,
        )
        gen = out.strip().splitlines()[0].strip()
        if gen.isdigit():
            return f"PCIe Gen {gen}"
    except Exception:
        pass
    return None


def detect_topology() -> str | None:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "topo", "-m", "--no-color"], text=True
        )
    except Exception:
        pass
    try:
        out = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
        return re.sub(r"\x1b\[[0-9;]*m", "", out)
    except Exception:
        return None


def detect_intra_node_interconnect() -> str | None:
    """Returns 'NVLink' when nvidia-smi topology contains NV# fabric links."""
    for cmd in (["nvidia-smi", "topo", "-m", "--no-color"], ["nvidia-smi", "topo", "-m"]):
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        except Exception:
            continue
        if re.search(r"\bNV\d+\b", out):
            return "NVLink"
    return None


def diagnostics(env: dict, accelerators: list[dict]) -> list[str]:
    notes: list[str] = []
    pytorch_v = env.get("pytorch_version") or ""
    runtime = env.get("runtime_version") or ""
    pcie = env.get("pcie_generation") or ""

    if pytorch_v == "unknown":
        notes.append(
            "PyTorch is not installed — pytorch_version is unknown. For GPU stack "
            "metadata: pip install torch (match your CUDA environment)."
        )
    if runtime == "unknown":
        notes.append(
            "Could not detect CUDA/runtime (tried PyTorch CUDA, nvcc, CUDA_HOME, "
            "nvidia-smi paths). runtime_version is unknown — install a CUDA toolkit "
            "or PyTorch with CUDA."
        )
    if pcie == "unknown":
        notes.append(
            "Could not read PCIe generation from nvidia-smi — pcie_generation is unknown."
        )
    if env.get("accelerator_topology") is None and accelerators:
        notes.append(
            "accelerator_topology is null — nvidia-smi topo did not return data."
        )
    return notes
