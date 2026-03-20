"""
AccelMark Environment Collector
Automatically collects hardware and software environment information.
Run this before every benchmark submission.

Usage:
    python scripts/collect_env.py --output ./my_submission/env_info.json
"""

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def collect_nvidia() -> list[dict]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            text=True
        )
        accelerators = []
        for line in out.strip().splitlines():
            idx, name, mem, driver = [x.strip() for x in line.split(",")]
            accelerators.append({
                "index": int(idx),
                "name": name,
                "memory_gb": round(float(mem) / 1024, 1),
                "driver_version": driver,
                "firmware_version": None,
            })
        return accelerators
    except Exception:
        return []


def collect_amd() -> list[dict]:
    try:
        out = subprocess.check_output(["rocm-smi", "--showmeminfo", "vram", "--json"], text=True)
        data = json.loads(out)
        accelerators = []
        for idx, (card, info) in enumerate(data.items()):
            accelerators.append({
                "index": idx,
                "name": info.get("Card series", "AMD GPU"),
                "memory_gb": round(int(info.get("VRAM Total Memory (B)", 0)) / (1024**3), 1),
                "driver_version": info.get("Driver version", "unknown"),
                "firmware_version": None,
            })
        return accelerators
    except Exception:
        return []


def collect_ascend() -> list[dict]:
    try:
        out = subprocess.check_output(["npu-smi", "info"], text=True)
        # Basic parsing — Ascend output format varies by CANN version
        accelerators = [{"index": 0, "name": "Huawei Ascend NPU",
                         "memory_gb": None, "driver_version": "unknown", "firmware_version": None}]
        return accelerators
    except Exception:
        return []


def collect_topology() -> str | None:
    for cmd in [["nvidia-smi", "topo", "-m"], ["rocm-smi", "--showtopo"]]:
        try:
            return subprocess.check_output(cmd, text=True)
        except Exception:
            continue
    return None


def collect_cpu() -> dict:
    try:
        import psutil
        return {
            "model": platform.processor() or "unknown",
            "physical_cores": psutil.cpu_count(logical=False) or 1,
            "logical_cores": psutil.cpu_count(logical=True) or 1,
            "numa_nodes": 1,  # psutil doesn't expose NUMA directly
        }
    except ImportError:
        return {
            "model": platform.processor() or "unknown",
            "physical_cores": 1,
            "logical_cores": 1,
            "numa_nodes": 1,
        }


def collect_memory_gb() -> float:
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        return 0.0


def detect_runtime_version() -> str:
    for cmd, label in [
        (["nvcc", "--version"], "CUDA"),
        (["rocm-smi", "--version"], "ROCm"),
    ]:
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
            return out.strip().splitlines()[-1]
        except Exception:
            continue
    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print("Collecting environment info...")

    # Try each vendor
    accelerators = collect_nvidia() or collect_amd() or collect_ascend()
    if not accelerators:
        print("WARNING: No accelerators detected. Collecting CPU-only info.")

    env = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "accelerators": accelerators,
        "accelerator_topology": collect_topology(),
        "cpu": collect_cpu(),
        "system_memory_gb": collect_memory_gb(),
        "pcie_generation": "unknown",
        "cpu_accelerator_bandwidth_gbs": None,
        "network_interfaces": None,
        "kernel_version": platform.release(),
        "runtime_version": detect_runtime_version(),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(env, f, indent=2)
    print(f"Environment info written to {out_path}")


if __name__ == "__main__":
    main()
