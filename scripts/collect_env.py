"""
AccelMark Environment Collector
Automatically collects hardware and software environment information.
Called automatically by the benchmark script — no need to run manually.

Usage:
    python scripts/collect_env.py --output ./results/community/<dir>/env_info.json
"""

import argparse
import json
import os
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
        out = subprocess.check_output(
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram",
             "--showdriverversion", "--json"],
            text=True, stderr=subprocess.DEVNULL
        )
        data = json.loads(out)
        accelerators = []
        for idx, (card_id, info) in enumerate(data.items()):
            # Skip non-card keys (e.g. "system" metadata in some versions)
            if not isinstance(info, dict):
                continue
            # Field names vary across rocm-smi versions — try all known variants
            name = (
                info.get("Card Series") or
                info.get("Card series") or
                info.get("Product Name") or
                info.get("product_name") or
                "AMD GPU"
            )
            mem_bytes = int(
                info.get("VRAM Total Memory (B)") or
                info.get("vram_total_memory_b") or
                info.get("VRAM Total Memory") or
                0
            )
            driver = (
                info.get("Driver version") or
                info.get("driver_version") or
                info.get("Driver Version") or
                "unknown"
            )
            accelerators.append({
                "index": idx,
                "name": name,
                "memory_gb": round(mem_bytes / (1024**3), 1),
                "driver_version": driver,
                "firmware_version": None,
            })
        return accelerators
    except Exception:
        return []


def collect_ascend() -> list[dict]:
    try:
        # npu-smi info -l lists all NPUs with detailed info
        out = subprocess.check_output(
            ["npu-smi", "info", "-l"],
            text=True, stderr=subprocess.DEVNULL
        )
        import re
        accelerators = []
        current_npu: dict | None = None

        for line in out.splitlines():
            # New NPU entry — line like "NPU ID         : 0"
            npu_match = re.search(r'NPU\s+ID\s*:\s*(\d+)', line, re.IGNORECASE)
            if npu_match:
                if current_npu:
                    accelerators.append(current_npu)
                current_npu = {
                    "index": int(npu_match.group(1)),
                    "name": "Huawei Ascend NPU",
                    "memory_gb": None,
                    "driver_version": _get_cann_version(),
                    "firmware_version": None,
                }

            if current_npu is None:
                continue

            # Chip name — line like "Chip Name      : 910B"
            chip_match = re.search(r'Chip\s+Name\s*:\s*(.+)', line, re.IGNORECASE)
            if chip_match:
                current_npu["name"] = f"Huawei Ascend {chip_match.group(1).strip()}"

            # Memory — line like "HBM Capacity(MB): 65536"
            mem_match = re.search(r'HBM\s+Capacity.*?:\s*(\d+)', line, re.IGNORECASE)
            if mem_match:
                current_npu["memory_gb"] = round(int(mem_match.group(1)) / 1024, 1)

            # Also try "Memory Capacity" format
            mem_match2 = re.search(r'Memory\s+Capacity.*?:\s*(\d+)\s*MB', line, re.IGNORECASE)
            if mem_match2 and current_npu["memory_gb"] is None:
                current_npu["memory_gb"] = round(int(mem_match2.group(1)) / 1024, 1)

        if current_npu:
            accelerators.append(current_npu)

        if accelerators:
            return accelerators

        # Fallback: try basic npu-smi info without -l
        out2 = subprocess.check_output(
            ["npu-smi", "info"], text=True, stderr=subprocess.DEVNULL
        )
        return [{
            "index": 0,
            "name": "Huawei Ascend NPU",
            "memory_gb": None,
            "driver_version": _get_cann_version(),
            "firmware_version": None,
        }]

    except Exception:
        return []


def _get_cann_version() -> str:
    """Get CANN (Compute Architecture for Neural Networks) version."""
    # Try npu-smi for driver/CANN version
    try:
        out = subprocess.check_output(
            ["npu-smi", "info", "-t", "common", "-i", "0"],
            text=True, stderr=subprocess.DEVNULL
        )
        import re
        for line in out.splitlines():
            match = re.search(r'(CANN|Driver)\s+Version\s*:\s*(.+)', line, re.IGNORECASE)
            if match:
                return match.group(2).strip()
    except Exception:
        pass
    # Try reading from CANN install path
    for cann_path in ["/usr/local/Ascend/ascend-toolkit/latest", "/usr/local/Ascend/nnae/latest"]:
        version_file = Path(cann_path) / "version.cfg"
        if version_file.exists():
            try:
                content = version_file.read_text()
                import re
                match = re.search(r'Version=(.+)', content)
                if match:
                    return f"CANN {match.group(1).strip()}"
            except Exception:
                pass
    return "unknown"


def collect_apple() -> list[dict]:
    """Detect Apple Silicon chips (M1/M2/M3/M4 series)."""
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True, stderr=subprocess.DEVNULL
        ).strip()
        if "Apple" not in chip:
            return []
        # Unified memory size
        mem_bytes = int(subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"],
            text=True, stderr=subprocess.DEVNULL
        ).strip())
        os_version = platform.mac_ver()[0]
        return [{
            "index": 0,
            "name": chip,           # e.g. "Apple M2 Ultra"
            "memory_gb": round(mem_bytes / (1024**3), 1),
            "driver_version": os_version,
            "firmware_version": None,
        }]
    except Exception:
        return []


def collect_topology() -> str | None:
    # Use --no-color flag for nvidia-smi to strip ANSI escape codes from output
    for cmd in [["nvidia-smi", "topo", "-m", "--no-color"], ["rocm-smi", "--showtopo"]]:
        try:
            return subprocess.check_output(cmd, text=True)
        except Exception:
            continue
    # Fallback: try without --no-color and strip ANSI manually
    try:
        import re
        out = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
        ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
        return ansi_escape.sub('', out)
    except Exception:
        return None


def collect_cpu() -> dict:
    # Try to get real CPU model name from /proc/cpuinfo (Linux)
    cpu_model = _get_cpu_model()

    try:
        import psutil
        return {
            "model": cpu_model,
            "physical_cores": psutil.cpu_count(logical=False) or 1,
            "logical_cores": psutil.cpu_count(logical=True) or 1,
            "numa_nodes": _get_numa_nodes(),
        }
    except ImportError:
        return {
            "model": cpu_model,
            "physical_cores": 1,
            "logical_cores": 1,
            "numa_nodes": _get_numa_nodes(),
        }


def _get_cpu_model() -> str:
    # Linux: read from /proc/cpuinfo
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    # macOS fallback
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True, stderr=subprocess.DEVNULL
        )
        return out.strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _get_numa_nodes() -> int:
    # Try lscpu for NUMA node count
    try:
        out = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if "NUMA node(s)" in line:
                return int(line.split(":")[1].strip())
    except Exception:
        pass
    # Fallback: count /sys/devices/system/node/node* directories
    try:
        nodes = list(Path("/sys/devices/system/node").glob("node[0-9]*"))
        if nodes:
            return len(nodes)
    except Exception:
        pass
    return 1


def collect_memory_gb() -> float:
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        return 0.0


def detect_pcie_gen() -> str:
    # Try nvidia-smi for PCIe generation
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=pcie.link.gen.current",
             "--format=csv,noheader"],
            text=True
        )
        gen = out.strip().splitlines()[0].strip()
        if gen.isdigit():
            return f"PCIe Gen {gen}"
    except Exception:
        pass
    return "unknown"


def detect_runtime_version() -> str:
    # Try torch first — most reliable when vLLM is installed
    try:
        import torch
        if torch.version.cuda:
            return f"CUDA {torch.version.cuda}"
    except ImportError:
        pass

    # Try nvcc
    try:
        out = subprocess.check_output(
            ["nvcc", "--version"],
            text=True, stderr=subprocess.STDOUT
        )
        for line in out.splitlines():
            if "release" in line.lower():
                # e.g. "Cuda compilation tools, release 12.2, V12.2.140"
                parts = line.split("release")
                if len(parts) > 1:
                    version = parts[1].split(",")[0].strip()
                    return f"CUDA {version}"
    except Exception:
        pass

    # Try reading from CUDA_HOME
    for env_var in ["CUDA_HOME", "CUDA_PATH"]:
        cuda_home = os.environ.get(env_var)
        if cuda_home:
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

    # Try ROCm
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--version"],
            text=True, stderr=subprocess.STDOUT
        )
        return f"ROCm {out.strip().splitlines()[-1]}"
    except Exception:
        pass

    return "unknown"


def collect_network_interfaces() -> list[dict] | None:
    """Collect high-speed network interfaces (InfiniBand, RoCE)."""
    interfaces = []

    # Check for InfiniBand via ibstat
    try:
        out = subprocess.check_output(
            ["ibstat"], text=True, stderr=subprocess.DEVNULL
        )
        # Count CA (Channel Adapter) entries
        import re
        cas = re.findall(r"^CA '(.+)'", out, re.MULTILINE)
        for ca in cas:
            interfaces.append({
                "name": ca,
                "type": "InfiniBand",
                "bandwidth_gbps": None,  # would need ibstatus for detailed info
            })
    except Exception:
        pass

    # Check for mlx5 devices from topology output (already collected)
    # These appear as NIC0: mlx5_0 etc. in nvidia-smi topo output
    # We just note their presence here
    try:
        out = subprocess.check_output(
            ["ls", "/sys/class/infiniband"],
            text=True, stderr=subprocess.DEVNULL
        )
        for dev in out.strip().splitlines():
            if dev and not any(i["name"] == dev for i in interfaces):
                interfaces.append({
                    "name": dev,
                    "type": "InfiniBand/RoCE",
                    "bandwidth_gbps": None,
                })
    except Exception:
        pass

    return interfaces if interfaces else None


def detect_pytorch_version() -> str:
    try:
        import torch
        return torch.__version__
    except ImportError:
        return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print("Collecting environment info...")

    # Try each vendor in order
    accelerators = (
        collect_nvidia() or
        collect_amd() or
        collect_ascend() or
        collect_apple() or
        []
    )
    if not accelerators:
        print("WARNING: No accelerators detected. Collecting CPU-only info.")

    env = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "accelerators": accelerators,
        "accelerator_topology": collect_topology(),
        "cpu": collect_cpu(),
        "system_memory_gb": collect_memory_gb(),
        "pcie_generation": detect_pcie_gen(),
        "cpu_accelerator_bandwidth_gbs": None,
        "network_interfaces": collect_network_interfaces(),
        "kernel_version": platform.release(),
        "runtime_version": detect_runtime_version(),
        "pytorch_version": detect_pytorch_version(),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(env, f, indent=2)
    print(f"Environment info written to {out_path}")

    # Print summary
    chip_names = [a["name"] for a in accelerators]
    print(f"  Accelerators: {chip_names}")
    print(f"  CPU: {env['cpu']['model']}")
    print(f"  Memory: {env['system_memory_gb']} GB")
    print(f"  Runtime: {env['runtime_version']}")
    print(f"  PyTorch: {env['pytorch_version']}")
    print(f"  PCIe: {env['pcie_generation']}")
    if env["network_interfaces"]:
        print(f"  Network: {[n['name'] for n in env['network_interfaces']]}")


if __name__ == "__main__":
    main()