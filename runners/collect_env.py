"""
AccelMark Environment Collector
===============================

Detects the host's accelerators, CPU, memory, runtime stack, network
fabric, and OS, and writes a normalised JSON file used to qualify and
contextualise benchmark results.

This script is intentionally *vendor-agnostic*: every accelerator family
is implemented as an independent plug-in under ``runners/platforms/``.
The top-level collector here only handles the parts that are common to
all platforms (CPU model, system memory, OS, network interfaces) plus
some lightweight orchestration:

* discover all plug-ins
* call ``collect()`` in priority order; the first plug-in that returns
  accelerator records becomes the *active* platform
* ask the active platform for its runtime version, PCIe generation,
  topology and intra-node interconnect; fall back to the union of
  remaining plug-ins when the active one declines to answer
* aggregate per-plug-in diagnostics into warning messages

To support a new accelerator family, drop a new file at
``runners/platforms/<my_platform>.py`` exporting the optional functions
documented in ``runners/platforms/__init__.py``. **No change to this
file is required.**

Usage:
    python runners/collect_env.py --output ./results/community/<dir>/env_info.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Iterable

# Make ``runners.platforms`` importable when this script is run directly.
_RUNNERS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _RUNNERS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from runners.platforms import discover_plugins  # noqa: E402


def _print_warning(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def _have_psutil() -> bool:
    try:
        import psutil  # noqa: F401
    except ImportError:
        return False
    return True


# ─── Plug-in invocation helpers ───────────────────────────────────────────────


def _call_optional(mod: ModuleType, name: str, *args, default=None):
    """Call an optional plug-in function defensively.

    Plug-ins are third-party-ish code paths: any exception from one
    must not break the whole environment report. Missing attributes are
    treated identically to functions that returned ``default``.
    """
    fn = getattr(mod, name, None)
    if fn is None:
        return default
    try:
        return fn(*args)
    except Exception:
        return default


def _collect_accelerators(plugins: Iterable[ModuleType]) -> tuple[list[dict], ModuleType | None]:
    """Try each plug-in's ``collect()`` and return the first non-empty result."""
    for mod in plugins:
        result = _call_optional(mod, "collect", default=[]) or []
        if result:
            return list(result), mod
    return [], None


def _detect_first(plugins: Iterable[ModuleType], fn_name: str, active: ModuleType | None) -> str | None:
    """Return the first non-empty answer to ``fn_name()`` across plug-ins.

    The active plug-in (the one whose ``collect()`` succeeded) is tried
    first so vendor-specific information is preferred over generic
    fallbacks.
    """
    ordered: list[ModuleType] = []
    if active is not None:
        ordered.append(active)
    ordered.extend(p for p in plugins if p is not active)
    for mod in ordered:
        answer = _call_optional(mod, fn_name)
        if answer:
            return answer
    return None


# ─── CPU / memory / OS — vendor-independent ─────────────────────────────────


def _get_cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        # aarch64: /proc/cpuinfo has no "model name" — try "Hardware" field
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("Hardware"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    try:
        out = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if line.startswith("Model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _get_numa_nodes() -> int:
    try:
        out = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if "NUMA node(s)" in line:
                return int(line.split(":")[1].strip())
    except Exception:
        pass
    try:
        nodes = list(Path("/sys/devices/system/node").glob("node[0-9]*"))
        if nodes:
            return len(nodes)
    except Exception:
        pass
    return 1


def collect_cpu() -> dict:
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


def collect_memory_gb() -> float:
    try:
        import psutil

        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        pass
    if platform.system() == "Darwin":
        try:
            mem = int(
                subprocess.check_output(
                    ["sysctl", "-n", "hw.memsize"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            )
            return round(mem / (1024 ** 3), 1)
        except Exception:
            pass
    if platform.system() == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return round(kb / (1024 ** 2), 1)
        except Exception:
            pass
    return 0.0


def collect_network_interfaces() -> list[dict] | None:
    """Collect high-speed network interfaces (InfiniBand, RoCE)."""
    interfaces: list[dict] = []
    try:
        out = subprocess.check_output(
            ["ibstat"], text=True, stderr=subprocess.DEVNULL
        )
        import re

        cas = re.findall(r"^CA '(.+)'", out, re.MULTILINE)
        for ca in cas:
            interfaces.append(
                {"name": ca, "type": "InfiniBand", "bandwidth_gbps": None}
            )
    except Exception:
        pass

    try:
        out = subprocess.check_output(
            ["ls", "/sys/class/infiniband"], text=True, stderr=subprocess.DEVNULL
        )
        for dev in out.strip().splitlines():
            if dev and not any(i["name"] == dev for i in interfaces):
                interfaces.append(
                    {"name": dev, "type": "InfiniBand/RoCE", "bandwidth_gbps": None}
                )
    except Exception:
        pass

    return interfaces if interfaces else None


def detect_os_version() -> str:
    try:
        with open("/etc/os-release") as f:
            info: dict[str, str] = {}
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    info[k] = v.strip('"')
            if "PRETTY_NAME" in info:
                return info["PRETTY_NAME"]
    except Exception:
        pass
    mac_ver = platform.mac_ver()[0]
    if mac_ver:
        return f"macOS {mac_ver}"
    return platform.platform()


def detect_python_version() -> str:
    return platform.python_version()


def detect_pytorch_version() -> str:
    try:
        import torch

        return torch.__version__
    except ImportError:
        return "unknown"


# ─── Warning aggregation ────────────────────────────────────────────────────


def _emit_global_warnings(env: dict, accelerators: list[dict]) -> None:
    """Warnings independent of any specific accelerator family."""
    if not _have_psutil():
        _print_warning(
            "Package 'psutil' is not installed — CPU physical/logical core counts may "
            "default to 1; install with: pip install psutil"
        )
    if float(env.get("system_memory_gb") or 0) == 0.0:
        _print_warning(
            "system_memory_gb is 0 — RAM could not be determined. "
            "Install psutil (pip install psutil) or ensure /proc/meminfo (Linux) / "
            "sysctl hw.memsize (macOS) is available."
        )
    if not accelerators and (
        os.environ.get("TPU_NAME") or os.environ.get("CLOUD_TPU_TASK")
    ):
        _print_warning(
            "TPU-related environment variables are set but no TPU devices were detected — "
            "install jax / tpu_inference when running on Cloud TPU."
        )


# ─── Main orchestration ─────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print("Collecting environment info...")

    plugins = discover_plugins()
    accelerators, active = _collect_accelerators(plugins)
    if not accelerators:
        _print_warning("No accelerators detected. Collecting CPU-only info.")

    runtime_version = _detect_first(plugins, "detect_runtime_version", active) or "unknown"
    pcie_generation = _detect_first(plugins, "detect_pcie_gen", active) or "unknown"
    topology = _detect_first(plugins, "detect_topology", active)
    intra_node = _detect_first(plugins, "detect_intra_node_interconnect", active)

    env = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "accelerators": accelerators,
        "accelerator_platform": getattr(active, "ID", None) if active else None,
        "accelerator_topology": topology,
        "intra_node_interconnect": intra_node,
        "cpu": collect_cpu(),
        "system_memory_gb": collect_memory_gb(),
        "pcie_generation": pcie_generation,
        "cpu_accelerator_bandwidth_gbs": None,
        "network_interfaces": collect_network_interfaces(),
        "os": detect_os_version(),
        "python_version": detect_python_version(),
        "kernel_version": platform.release(),
        "runtime_version": runtime_version,
        "pytorch_version": detect_pytorch_version(),
    }

    _emit_global_warnings(env, accelerators)
    # Plug-in-specific diagnostics — only the *active* plug-in (the one
    # whose accelerators were collected) gets to emit hardware-specific
    # advice, so we do not produce a wall of irrelevant warnings on
    # hosts that happen to have e.g. nvidia-smi installed but no GPU.
    if active is not None:
        for note in _call_optional(active, "diagnostics", env, accelerators, default=[]) or []:
            _print_warning(note)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(env, f, indent=2)
    print(f"Environment info written to {out_path}")

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
