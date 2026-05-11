"""
AccelMark Environment Collector
Automatically collects hardware and software environment information.
Called automatically by the benchmark script — no need to run manually.

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


def _print_warning(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def _have_psutil() -> bool:
    try:
        import psutil  # noqa: F401
    except ImportError:
        return False
    return True


def _warn_for_incomplete_env_report(env: dict, accelerators: list[dict]) -> None:
    """Emit warnings when the report used fallbacks or missing optional tools."""
    if not _have_psutil():
        _print_warning(
            "Package 'psutil' is not installed — CPU physical/logical core counts may "
            "default to 1; install with: pip install psutil"
        )

    mem = float(env.get("system_memory_gb") or 0)
    if mem == 0.0:
        _print_warning(
            "system_memory_gb is 0 — RAM could not be determined. "
            "Install psutil (pip install psutil) or ensure /proc/meminfo (Linux) / "
            "sysctl hw.memsize (macOS) is available."
        )

    if not accelerators:
        if os.environ.get("TPU_NAME") or os.environ.get("CLOUD_TPU_TASK"):
            _print_warning(
                "TPU-related environment variables are set but no TPU devices were detected — "
                "install jax / tpu_inference when running on Cloud TPU."
            )
        return

    vendor = (accelerators[0].get("vendor") or "").strip()
    pytorch_v = env.get("pytorch_version") or ""
    runtime = env.get("runtime_version") or ""
    pcie = env.get("pcie_generation") or ""

    if pytorch_v == "unknown" and vendor in ("NVIDIA", "AMD", "Huawei"):
        _print_warning(
            "PyTorch is not installed — pytorch_version is unknown. "
            "For GPU stack metadata: pip install torch (match your CUDA/ROCm/CANN environment)."
        )

    if runtime == "unknown":
        if vendor == "NVIDIA":
            _print_warning(
                "Could not detect CUDA/runtime (tried PyTorch CUDA, nvcc, CUDA_HOME, nvidia-smi paths). "
                "runtime_version is unknown — install a CUDA toolkit or PyTorch with CUDA."
            )
        elif vendor == "AMD":
            _print_warning(
                "Could not detect ROCm runtime (rocm-smi / PyTorch ROCm). "
                "runtime_version is unknown."
            )
        elif vendor == "Huawei":
            _print_warning(
                "Could not detect CANN/runtime from npu-smi / install paths. "
                "runtime_version is unknown."
            )
        elif vendor == "Google":
            _print_warning(
                "Could not detect JAX/runtime for TPU. runtime_version is unknown — "
                "install jax if you use Cloud TPU."
            )

    if pcie == "unknown" and vendor == "NVIDIA":
        _print_warning(
            "Could not read PCIe generation from nvidia-smi — pcie_generation is unknown."
        )

    if env.get("accelerator_topology") is None and vendor in ("NVIDIA", "AMD"):
        _print_warning(
            "accelerator_topology is null — nvidia-smi topo / rocm-smi --showtopo did not return data."
        )

    if vendor == "Apple":
        if runtime.startswith("macOS ") and "MLX" not in runtime and "Metal MPS" not in runtime:
            _print_warning(
                "Neither MLX nor PyTorch with MPS is available — runtime_version only reflects "
                "macOS build. For ML stack: pip install mlx  or  pip install torch (with MPS)."
            )

    if vendor == "Huawei":
        for a in accelerators:
            if a.get("memory_gb") is None:
                try:
                    import torch_npu  # noqa: F401
                except ImportError:
                    _print_warning(
                        "Ascend HBM memory could not be parsed from npu-smi — optional "
                        "pip install torch_npu may fill memory_gb via the runtime API."
                    )
                else:
                    _print_warning(
                        "Ascend HBM memory_gb is still unknown (torch_npu is importable) — "
                        "check ASCEND_VISIBLE_DEVICES, driver, and npu-smi output."
                    )
                break


def collect_nvidia() -> list[dict]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version,compute_cap",
             "--format=csv,noheader,nounits"],
            text=True
        )
        accelerators = []
        for line in out.strip().splitlines():
            idx, name, mem, driver, compute_cap = [x.strip() for x in line.split(",")]
            try:
                cc_float = float(compute_cap) if compute_cap else 0.0
                supports_bf16 = cc_float >= 8.0
            except (ValueError, TypeError):
                supports_bf16 = True  # unknown — assume capable, runner will handle
            accelerators.append({
                "index": int(idx),
                "name": name,
                "vendor": "NVIDIA",
                "memory_gb": round(float(mem) / 1024, 1),
                "driver_version": driver,
                "firmware_version": None,
                "compute_capability": compute_cap,
                "supports_bf16": supports_bf16,
            })
        return accelerators
    except Exception:
        return []


# Known AMD architectures with BF16 support
_AMD_BF16_SUPPORTED = {
    "cdna2", "cdna3",                          # MI200, MI300 series
    "rdna3", "rdna4",                          # RX 7000+ series
    "gfx90a",                                  # MI250X arch code
    "gfx940", "gfx941", "gfx942",             # MI300 arch codes
    "gfx1100", "gfx1101", "gfx1102",          # RDNA3 arch codes
}

# Known AMD architectures WITHOUT BF16
_AMD_NO_BF16 = {
    "cdna1",                                   # MI100
    "rdna", "rdna1", "rdna2",                 # RX 5000, RX 6000 series
    "gfx908",                                  # MI100 arch code
    "gfx1030", "gfx1031",                     # RDNA2 arch codes
}


def _amd_supports_bf16(arch_str: str) -> bool:
    """Determine BF16 support from AMD architecture string."""
    if not arch_str:
        return True   # unknown — assume capable
    arch_lower = arch_str.lower()
    for known in _AMD_BF16_SUPPORTED:
        if known in arch_lower:
            return True
    for known in _AMD_NO_BF16:
        if known in arch_lower:
            return False
    return True   # unrecognized — assume capable


def collect_amd() -> list[dict]:
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram",
             "--showdriverversion", "--json"],
            text=True, stderr=subprocess.DEVNULL
        )
        data = json.loads(out)

        # Try to get architecture string for BF16 detection
        arch_str = ""
        try:
            arch_out = subprocess.check_output(
                ["rocm-smi", "--showallinfo"],
                text=True, stderr=subprocess.DEVNULL
            )
            import re as _re
            gfx_matches = _re.findall(r'gfx\d+[a-z]?', arch_out.lower())
            arch_str = gfx_matches[0] if gfx_matches else ""
        except Exception:
            pass

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
                "vendor": "AMD",
                "memory_gb": round(mem_bytes / (1024**3), 1),
                "driver_version": driver,
                "firmware_version": None,
                "supports_bf16": _amd_supports_bf16(arch_str),
            })
        return accelerators
    except Exception:
        return []


_ASCEND_BF16_SUPPORTED = {
    "910b", "atlas 800t a2", "910b1", "910b2", "910b3", "910b4",
}
_ASCEND_NO_BF16 = {
    "310", "310p", "atlas 300",
}


def _ascend_supports_bf16(chip_name: str) -> bool:
    if not chip_name:
        return True
    name_lower = chip_name.lower()
    for known in _ASCEND_BF16_SUPPORTED:
        if known in name_lower:
            return True
    for known in _ASCEND_NO_BF16:
        if known in name_lower:
            return False
    return True   # unknown Ascend chip — assume capable


def _ascend_enrich_via_torch_npu(accelerators: list[dict]) -> None:
    """Backfill memory_gb and name via torch_npu runtime API.

    torch_npu.npu.get_device_properties(i) mirrors torch.cuda:
      - .total_memory  — total HBM bytes
      - .name          — chip name string (e.g. "910B2")

    Logical indices 0..N-1 map positionally to npu-smi enumeration order
    when all devices are visible (respects ASCEND_VISIBLE_DEVICES masking).
    Only fills fields still None so parsed values are never overwritten.
    """
    try:
        import torch_npu
        logical_count = torch_npu.npu.device_count()
    except Exception:
        return
    for logical_idx in range(min(logical_count, len(accelerators))):
        rec = accelerators[logical_idx]
        try:
            props = torch_npu.npu.get_device_properties(logical_idx)
            if rec.get("memory_gb") is None and props.total_memory:
                rec["memory_gb"] = round(props.total_memory / (1024 ** 3), 1)
            if rec.get("name") in (None, "Huawei Ascend NPU") and props.name:
                rec["name"] = f"Huawei Ascend {props.name.strip()}"
        except Exception:
            continue


def _parse_npu_smi_table(out: str, cann_version: str) -> list[dict]:
    """Parse the tabular output of plain `npu-smi info`.

    The table format has two data rows per device:
      Row 1: | <NPU_ID>  <ChipName>  | <Health> | <Power> <Temp> <Hugepages> |
      Row 2: | <ChipID>              | <Bus-Id> | <AICore> <Mem-Usage>  <HBM-Usage(used/total MB)> |

    Example:
      | 7     910B2               | OK            | 96.5        49                0    / 0             |
      | 0                         | 0000:42:00.0  | 0           0    / 0          3389 / 65536         |
    """
    import re
    accelerators = []
    lines = out.splitlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        # Row 1: starts with "| <int>  <ChipName>" — NPU ID and chip name
        row1 = re.match(r'\|\s*(\d+)\s+(\S+)\s*\|', line)
        if row1:
            npu_id = int(row1.group(1))
            chip_name = row1.group(2).strip()
            # Row 2 is the very next table row — HBM total is second number in
            # the last "used / total" pair on that line
            hbm_total_mb = None
            if i + 1 < len(lines):
                row2 = lines[i + 1]
                # Match "used / total" at end of line, e.g. "3389 / 65536"
                hbm_match = re.search(r'(\d+)\s*/\s*(\d+)\s*\|?\s*$', row2)
                if hbm_match:
                    hbm_total_mb = int(hbm_match.group(2))
                i += 1  # consume row 2

            memory_gb = round(hbm_total_mb / 1024, 1) if hbm_total_mb else None
            name = f"Huawei Ascend {chip_name}" if chip_name else "Huawei Ascend NPU"
            accelerators.append({
                "index": npu_id,
                "name": name,
                "vendor": "Huawei",
                "memory_gb": memory_gb,
                "driver_version": cann_version,
                "firmware_version": None,
                "supports_bf16": _ascend_supports_bf16(name),
            })
        i += 1

    return accelerators


def collect_ascend() -> list[dict]:
    import re

    try:
        # Primary: plain `npu-smi info` — tabular format with chip name + HBM.
        # `npu-smi info -l` only returns NPU ID and Chip Count on some firmware
        # versions (e.g. 24.1.x on openEuler/aarch64) so it is not reliable.
        out = subprocess.check_output(
            ["npu-smi", "info"], text=True, stderr=subprocess.DEVNULL
        )
        # Parse with a placeholder driver/firmware — filled in per-device below
        accelerators = _parse_npu_smi_table(out, "unknown")

        if not accelerators:
            # Secondary: try -l in case this firmware uses key-value format
            out_l = subprocess.check_output(
                ["npu-smi", "info", "-l"], text=True, stderr=subprocess.DEVNULL
            )
            current_npu: dict | None = None
            for line in out_l.splitlines():
                npu_match = re.search(r'NPU\s+ID\s*:\s*(\d+)', line, re.IGNORECASE)
                if npu_match:
                    if current_npu:
                        current_npu["supports_bf16"] = _ascend_supports_bf16(current_npu.get("name", ""))
                        accelerators.append(current_npu)
                    current_npu = {
                        "index": int(npu_match.group(1)),
                        "name": "Huawei Ascend NPU",
                        "vendor": "Huawei",
                        "memory_gb": None,
                        "driver_version": "unknown",
                        "firmware_version": None,
                    }
                if current_npu is None:
                    continue
                chip_match = re.search(r'Chip\s+Name\s*:\s*(.+)', line, re.IGNORECASE)
                if chip_match:
                    current_npu["name"] = f"Huawei Ascend {chip_match.group(1).strip()}"
                mem_match = re.search(r'HBM\s+Capacity.*?:\s*(\d+)', line, re.IGNORECASE)
                if mem_match:
                    current_npu["memory_gb"] = round(int(mem_match.group(1)) / 1024, 1)
                if current_npu["memory_gb"] is None:
                    mem_match2 = re.search(r'Memory\s+Capacity.*?:\s*(\d+)\s*MB', line, re.IGNORECASE)
                    if mem_match2:
                        current_npu["memory_gb"] = round(int(mem_match2.group(1)) / 1024, 1)
                if current_npu["firmware_version"] is None:
                    fw_match = re.search(r'Firmware\s+Version\s*:\s*(.+)', line, re.IGNORECASE)
                    if fw_match:
                        current_npu["firmware_version"] = fw_match.group(1).strip()
            if current_npu:
                current_npu["supports_bf16"] = _ascend_supports_bf16(current_npu.get("name", ""))
                accelerators.append(current_npu)

        if accelerators:
            # Enrich driver_version and firmware_version per device via -t board
            for rec in accelerators:
                board_info = _get_npu_board_info(str(rec["index"]))
                rec["driver_version"] = board_info["driver_version"]
                rec["firmware_version"] = board_info["firmware_version"]
            # Enrich any still-missing memory/name via torch_npu runtime API
            _ascend_enrich_via_torch_npu(accelerators)
            return accelerators

    except Exception:
        pass

    return []


def _get_npu_board_info(npu_id: str) -> dict:
    """Query driver version and firmware version for a single NPU via -t board.

    `npu-smi info -t board -i <NPU_ID>` returns key-value fields including:
      Software Version  : 24.1.0.3   (driver / npu-smi package version)
      Firmware Version  : NA          (NA means not available on this board)

    Returns dict with keys "driver_version" and "firmware_version".
    Falls back to CANN install-path files for driver_version if the command
    fails or produces no match.
    """
    import re

    result = {"driver_version": "unknown", "firmware_version": None}

    try:
        out = subprocess.check_output(
            ["npu-smi", "info", "-t", "board", "-i", npu_id],
            text=True, stderr=subprocess.DEVNULL
        )
        for line in out.splitlines():
            # Software Version is the driver/npu-smi package version
            sw_match = re.search(r'Software\s+Version\s*:\s*(.+)', line, re.IGNORECASE)
            if sw_match:
                result["driver_version"] = sw_match.group(1).strip()
            # Firmware Version — treat "NA" as not available
            fw_match = re.search(r'Firmware\s+Version\s*:\s*(.+)', line, re.IGNORECASE)
            if fw_match:
                fw = fw_match.group(1).strip()
                result["firmware_version"] = None if fw.upper() == "NA" else fw
    except Exception:
        pass

    # Fallback for driver_version: CANN toolkit install path
    if result["driver_version"] == "unknown":
        for cann_path in ["/usr/local/Ascend/ascend-toolkit/latest", "/usr/local/Ascend/nnae/latest"]:
            version_file = Path(cann_path) / "version.cfg"
            if version_file.exists():
                try:
                    text = version_file.read_text()
                    m = re.search(r'Version=(.+)', text)
                    if m:
                        result["driver_version"] = f"CANN {m.group(1).strip()}"
                        break
                except Exception:
                    pass

    return result

def _apple_supports_bf16(chip_name: str) -> bool:
    """M1 has limited/slow BF16. M2+ has full hardware BF16."""
    if not chip_name:
        return True
    name_lower = chip_name.lower()
    # M1 variants: "Apple M1", "Apple M1 Pro", "Apple M1 Max", "Apple M1 Ultra"
    if "m1" in name_lower and "m10" not in name_lower:  # avoid matching "m10x"
        return False
    return True  # M2, M3, M4 and unknown — assume supported


def _apple_silicon_brand() -> str | None:
    """Return SoC marketing name (e.g. 'Apple M3 Pro') if this is Apple Silicon, else None."""
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return chip if "Apple" in chip else None
    except Exception:
        return None


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


def _apple_metal_summary() -> str | None:
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


def collect_apple() -> list[dict]:
    """Detect Apple Silicon chips (M1/M2/M3/M4 series) in the local environment."""
    try:
        chip = _apple_silicon_brand()
        if not chip:
            return []
        mem_bytes = int(
            subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return [{
            "index": 0,
            "name": chip,
            "vendor": "Apple",
            "memory_gb": round(mem_bytes / (1024**3), 1),
            "driver_version": _macos_build_string(),
            "firmware_version": None,
            "compute_capability": None,
            "supports_bf16": _apple_supports_bf16(chip),
        }]
    except Exception:
        return []


def collect_tpu() -> list[dict]:
    """Detect Google Cloud TPU chips via tpu_info and JAX."""
    try:
        from tpu_inference import tpu_info

        num_chips = tpu_info.get_num_chips()
        tpu_type  = tpu_info.get_tpu_type()   # e.g. "v5e-1", "v5litepod-4", "v6e-8"
        node_name = tpu_info.get_node_name()  # GCE instance ID, or None on Colab

        if not num_chips or num_chips == 0:
            return []

        # Resolve human-readable chip name and HBM per chip.
        # tpu_type on Colab is "v5e-1" (not "v5litepod-1") — handled by the mapping.
        chip_name, memory_gb = _tpu_chip_name_and_memory(tpu_type)

        # get_num_cores_per_chip() in tpu_info checks for "v5litepod" or "v6e"
        # but Colab returns "v5e-1", so it falls through to the default of 2,
        # which is wrong. Correct it by checking the tpu_type string ourselves.
        t = (tpu_type or "").lower()
        if "v5litepod" in t or "v5e" in t or "v6e" in t or "trillium" in t:
            num_cores_per_chip = 1
        else:
            num_cores_per_chip = tpu_info.get_num_cores_per_chip()

        # Try to get more detail from JAX (version, memory stats)
        jax_version = "unknown"
        try:
            import jax
            jax_version = jax.__version__
            jax_devices = jax.devices()
            # jax.devices() on v5e-1:
            # [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)]
            if jax_devices and memory_gb is None:
                mem = getattr(jax_devices[0], "memory_stats", None)
                if mem and "bytes_limit" in mem:
                    memory_gb = round(mem["bytes_limit"] / (1024 ** 3), 1)
        except Exception:
            pass

        # Build one record per chip
        accelerators = []
        for i in range(num_chips):
            accelerators.append({
                "index":              i,
                "name":               chip_name,
                "vendor":             "Google",
                "memory_gb":          memory_gb,
                "driver_version":     f"JAX {jax_version}" if jax_version != "unknown" else "unknown",
                "firmware_version":   None,
                "tpu_type":           tpu_type,      # raw type string e.g. "v5e-1"
                "tpu_node_name":      node_name,     # None on Colab (GCE metadata 404)
                "num_cores_per_chip": num_cores_per_chip,
                "supports_bf16":      True,          # all modern TPU generations
            })
        return accelerators

    except Exception:
        return []


def _tpu_chip_name_and_memory(tpu_type: str | None) -> tuple[str, float | None]:
    """
    Map GCE accelerator-type string to a human-readable chip name and HBM size.

    tpu_type examples and their meanings:
      "v5litepod-1"  → v5e, 1 chip,  16 GiB HBM per chip
      "v5litepod-4"  → v5e, 4 chips, 16 GiB HBM per chip
      "v5litepod-8"  → v5e, 8 chips, 16 GiB HBM per chip
      "v6e-1"        → v6e (Trillium), 1 chip, 32 GiB HBM per chip
      "v6e-8"        → v6e, 8 chips, 32 GiB HBM per chip
      "v7x-1"        → v7x (Ironwood), 1 chip, 192 GiB HBM per chip
      "v4-8"         → v4, 8 chips, 32 GiB HBM per chip
      "v5p-8"        → v5p, 8 chips, 95 GiB HBM per chip

    Returns (chip_name, memory_gb_per_chip). memory_gb may be None for
    unknown types.
    """
    if not tpu_type:
        return "Google TPU", None

    t = tpu_type.lower()

    if "v5litepod" in t or "v5e" in t:
        return "Google TPU v5e", 16.0
    if "v6e" in t or "trillium" in t:
        return "Google TPU v6e", 32.0
    if "v7x" in t or "ironwood" in t:
        return "Google TPU v7x", 192.0
    if "v5p" in t:
        return "Google TPU v5p", 95.0
    if "v4" in t:
        return "Google TPU v4", 32.0
    if "v3" in t:
        return "Google TPU v3", 16.0
    if "v2" in t:
        return "Google TPU v2", 8.0

    # Unknown type — return the raw string as name, no memory info
    return f"Google TPU ({tpu_type})", None


def collect_topology() -> str | None:
    brand = _apple_silicon_brand()
    if brand:
        lines = [
            f"Apple Silicon — integrated GPU in {brand} (unified system memory).",
            f"Machine: {platform.machine()}",
        ]
        metal = _apple_metal_summary()
        if metal:
            lines.append(metal)
        return "\n".join(lines)

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
        # aarch64: /proc/cpuinfo has no "model name" — try "Hardware" field
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("Hardware"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    # Try lscpu for model name (works on aarch64)
    try:
        out = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if line.startswith("Model name"):
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
        pass
    # macOS / Apple Silicon: sysctl when psutil is not installed
    if platform.system() == "Darwin":
        try:
            mem = int(
                subprocess.check_output(
                    ["sysctl", "-n", "hw.memsize"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            )
            return round(mem / (1024**3), 1)
        except Exception:
            pass
    # Linux: MemTotal in kB when psutil is unavailable
    if platform.system() == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return round(kb / (1024**2), 1)
        except Exception:
            pass
    return 0.0


def detect_pcie_gen() -> str:
    if _apple_silicon_brand():
        return "SoC integrated (no discrete PCIe GPU)"

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

    # Try Ascend/CANN — reuse board info from npu-smi
    try:
        import re as _re
        info_out = subprocess.check_output(
            ["npu-smi", "info"], text=True, stderr=subprocess.DEVNULL
        )
        m = _re.search(r'\|\s*(\d+)\s+\S+\s*\|', info_out)
        if m:
            board = _get_npu_board_info(m.group(1))
            if board["driver_version"] != "unknown":
                return f"CANN {board['driver_version']}"
    except Exception:
        pass

    # Try JAX/TPU
    try:
        import jax
        return f"JAX {jax.__version__}"
    except ImportError:
        pass

    # Apple Silicon — local MLX / Metal (PyTorch MPS) / bare macOS
    if platform.system() == "Darwin" and _apple_silicon_brand():
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


def detect_os_version() -> str:
    # Try /etc/os-release first (Linux standard)
    try:
        with open("/etc/os-release") as f:
            info = {}
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    info[k] = v.strip('"')
            if "PRETTY_NAME" in info:
                return info["PRETTY_NAME"]
    except Exception:
        pass
    # macOS fallback
    mac_ver = platform.mac_ver()[0]
    if mac_ver:
        return f"macOS {mac_ver}"
    return platform.platform()


def detect_python_version() -> str:
    return platform.python_version()


def detect_intra_node_interconnect() -> str | None:
    """Detect intra-node GPU interconnect type from nvidia-smi topology output.

    Returns e.g. 'NVLink' if NVLink connections exist between GPUs, None otherwise.
    The topology is already collected by collect_topology(), but we re-query here
    to keep this function self-contained and callable independently.
    """
    try:
        import re
        out = subprocess.check_output(
            ["nvidia-smi", "topo", "-m", "--no-color"],
            text=True, stderr=subprocess.DEVNULL,
        )
        # NV# entries (e.g. NV12, NV18) in the topology matrix indicate NVLink
        if re.search(r'\bNV\d+\b', out):
            return "NVLink"
    except Exception:
        pass
    # Fallback: try without --no-color
    try:
        import re
        out = subprocess.check_output(
            ["nvidia-smi", "topo", "-m"],
            text=True, stderr=subprocess.DEVNULL,
        )
        if re.search(r'\bNV\d+\b', out):
            return "NVLink"
    except Exception:
        pass
    if _apple_silicon_brand():
        return "SoC unified memory (no GPU-GPU interconnect)"
    return None


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
        collect_tpu() or
        []
    )
    if not accelerators:
        _print_warning("No accelerators detected. Collecting CPU-only info.")

    env = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "accelerators": accelerators,
        "accelerator_topology": collect_topology(),
        "intra_node_interconnect": detect_intra_node_interconnect(),
        "cpu": collect_cpu(),
        "system_memory_gb": collect_memory_gb(),
        "pcie_generation": detect_pcie_gen(),
        "cpu_accelerator_bandwidth_gbs": None,
        "network_interfaces": collect_network_interfaces(),
        "os": detect_os_version(),
        "python_version": detect_python_version(),
        "kernel_version": platform.release(),
        "runtime_version": detect_runtime_version(),
        "pytorch_version": detect_pytorch_version(),
    }

    _warn_for_incomplete_env_report(env, accelerators)

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