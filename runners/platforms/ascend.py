"""Huawei Ascend NPU platform plug-in."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ID = "ascend"
DISPLAY_NAME = "Huawei Ascend"
VENDOR_LABEL = "Huawei"
PRIORITY = 30

_BF16_SUPPORTED = {"910b", "atlas 800t a2", "910b1", "910b2", "910b3", "910b4"}
_NO_BF16 = {"310", "310p", "atlas 300"}


def _supports_bf16(chip_name: str) -> bool:
    if not chip_name:
        return True
    name_lower = chip_name.lower()
    if any(k in name_lower for k in _BF16_SUPPORTED):
        return True
    if any(k in name_lower for k in _NO_BF16):
        return False
    return True


def _enrich_via_torch_npu(accelerators: list[dict]) -> None:
    """Backfill memory_gb and name via torch_npu runtime API.

    torch_npu.npu.get_device_properties(i) mirrors torch.cuda:
        .total_memory  — total HBM bytes
        .name          — chip name string (e.g. "910B2")
    Logical indices 0..N-1 map positionally to npu-smi enumeration order
    when all devices are visible. Only fills fields still None so parsed
    values are never overwritten.
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
    """Parse the tabular output of plain ``npu-smi info``.

    The table format has two data rows per device:
        Row 1: | <NPU_ID>  <ChipName>  | <Health> | <Power> <Temp> <Hugepages> |
        Row 2: | <ChipID>              | <Bus-Id> | <AICore> <Mem-Usage>  <HBM-Usage(used/total MB)> |

    Example:
        | 7     910B2               | OK            | 96.5        49                0    / 0             |
        | 0                         | 0000:42:00.0  | 0           0    / 0          3389 / 65536         |
    """
    accelerators: list[dict] = []
    lines = out.splitlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        row1 = re.match(r"\|\s*(\d+)\s+(\S+)\s*\|", line)
        if row1:
            npu_id = int(row1.group(1))
            chip_name = row1.group(2).strip()
            hbm_total_mb = None
            if i + 1 < len(lines):
                row2 = lines[i + 1]
                hbm_match = re.search(r"(\d+)\s*/\s*(\d+)\s*\|?\s*$", row2)
                if hbm_match:
                    hbm_total_mb = int(hbm_match.group(2))
                i += 1

            memory_gb = round(hbm_total_mb / 1024, 1) if hbm_total_mb else None
            name = f"Huawei Ascend {chip_name}" if chip_name else "Huawei Ascend NPU"
            accelerators.append(
                {
                    "index": npu_id,
                    "name": name,
                    "vendor": VENDOR_LABEL,
                    "memory_gb": memory_gb,
                    "driver_version": cann_version,
                    "firmware_version": None,
                    "supports_bf16": _supports_bf16(name),
                }
            )
        i += 1

    return accelerators


def _get_board_info(npu_id: str) -> dict:
    """Query driver and firmware version for a single NPU via ``-t board``.

    Returns dict with keys ``driver_version`` and ``firmware_version``.
    Falls back to CANN install-path files for driver_version if the command
    fails or produces no match.
    """
    result = {"driver_version": "unknown", "firmware_version": None}

    try:
        out = subprocess.check_output(
            ["npu-smi", "info", "-t", "board", "-i", npu_id],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            sw_match = re.search(r"Software\s+Version\s*:\s*(.+)", line, re.IGNORECASE)
            if sw_match:
                result["driver_version"] = sw_match.group(1).strip()
            fw_match = re.search(r"Firmware\s+Version\s*:\s*(.+)", line, re.IGNORECASE)
            if fw_match:
                fw = fw_match.group(1).strip()
                result["firmware_version"] = None if fw.upper() == "NA" else fw
    except Exception:
        pass

    if result["driver_version"] == "unknown":
        for cann_path in (
            "/usr/local/Ascend/ascend-toolkit/latest",
            "/usr/local/Ascend/nnae/latest",
        ):
            version_file = Path(cann_path) / "version.cfg"
            if version_file.exists():
                try:
                    text = version_file.read_text()
                    m = re.search(r"Version=(.+)", text)
                    if m:
                        result["driver_version"] = f"CANN {m.group(1).strip()}"
                        break
                except Exception:
                    pass

    return result


def collect() -> list[dict]:
    try:
        out = subprocess.check_output(
            ["npu-smi", "info"], text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return []

    accelerators = _parse_npu_smi_table(out, "unknown")

    if not accelerators:
        try:
            out_l = subprocess.check_output(
                ["npu-smi", "info", "-l"], text=True, stderr=subprocess.DEVNULL
            )
        except Exception:
            return []
        current_npu: dict | None = None
        for line in out_l.splitlines():
            npu_match = re.search(r"NPU\s+ID\s*:\s*(\d+)", line, re.IGNORECASE)
            if npu_match:
                if current_npu:
                    current_npu["supports_bf16"] = _supports_bf16(current_npu.get("name", ""))
                    accelerators.append(current_npu)
                current_npu = {
                    "index": int(npu_match.group(1)),
                    "name": "Huawei Ascend NPU",
                    "vendor": VENDOR_LABEL,
                    "memory_gb": None,
                    "driver_version": "unknown",
                    "firmware_version": None,
                }
            if current_npu is None:
                continue
            chip_match = re.search(r"Chip\s+Name\s*:\s*(.+)", line, re.IGNORECASE)
            if chip_match:
                current_npu["name"] = f"Huawei Ascend {chip_match.group(1).strip()}"
            mem_match = re.search(r"HBM\s+Capacity.*?:\s*(\d+)", line, re.IGNORECASE)
            if mem_match:
                current_npu["memory_gb"] = round(int(mem_match.group(1)) / 1024, 1)
            if current_npu.get("memory_gb") is None:
                mem_match2 = re.search(
                    r"Memory\s+Capacity.*?:\s*(\d+)\s*MB", line, re.IGNORECASE
                )
                if mem_match2:
                    current_npu["memory_gb"] = round(int(mem_match2.group(1)) / 1024, 1)
            if current_npu.get("firmware_version") is None:
                fw_match = re.search(r"Firmware\s+Version\s*:\s*(.+)", line, re.IGNORECASE)
                if fw_match:
                    current_npu["firmware_version"] = fw_match.group(1).strip()
        if current_npu:
            current_npu["supports_bf16"] = _supports_bf16(current_npu.get("name", ""))
            accelerators.append(current_npu)

    if accelerators:
        for rec in accelerators:
            board = _get_board_info(str(rec["index"]))
            rec["driver_version"] = board["driver_version"]
            rec["firmware_version"] = board["firmware_version"]
        _enrich_via_torch_npu(accelerators)

    return accelerators


def detect_runtime_version() -> str | None:
    try:
        info_out = subprocess.check_output(
            ["npu-smi", "info"], text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return None
    m = re.search(r"\|\s*(\d+)\s+\S+\s*\|", info_out)
    if not m:
        return None
    board = _get_board_info(m.group(1))
    if board["driver_version"] != "unknown":
        return f"CANN {board['driver_version']}"
    return None


def diagnostics(env: dict, accelerators: list[dict]) -> list[str]:
    notes: list[str] = []
    if (env.get("pytorch_version") or "") == "unknown":
        notes.append(
            "PyTorch is not installed — pytorch_version is unknown. For GPU stack "
            "metadata: pip install torch (with torch_npu)."
        )
    if (env.get("runtime_version") or "") == "unknown":
        notes.append(
            "Could not detect CANN/runtime from npu-smi / install paths. "
            "runtime_version is unknown."
        )
    for a in accelerators:
        if a.get("memory_gb") is None:
            try:
                import torch_npu  # noqa: F401
            except ImportError:
                notes.append(
                    "Ascend HBM memory could not be parsed from npu-smi — optional "
                    "pip install torch_npu may fill memory_gb via the runtime API."
                )
            else:
                notes.append(
                    "Ascend HBM memory_gb is still unknown (torch_npu is importable) — "
                    "check ASCEND_VISIBLE_DEVICES, driver, and npu-smi output."
                )
            break
    return notes
