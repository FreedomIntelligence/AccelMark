"""Huawei Ascend NPU platform plug-in."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ID = "ascend"
DISPLAY_NAME = "Huawei Ascend"
VENDOR_LABEL = "Huawei"
PRIORITY = 30

_BF16_SUPPORTED = {"910b", "atlas 800t a2", "910b1", "910b2", "910b3", "910b4", "910_"}
_NO_BF16 = {"310", "310p", "atlas 300"}

# Subsystem Device ID → 910C variant discriminator (from npu-smi info -t board).
# Keys are hex strings as reported by "Subsystem Device ID" in -t board output.
# Values are human-readable variant suffixes for use in chip name construction.
#
# Known 910C sub-variants (specs are indicative — verify on your hardware):
#   910_9391/9392: 24 AI Cores, 1850 MHz, 64 GB HBM
#   910_9381/9382: 24 AI Cores, 1800 MHz, 64 GB HBM
#   910_9372:      20 AI Cores, 1800 MHz, 64 GB HBM
#   910_9362:      20 AI Cores, 1500 MHz, 32 GB HBM
#
# Subsystem Device IDs below are from verified hardware; update as new
# variants are identified.  If your 910C shows "Ascend910" but its SS-ID is
# not listed here, collect the board info and add an entry.
_910C_VARIANT_MAP: dict[str, str] = {
    "0x3001": "910_938x",   # Confirmed: 910_9381/9382 — 24 AI Cores, 1800 MHz, 64 GB
    # "0x3002": "910_939x",   # TODO: 910_9391/9392 — 24 AI Cores, 1850 MHz, 64 GB
    # "0x3003": "910_9372",   # TODO: 910_9372 — 20 AI Cores, 1800 MHz, 64 GB
    # "0x3004": "910_9362",   # TODO: 910_9362 — 20 AI Cores, 1500 MHz, 32 GB
}


def _resolve_910c_variant(subsystem_device_id: str | None) -> str | None:
    """Resolve a 910C subsystem device ID to a human-readable variant suffix.

    Returns e.g. ``"910_939x"`` for ``"0x3001"``, or ``None`` if the ID is
    unknown or not present in ``_910C_VARIANT_MAP``.
    """
    if not subsystem_device_id:
        return None
    return _910C_VARIANT_MAP.get(subsystem_device_id.strip())


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
    """Backfill memory_gb, name, and hardware identifiers via torch_npu runtime API.

    torch_npu.npu.get_device_properties(i) mirrors torch.cuda:
        .total_memory  — total HBM bytes
        .name          — chip name string (e.g. "910B2")

    Logical indices 0..N-1 map positionally to npu-smi enumeration order
    when all devices are visible. Only fills fields still None so parsed
    values are never overwritten.

    Additionally probes for vendor-specific attributes
    (``ai_core_count``, ``ai_core_frequency_mhz``, etc.) and uses
    the Subsystem Device ID to resolve 910C variant names when the
    chip name is generic (e.g. "Ascend910").
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

            raw_name = (props.name or "").strip()
            # Normalise: strip leading "Ascend" / "ascend" prefix so we
            # produce "Huawei Ascend 910_9382" instead of the redundant
            # "Huawei Ascend Ascend910_9382".
            if raw_name.lower().startswith("ascend"):
                raw_name = raw_name[6:].strip()

            # Only overwrite the name if it is still generic (table-parsed
            # "Ascend910" or list-format "Huawei Ascend NPU").
            cur_name = (rec.get("name") or "").lower()
            is_generic = (
                not rec.get("name")
                or rec.get("name") == "Huawei Ascend NPU"
                or "ascend910" in cur_name
            )
            if is_generic and raw_name:
                rec["name"] = f"Huawei Ascend {raw_name}"

            # ── Backfill vendor-specific hardware attributes ──
            # Attribute names from torch_npu (verified on CANN 25.5.x):
            #   cube_core_num    — AI Cube cores (primary compute units)
            #   vector_core_num  — AI Vector cores
            #   L2_cache_size    — L2 cache in bytes
            #   gcnArchName      — GCN architecture name (may be None)
            _VENDOR_ATTRS = (
                "cube_core_num",
                "vector_core_num",
                "L2_cache_size",
                "gcnArchName",
            )
            for attr in _VENDOR_ATTRS:
                if rec.get(attr) is None:
                    val = getattr(props, attr, None)
                    if val is not None:
                        rec[attr] = val

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
            bus_id = None
            if i + 1 < len(lines):
                row2 = lines[i + 1]
                hbm_match = re.search(r"(\d+)\s*/\s*(\d+)\s*\|?\s*$", row2)
                if hbm_match:
                    hbm_total_mb = int(hbm_match.group(2))
                # Row 2 format: | <ChipID> | <Bus-Id> | ...
                # Bus-Id is the second pipe-delimited field (PCIe address).
                bus_match = re.match(r"\|\s*\d+\s*\|\s*(\S+)\s*\|", row2)
                if bus_match:
                    bus_id = bus_match.group(1).strip()
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
                    "bus_id": bus_id,
                }
            )
        i += 1

    return accelerators


def _get_board_info(npu_id: str) -> dict:
    """Query hardware identity for a single NPU via ``-t board``.

    Returns dict with keys ``driver_version``, ``firmware_version``,
    ``product_name``, ``pci_vendor_id``, ``pci_device_id``,
    ``subsystem_vendor_id``, ``subsystem_device_id``, ``chip_count``,
    and ``board_id``.  Falls back to CANN install-path files for
    ``driver_version`` if the command fails or produces no match.
    All fields other than ``driver_version`` default to ``None``.
    """
    result = {
        "driver_version": "unknown",
        "firmware_version": None,
        "product_name": None,
        "pci_vendor_id": None,
        "pci_device_id": None,
        "subsystem_vendor_id": None,
        "subsystem_device_id": None,
        "chip_count": None,
        "board_id": None,
    }

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
            # ── new hardware-identity fields ──
            m = re.search(r"Product\s+Name\s*:\s*(.+)", line, re.IGNORECASE)
            if m:
                result["product_name"] = m.group(1).strip()
            m = re.search(r"PCI\s+Vendor\s+ID\s*:\s*(.+)", line, re.IGNORECASE)
            if m:
                result["pci_vendor_id"] = m.group(1).strip()
            m = re.search(r"PCI\s+Device\s+ID\s*:\s*(.+)", line, re.IGNORECASE)
            if m:
                result["pci_device_id"] = m.group(1).strip()
            m = re.search(r"Subsystem\s+Vendor\s+ID\s*:\s*(.+)", line, re.IGNORECASE)
            if m:
                result["subsystem_vendor_id"] = m.group(1).strip()
            m = re.search(r"Subsystem\s+Device\s+ID\s*:\s*(.+)", line, re.IGNORECASE)
            if m:
                result["subsystem_device_id"] = m.group(1).strip()
            m = re.search(r"Chip\s+Count\s*:\s*(\d+)", line, re.IGNORECASE)
            if m:
                result["chip_count"] = int(m.group(1))
            m = re.search(r"Board\s+ID\s*:\s*(.+)", line, re.IGNORECASE)
            if m:
                result["board_id"] = m.group(1).strip()
    except Exception:
        pass

    if result["driver_version"] == "unknown":
        # Only check version.cfg as a last-resort fallback for the *driver*
        # version field.  Prefer the Software Version line from -t board
        # (parsed above); version.cfg contains the CANN *toolkit* version,
        # not the driver version, but it is better than "unknown".
        for cann_path in (
            "/usr/local/Ascend/ascend-toolkit/latest",
            "/usr/local/Ascend/nnae/latest",
        ):
            version_file = Path(cann_path) / "version.cfg"
            if version_file.exists():
                try:
                    text = version_file.read_text()
                    m = re.search(r"Version\s*=\s*(.+)", text, re.IGNORECASE)
                    if m:
                        result["driver_version"] = f"driver unknown (CANN toolkit {m.group(1).strip()})"
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
                    "bus_id": None,
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
            # ── Board-level hardware identity fields ──
            rec["product_name"] = board.get("product_name")
            rec["pci_vendor_id"] = board.get("pci_vendor_id")
            rec["pci_device_id"] = board.get("pci_device_id")
            rec["subsystem_vendor_id"] = board.get("subsystem_vendor_id")
            rec["subsystem_device_id"] = board.get("subsystem_device_id")
            rec["board_id"] = board.get("board_id")
            if board.get("chip_count") is not None:
                rec["chip_count"] = board["chip_count"]
        _enrich_via_torch_npu(accelerators)

        # ── Final pass: resolve names that are still generic ──
        # torch_npu (if installed) already handled this; this is the
        # fallback for systems without torch_npu.
        for rec in accelerators:
            name = (rec.get("name") or "").lower()
            if not name or "ascend910" in name or "huawei ascend npu" in name:
                # 1) Check static lookup table (cosmetic — maps known SS-IDs to
                #    human-friendly variant names like "910_938x").
                ss_id = rec.get("subsystem_device_id")
                variant = _resolve_910c_variant(ss_id)
                if variant:
                    rec["name"] = f"Huawei Ascend {variant}"
                    continue
                # 2) No mapping found — construct a unique identifier dynamically
                #    from the board-info fields that npu-smi -t board provides.
                #    This handles unknown 910C variants without code changes.
                product = rec.get("product_name")
                if product:
                    rec["name"] = f"Huawei Ascend Ascend910 ({product})"
                elif ss_id:
                    rec["name"] = f"Huawei Ascend Ascend910 (SS:{ss_id})"
                # else: leave the generic name as-is (last resort)

    return accelerators


def detect_runtime_version() -> str | None:
    """Return the CANN *toolkit* version (e.g. ``"CANN 8.5.1"``).

    This is the SDK / compiler / runtime stack — analogous to CUDA toolkit
    version on NVIDIA.  The NPU *driver* version (e.g. 25.2.3) is stored
    per-accelerator in ``driver_version`` and is a separate thing.

    Detection order:
      1. ``/usr/local/Ascend/ascend-toolkit/latest/version.cfg`` (primary)
      2. ``/usr/local/Ascend/nnae/latest/version.cfg`` (legacy installs)
      3. ``ASCEND_TOOLKIT_HOME`` / ``ASCEND_HOME`` env vars → version.cfg
      4. Fallback: ``npu-smi info -t board`` Software Version, labelled as
         driver (not toolkit) via a ``_note`` suffix.
    """
    import os as _os

    # Canonical CANN toolkit install paths (in priority order).
    _CANN_ROOTS = [
        "/usr/local/Ascend/ascend-toolkit/latest",
        "/usr/local/Ascend/nnae/latest",
    ]
    # Environment variables that may point to a CANN toolkit root.
    _CANN_ENV_VARS = ["ASCEND_TOOLKIT_HOME", "ASCEND_HOME"]

    for var in _CANN_ENV_VARS:
        val = _os.environ.get(var, "").strip()
        if val:
            _CANN_ROOTS.append(val)

    for root in _CANN_ROOTS:
        version_file = Path(root) / "version.cfg"
        if version_file.exists():
            try:
                text = version_file.read_text()
                m = re.search(r"Version\s*=\s*(.+)", text, re.IGNORECASE)
                if m:
                    return f"CANN {m.group(1).strip()}"
            except Exception:
                continue

    # Symlink-target fallback: ascend-toolkit/latest → 8.5.1
    for root in _CANN_ROOTS[:2]:  # only the well-known paths
        try:
            resolved = Path(root).resolve()
            version_dir = resolved.name  # e.g. "8.5.1"
            if re.match(r"\d+\.\d+", version_dir):
                return f"CANN {version_dir}"
        except Exception:
            continue

    # Last resort: try npu-smi for the driver version (NOT the toolkit).
    # Clearly label this so it isn't mistaken for the CANN toolkit version.
    try:
        info_out = subprocess.check_output(
            ["npu-smi", "info"], text=True, stderr=subprocess.DEVNULL
        )
        m = re.search(r"\|\s*(\d+)\s+\S+\s*\|", info_out)
        if m:
            board = _get_board_info(m.group(1))
            if board["driver_version"] != "unknown":
                return f"CANN driver {board['driver_version']} (toolkit version unknown)"
    except Exception:
        pass

    return None


def sample_power_watts() -> float | None:
    """Return instantaneous total board power (watts) summed across all
    visible Ascend NPUs, or None if unavailable. Respects ASCEND_RT_VISIBLE_DEVICES."""
    import os

    try:
        out = subprocess.check_output(
            ["npu-smi", "info"], text=True, stderr=subprocess.DEVNULL, timeout=5
        )
    except Exception:
        return None

    # Respect ASCEND_RT_VISIBLE_DEVICES
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    visible_indices: set[int] | None = None
    if visible:
        try:
            visible_indices = {int(x.strip()) for x in visible.split(",") if x.strip()}
        except ValueError:
            pass

    total = 0.0
    found = 0
    # Parse npu-smi info tabular output: each device row has power as the
    # 4th pipe-delimited field after NPU ID and ChipName and Health.
    # Example: | 7     910B2               | OK            | 96.5  49  ... |
    lines = out.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        row_match = re.match(r"\|\s*(\d+)\s+\S+\s*\|", line)
        if row_match:
            npu_id = int(row_match.group(1))
            if visible_indices is not None and npu_id not in visible_indices:
                i += 1
                continue
            # Power field is after Health field in the same row
            # Format: | NPU ChipName | Health | Power(W) Temp(C) ... |
            power_match = re.search(r"\|\s*\S+\s*\|\s*(\d+\.?\d*)\s", line)
            if power_match:
                try:
                    total += float(power_match.group(1))
                    found += 1
                except ValueError:
                    pass
        i += 1

    return round(total, 1) if found > 0 else None


def diagnostics(env: dict, accelerators: list[dict]) -> list[str]:
    notes: list[str] = []
    if (env.get("pytorch_version") or "") == "unknown":
        notes.append(
            "PyTorch is not installed — pytorch_version is unknown. For GPU stack "
            "metadata: pip install torch (with torch_npu)."
        )
    if (env.get("runtime_version") or "") == "unknown":
        notes.append(
            "Could not detect CANN toolkit version from "
            "/usr/local/Ascend/ascend-toolkit/latest/version.cfg or "
            "/usr/local/Ascend/nnae/latest/version.cfg. "
            "runtime_version is unknown — this may affect result reproducibility."
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
    if accelerators and sample_power_watts() is None:
        notes.append(
            "Power sampling returned None — npu-smi power field is unavailable. "
            "tokens_per_sec_per_watt will not be computed."
        )
    return notes
