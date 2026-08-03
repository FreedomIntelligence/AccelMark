"""
Chip peak specifications for roofline analysis.

Each entry maps a chip name to its peak FP16 / BF16 tensor-core TFLOPS
and peak HBM bandwidth in GB / s.  Sources are vendor datasheets.

**Verification status**: Chips marked in ``_VERIFIED_CHIPS`` have been
cross-checked against the paper's benchmark results (18 accelerators).
Entries for unverified chips are datasheet values only and should be
treated as provisional.  Verified values may still differ from
datasheet-sourced values — datasheet numbers often report peak with
sparsity (2×) while actual achievable throughput is lower.

Additions and corrections for new chips are welcome — open a PR.
"""

from __future__ import annotations

from typing import Optional

# Chips that have been benchmarked in the paper — these specs are validated.
# Others are datasheet-only and should be treated as provisional.
_VERIFIED_CHIPS: set[str] = {
    # NVIDIA — 10 chips benchmarked in the paper
    "NVIDIA A800-SXM4-80GB", "NVIDIA A100-SXM4-80GB", "NVIDIA A100-SXM4-40GB",
    "NVIDIA H100-SXM-80GB", "NVIDIA H200-SXM-141GB", "NVIDIA H20-3e",
    "NVIDIA RTX 5090", "NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 4090 D",
    "NVIDIA GeForce RTX 3090",
    # Huawei Ascend — 3 chips
    "Huawei Ascend 910B2", "Huawei Ascend 910B", "Huawei Ascend 910",
    # Google TPU — 3 chips
    "Google TPU v6e", "Google TPU v5e", "Google TPU v5p",
    # AMD — 1 chip
    "AMD Instinct MI300X",
    # Moore Threads — 1 chip
    "Moore Threads S4000",
}

# ── Peak spec database ────────────────────────────────────────────────────────
# fmt: off
_CHIP_PEAK_SPECS: dict[str, dict[str, float]] = {
    # ── NVIDIA ────────────────────────────────────────────────────────────
    "NVIDIA A800-SXM4-80GB":      {"tflops": 312.0, "bw_gbps": 2039.0},
    "NVIDIA A100-SXM4-80GB":      {"tflops": 312.0, "bw_gbps": 2039.0},
    "NVIDIA A100-SXM4-40GB":      {"tflops": 312.0, "bw_gbps": 1555.0},
    "NVIDIA A100-PCIe-80GB":      {"tflops": 312.0, "bw_gbps": 1935.0},
    "NVIDIA H100-SXM-80GB":       {"tflops": 989.0, "bw_gbps": 3350.0},
    "NVIDIA H100-PCIe-80GB":      {"tflops": 756.0, "bw_gbps": 2039.0},
    "NVIDIA H200-SXM-141GB":      {"tflops": 989.0, "bw_gbps": 4800.0},
    "NVIDIA H20-3e":              {"tflops": 148.0, "bw_gbps": 4000.0},
    "NVIDIA RTX 5090":            {"tflops": 838.0, "bw_gbps": 1792.0},
    "NVIDIA GeForce RTX 5090":    {"tflops": 838.0, "bw_gbps": 1792.0},
    "NVIDIA L40S":                {"tflops": 362.0, "bw_gbps": 864.0},
    "NVIDIA RTX 6000 Ada Generation": {"tflops": 362.0, "bw_gbps": 960.0},
    "NVIDIA RTX A6000":           {"tflops": 309.7, "bw_gbps": 768.0},
    "NVIDIA GeForce RTX 4090":    {"tflops": 330.3, "bw_gbps": 1008.0},
    "NVIDIA GeForce RTX 4090 D":  {"tflops": 330.3, "bw_gbps": 1008.0},
    "NVIDIA GeForce RTX 3090":    {"tflops": 142.0, "bw_gbps": 936.0},
    "NVIDIA V100-SXM2-32GB":      {"tflops": 125.0, "bw_gbps": 900.0},
    "NVIDIA Tesla V100S-PCIE-32GB": {"tflops": 122.6, "bw_gbps": 1134.0},
    "NVIDIA Tesla V100-PCIE-32GB": {"tflops": 112.0, "bw_gbps": 900.0},
    "NVIDIA T4":                  {"tflops": 65.0,  "bw_gbps": 320.0},
    "NVIDIA L4":                  {"tflops": 121.0, "bw_gbps": 300.0},

    # ── Huawei Ascend ─────────────────────────────────────────────────────
    "Huawei Ascend 910B2":        {"tflops": 320.0, "bw_gbps": 1200.0},
    "Huawei Ascend 910B":         {"tflops": 256.0, "bw_gbps": 1200.0},
    "Huawei Ascend 910":          {"tflops": 256.0, "bw_gbps": 1200.0},

    # ── Google TPU ────────────────────────────────────────────────────────
    # TPU v6e: bf16 peak per chip.  Source: Google Cloud TPU documentation.
    "Google TPU v6e":             {"tflops": 467.0, "bw_gbps": 1640.0},
    "Google TPU v5e":             {"tflops": 197.0, "bw_gbps": 820.0},
    "Google TPU v5p":             {"tflops": 459.0, "bw_gbps": 2765.0},

    # ── AMD ───────────────────────────────────────────────────────────────
    "AMD Instinct MI300X":        {"tflops": 1307.0, "bw_gbps": 5300.0},
    "AMD Instinct MI250X":        {"tflops": 383.0, "bw_gbps": 3350.0},
    "AMD Instinct MI210":         {"tflops": 181.0, "bw_gbps": 1638.0},

    # ── Apple Silicon ─────────────────────────────────────────────────────
    "Apple M1":                   {"tflops": 2.6,   "bw_gbps": 68.0},
    "Apple M1 Max":               {"tflops": 5.2,   "bw_gbps": 400.0},
    "Apple M1 Ultra":             {"tflops": 10.4,  "bw_gbps": 800.0},
    "Apple M2":                   {"tflops": 3.6,   "bw_gbps": 100.0},
    "Apple M2 Max":               {"tflops": 7.2,   "bw_gbps": 400.0},
    "Apple M2 Ultra":             {"tflops": 14.4,  "bw_gbps": 800.0},
    "Apple M3 Max":               {"tflops": 11.0,  "bw_gbps": 400.0},
    "Apple M3 Ultra":             {"tflops": 22.0,  "bw_gbps": 800.0},
    "Apple M4 Pro":               {"tflops": 9.0,   "bw_gbps": 273.0},
    "Apple M4 Max":               {"tflops": 13.6,  "bw_gbps": 546.0},

    # ── Moore Threads ─────────────────────────────────────────────────────
    "Moore Threads S4000":        {"tflops": 200.0, "bw_gbps": 896.0},
    "Moore Threads S5000":        {"tflops": 256.0, "bw_gbps": 896.0},
}


def lookup(chip_name: str) -> tuple[Optional[float], Optional[float]]:
    """Return ``(peak_tflops, peak_bw_gbps)`` for *chip_name*.

    Performs exact match first, then falls back to case-insensitive
    substring matching.  Returns ``(None, None)`` when no match is
    found.
    """
    if not chip_name:
        return None, None

    # Exact match
    spec = _CHIP_PEAK_SPECS.get(chip_name)
    if spec:
        return spec["tflops"], spec["bw_gbps"]

    # Fuzzy match: case-insensitive substring in either direction
    name_lower = chip_name.lower()
    for key, s in _CHIP_PEAK_SPECS.items():
        key_lower = key.lower()
        if key_lower in name_lower or name_lower in key_lower:
            return s["tflops"], s["bw_gbps"]

    return None, None


def list_known_chips() -> list[str]:
    """Return sorted list of all known chip names."""
    return sorted(_CHIP_PEAK_SPECS.keys())


def register_chip(name: str, tflops: float, bw_gbps: float) -> None:
    """Register a new chip peak spec at runtime.

    Useful for CI or container environments where the detected chip
    name doesn't match the built-in database.
    """
    _CHIP_PEAK_SPECS[name] = {"tflops": tflops, "bw_gbps": bw_gbps}


def i_star_table() -> list[dict]:
    """Return I* (ridge point) for every chip in the spec database.

    I* = peak_TFLOPS / peak_BW_Gbps, with TFLOPS converted to FLOP/s.
    This is the roofline ridge point — workloads with AI > I* are
    compute-bound; AI ≤ I* are bandwidth-bound.

    Returns a list of dicts sorted by I* (ascending), suitable for
    direct inclusion in the paper's Table 1.
    """
    from profiling.roofline import ridge_point as _rp

    rows = []
    for name, spec in _CHIP_PEAK_SPECS.items():
        tflops = spec["tflops"]
        bw = spec["bw_gbps"]
        i_star = _rp(tflops, bw)
        rows.append({
            "chip": name,
            "tflops": tflops,
            "bw_gbps": bw,
            "i_star": i_star,
            "verified": name in _VERIFIED_CHIPS,
        })
    rows.sort(key=lambda r: r["i_star"])
    return rows


def print_i_star_table() -> None:
    """Pretty-print the I* table for all chips."""
    rows = i_star_table()
    print(f"{'Chip':45s} {'TFLOPS':>8s} {'BW GB/s':>8s} {'I*':>8s}  Status")
    print("-" * 80)
    for r in rows:
        status = "✓ verified" if r["verified"] else "  datasheet"
        print(f"  {r['chip']:42s} {r['tflops']:8.0f} {r['bw_gbps']:8.0f} "
              f"{r['i_star']:8.1f}  {status}")
    print()
    i_vals = [r["i_star"] for r in rows]
    n_verified = sum(1 for r in rows if r["verified"])
    print(f"  I* range: {min(i_vals):.0f} – {max(i_vals):.0f} "
          f"(span {max(i_vals)/min(i_vals):.1f}×)  |  "
          f"{n_verified}/{len(rows)} chips verified")
