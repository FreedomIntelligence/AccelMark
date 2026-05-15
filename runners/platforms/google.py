"""Google Cloud TPU platform plug-in."""
from __future__ import annotations

import os

ID = "google"
DISPLAY_NAME = "Google TPU"
VENDOR_LABEL = "Google"
PRIORITY = 50


def _chip_name_and_memory(tpu_type: str | None) -> tuple[str, float | None]:
    """Map a TPU GCE accelerator-type string to chip name and HBM per chip.

    Examples:
        v5litepod-*  → v5e, 16 GiB / chip
        v5e-*        → v5e (alias), 16 GiB / chip
        v6e-*        → v6e Trillium, 32 GiB / chip
        v7x-*        → v7x Ironwood, 192 GiB / chip
        v5p-*        → v5p, 95 GiB / chip
        v4-*         → v4, 32 GiB / chip
        v3-*         → v3, 16 GiB / chip
        v2-*         → v2, 8 GiB / chip
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
    return f"Google TPU ({tpu_type})", None


def collect() -> list[dict]:
    try:
        from tpu_inference import tpu_info

        num_chips = tpu_info.get_num_chips()
        tpu_type = tpu_info.get_tpu_type()
        node_name = tpu_info.get_node_name()

        if not num_chips or num_chips == 0:
            return []

        chip_name, memory_gb = _chip_name_and_memory(tpu_type)

        # tpu_info.get_num_cores_per_chip() misclassifies the GCE alias
        # form "v5e-1" (vs the canonical "v5litepod-1"), so disambiguate
        # via the raw tpu_type string ourselves.
        t = (tpu_type or "").lower()
        if "v5litepod" in t or "v5e" in t or "v6e" in t or "trillium" in t:
            num_cores_per_chip = 1
        else:
            num_cores_per_chip = tpu_info.get_num_cores_per_chip()

        jax_version = "unknown"
        try:
            import jax

            jax_version = jax.__version__
            jax_devices = jax.devices()
            if jax_devices and memory_gb is None:
                mem = getattr(jax_devices[0], "memory_stats", None)
                if mem and "bytes_limit" in mem:
                    memory_gb = round(mem["bytes_limit"] / (1024 ** 3), 1)
        except Exception:
            pass

        accelerators: list[dict] = []
        for i in range(num_chips):
            accelerators.append(
                {
                    "index": i,
                    "name": chip_name,
                    "vendor": VENDOR_LABEL,
                    "memory_gb": memory_gb,
                    "driver_version": f"JAX {jax_version}" if jax_version != "unknown" else "unknown",
                    "firmware_version": None,
                    "tpu_type": tpu_type,
                    "tpu_node_name": node_name,
                    "num_cores_per_chip": num_cores_per_chip,
                    "supports_bf16": True,
                }
            )
        return accelerators

    except Exception:
        return []


def detect_runtime_version() -> str | None:
    try:
        import jax

        return f"JAX {jax.__version__}"
    except ImportError:
        return None


def diagnostics(env: dict, accelerators: list[dict]) -> list[str]:
    notes: list[str] = []
    if not accelerators and (
        os.environ.get("TPU_NAME") or os.environ.get("CLOUD_TPU_TASK")
    ):
        notes.append(
            "TPU-related environment variables are set but no TPU devices were detected — "
            "install jax / tpu_inference when running on Cloud TPU."
        )
    if accelerators and (env.get("runtime_version") or "") == "unknown":
        notes.append(
            "Could not detect JAX/runtime for TPU. runtime_version is unknown — "
            "install jax if you use Cloud TPU."
        )
    return notes
