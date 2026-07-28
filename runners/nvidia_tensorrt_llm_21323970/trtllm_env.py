"""
Pre-import MPI workaround for TensorRT-LLM.

TensorRT-LLM links against mpi4py which calls MPI_Init at module-import
time.  On containerised systems the MPI installation may have been built
at a different path than where it is deployed (common with HPC-X and
NVIDIA NGC containers).  The symptom is::

    Sorry! You were supposed to get help about:
        opal_init:startup:internal-failure
    But I couldn't open the help file:
        /build-result/hpcx-v2.19-.../ompi/share/openmpi/help-opal-runtime.txt

The fix is to set ``OPAL_PREFIX`` to point at the *runtime* MPI prefix
so OpenMPI finds its help files and shared libraries at initialisation.

Import this module **before** ``import tensorrt_llm``::

    import trtllm_env  # noqa: F401 — side-effects only
    from tensorrt_llm import LLM

Ref: https://github.com/open-mpi/ompi/issues/8058#issuecomment-1514315391
"""

import os
import sys

# ── Detect the runtime MPI prefix ─────────────────────────────────────────
# Try the standard HPC-X deployment location first.

_RUNTIME_MPI_PREFIX = None
_CANDIDATES = [
    sys.prefix,                # conda/virtualenv prefix (covers most installs)
    "/opt/hpcx/ompi",          # HPC-X runtime (NGC containers)
    "/usr/local/mpi",          # generic MPI install
]
for _cand in _CANDIDATES:
    _help_file = os.path.join(_cand, "share", "openmpi", "help-opal-runtime.txt")
    if os.path.isfile(_help_file):
        _RUNTIME_MPI_PREFIX = _cand
        break

if _RUNTIME_MPI_PREFIX is None:
    # Best-effort — try the current Python prefix (conda/virtualenv).
    _RUNTIME_MPI_PREFIX = sys.prefix


# ── OPAL_PREFIX ───────────────────────────────────────────────────────────
# Override the hard-coded build-time prefix so OpenMPI finds its help files
# and configuration data at runtime.
os.environ["OPAL_PREFIX"] = _RUNTIME_MPI_PREFIX

# ── MPI configuration for single-node operation ──────────────────────────
# On systems without InfiniBand or with incomplete MPI installations,
# force local-only communication (shared memory) and disable CUDA-aware MPI
# to avoid MPI_Init failures.
os.environ.setdefault("OMPI_MCA_btl", "self,sm")
os.environ.setdefault("OMPI_MCA_osc", "sm")
os.environ.setdefault("OMPI_MCA_mpi_cuda_support", "0")
os.environ.setdefault("OMPI_MCA_opal_cuda_support", "0")
# Allow MPI to run as root (required in container/root environments)
os.environ.setdefault("OMPI_MCA_routed", "direct")
os.environ.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
os.environ.setdefault("PRTE_ALLOW_RUN_AS_ROOT", "1")

# ── Pre-initialise MPI ───────────────────────────────────────────────────
# TRT-LLM imports mpi4py.futures which calls MPI_Comm_spawn internally.
# On systems with incomplete MPI this fails.  Importing mpi4py.MPI first
# initialises MPI in simple single-process mode, which is all we need.
try:
    import mpi4py.MPI  # noqa: F401 — side-effect: MPI_Init
except Exception:
    pass

# ── MPI session already patched on disk ─────────────────────────────────
# mpi_session.py has been patched to skip MPI spawn for n_workers <= 1.
# See the patch in the installed tensorrt_llm package.

# ── CUDA 13 shared libraries ──────────────────────────────────────────────
# TRT-LLM 1.2.1 is compiled against CUDA 13, but the pip-installed cuBLAS
# and other CUDA libs live under site-packages/nvidia/cu13/lib.  Setting
# LD_LIBRARY_PATH in-process is unreliable (the dynamic linker may have
# cached it at startup).  Instead, pre-load the key libraries with ctypes
# so they are already in the process image when tensorrt_llm.bindings
# (a C extension) tries to resolve its CUDA dependencies.

def _preload_nvidia_libs():
    """Pre-load NVIDIA CUDA libraries into the process via ctypes.

    TRT-LLM's C extension (``tensorrt_llm.bindings``) links against CUDA
    shared libraries.  On systems where ``LD_LIBRARY_PATH`` is not set at
    process startup (conda envs, containers), the dynamic linker won't find
    the pip-installed CUDA libs under site-packages.  Pre-loading them with
    ctypes before the C extension import avoids ``ImportError: libcublas.so``.

    The library version (12 or 13) is auto-detected so the same code works
    with both CUDA 12 and CUDA 13 TRT-LLM builds.
    """
    import ctypes
    import site

    _lib_dir = None
    # Search order: cu13 (newer), cu12, cublas (flat cu12)
    for _sub in ("nvidia/cu13/lib", "nvidia/cu12/lib", "nvidia/cublas/lib"):
        for _sp in site.getsitepackages():
            _cand = os.path.join(_sp, _sub)
            if os.path.isdir(_cand) and os.listdir(_cand):
                _lib_dir = _cand
                break
        if _lib_dir is not None:
            break

    if _lib_dir is None:
        return

    _loaded = 0
    for _soname in sorted(os.listdir(_lib_dir)):
        if not _soname.endswith((".so", ".so.13", ".so.13.3", ".so.12")):
            continue
        _full = os.path.join(_lib_dir, _soname)
        try:
            ctypes.CDLL(_full, mode=ctypes.RTLD_GLOBAL)
            _loaded += 1
        except OSError:
            pass

_preload_nvidia_libs()

# ── Misc OMPI / UCX tuning ────────────────────────────────────────────────
os.environ.setdefault("OMPI_MCA_orte_base_help_aggregate", "0")
os.environ.setdefault("OMPI_MCA_mpi_abort_delay", "0")
# Disable UCX (InfiniBand) — the mlx5_0 errors are from UCX trying to use
# an inactive IB fabric.  Force OMPI to fall back to the TCP BTL instead.
os.environ.setdefault("OMPI_MCA_pml", "ob1")           # use ob1 PML (not UCX)
os.environ.setdefault("OMPI_MCA_osc", "pt2pt")         # disable RDMA one-sided
os.environ["OMPI_MCA_btl"] = "^openib,sm,uct"          # TCP only
os.environ["UCX_TLS"] = "^ib,rc,dc,ud"                 # disable all IB transports in UCX
os.environ["UCX_NET_DEVICES"] = "lo"                   # loopback only

# Set CUDA arch for JIT compilation so FlashInfer / TRT-LLM kernels target
# the correct SM version.  A800 = SM80, A100 = SM80, H100 = SM90.
# Without this, JIT-compiled kernels may produce CUDA_ERROR_INVALID_IMAGE.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;9.0")

# TRT-LLM 0.21 has multiple attention backends.  The default "TRTLLM"
# backend uses JIT-compiled CUDA kernels which can fail with
# CUDA_ERROR_INVALID_IMAGE on older driver / toolkit combos.  FlashInfer
# is the recommended fallback for Ampere (SM80) GPUs.
os.environ.setdefault("TRTLLM_ATTN_BACKEND", "FLASHINFER")
