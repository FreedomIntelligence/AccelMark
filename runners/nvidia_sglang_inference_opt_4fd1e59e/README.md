# SGLang Suite F inference fast path (v5)

Validated runner ID: `nvidia_sglang_inference_opt_4fd1e59e`

This runner subclasses `nvidia_sglang_inference_opt_c45ad27a` and keeps the
same synchronous batch inference fast path.  It ships a runner-local copy of
SGLang 0.5.6 under `.vendor/sglang` so framework-level patches stay inside the
runner directory and never modify global site-packages.  The runner bootstraps
the vendored package, forces `TMPDIR=/tmp` for reliable ZMQ IPC sockets, and
adds the runtime Python's `ninja` to `PATH` for JIT compilation.

v5 extends the v4 search space with aggressive, workload-aware optimizations:

- Framework-level source patches under `.vendor/sglang` (decode attention,
  scheduler, CUDA graph capture, memory allocation).
- Custom Triton kernels or autotune overrides for the 0.5B decode path.
- Static / minimal CUDA graph shapes for the fixed Suite F offline batch.
- Single-instance scheduler tuning; multi-instance data-parallel is left as a
  future candidate if single-instance saturation is reached.

The previous A100 leaderboard best for Suite F was 43,718 tok/s.  The v4
runner family reached ~45.9K tok/s; v5 aims to push further through source-level
optimization while staying within AccelMark integrity rules.

To install in an AccelMark checkout, copy this directory to
`runners/nvidia_sglang_inference_opt_4fd1e59e`, then copy the desired config to
`configs/runner_configs/runner_nvidia_sglang_inference_opt_4fd1e59e.yaml`.
The SGLang 0.5.6 runtime used for validation must be selected by the external
harness/runtime registry.
