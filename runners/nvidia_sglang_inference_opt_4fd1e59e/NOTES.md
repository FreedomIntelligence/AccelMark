# Suite F v5 Bottleneck Diagnosis and Optimization Notes

## Workload

- Model: Qwen/Qwen2.5-0.5B-Instruct (BF16)
- Hardware: single NVIDIA A100-SXM4-80GB
- Scenario: AccelMark Suite F offline, client concurrency 4/16/64, 200 requests
- Primary metric: output-only throughput at concurrency 64 (tok/s)
- Model dims: hidden_size=896, intermediate_size=4864, num_attention_heads=14,
  num_key_value_heads=2, num_hidden_layers=24, head_dim=64.

## Bottleneck diagnosis

1. **Small model on a large accelerator**: the 0.5B model is compute-light on
   A100. A single decode step is too small to saturate the GPU, so the main
   gains come from reducing scheduler/Python overhead and keeping the GPU
   busy with larger effective batch sizes.

2. **SGLang's default FlashInfer path is not fastest**: for this tiny model,
   the Triton attention backend with CUDA-graph capture is faster because it
   avoids backend dispatch overhead.

3. **CUDA-graph padding hurts**: padding batches up to the maximum captured
   shape adds wasted memory-bandwidth work. Disabling padding (`disable_cuda_graph_padding`)
   was the single largest win in the v3/v4 sweep.

4. **Scheduler/Python overhead is now the dominant residual cost**: with
   `continuous_decode_steps=4` and CUDA graph capture, raw kernel time is
   small; the remaining gap to hardware peak is mostly Python scheduler work,
   batch metadata construction, and small-launch overheads in decode attention.

5. **Decode attention metadata dominates per-step CPU work**: the Triton decode
   backend rebuilds `kv_indptr`, `kv_indices`, `num_kv_splits`, and temporary
   attention buffers on every step. For very short sequences (Suite F output
   tokens are short), this metadata work is large relative to the GEMM-like
   attention kernel itself.

6. **Framework-level patches must stay legal**: all patches live under the
   runner's `.vendor/sglang` directory and are loaded via `sys.path` insertion
   in `runner.py`. No global site-packages are modified.

## v5 optimization directions

- **Decode attention kernel tuning**: reduce `BLOCK_N`, `num_warps`, and
  `num_stages` for tiny batches; avoid over-parallelization on short KV
  sequences; make `num_kv_splits` smaller for the short-context Suite F.

- **KV-cache memory layout**: keep the default MHA layout but disable the
  alternate-stream copy path during CUDA graph capture to remove stream
  synchronization overhead.

- **CUDA graph capture minimalization**: capture only the batch sizes that
  actually occur in Suite F offline (1..64 with steps) rather than the
  default 1..160 dense ladder, reducing capture time and memory.

- **Scheduler fast path**: for the synchronous offline batch, skip dynamic
  prefill/decode interleaving once all requests are in decode phase and use
  larger continuous-decode steps.

- **Explored but not chosen unless measured faster**: FlashInfer backend
  variants, custom hand-written Triton fused MLP, multi-instance data-parallel.

## Experiment log

See the v5 autoresearch summary for the per-attempt results.
