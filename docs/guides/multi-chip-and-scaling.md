# Multi-Chip & Scaling Guides (Suites B, E, G)

Three AccelMark suites evaluate multi-accelerator setups:

| Suite | Model | Typical Chips | What It Measures |
|---|---|---|---|
| **Suite B** | Llama-3-70B | 4 or 8 chips | Large-model VRAM capacity & tensor-parallel serving |
| **Suite E** | Llama-3-8B | 1× / 2× / 4× / 8× | Linear scaling efficiency & interconnect Amdahl overhead |
| **Suite G** | Mixtral-8x7B | ≥2 chips (auto-scaled) | Sparse MoE routing & aggregate HBM bandwidth |

## Running Suite B (70B TP)

```bash
python run.py --runner auto --suite suite_B --tensor-parallel-size 4
# Or on an 8-chip node:
python run.py --runner auto --suite suite_B --tensor-parallel-size 8
```

## Running Suite E (Scaling Sweep)

Suite E automatically sweeps across available chip counts (`1, 2, 4, 8`) on your node and computes `scaling_efficiency_2x`, `scaling_efficiency_4x`, and `scaling_efficiency_8x`:

```bash
python run.py --runner auto --suite suite_E
```

## Running Suite G (Sparse MoE)

```bash
python run.py --runner auto --suite suite_G --tensor-parallel-size 8
```

!!! tip "Why does H200 have lower 2× TP efficiency than RTX 4090 D?"
    Scaling efficiency tracks the ratio of per-die compute throughput to interconnect bandwidth (Amdahl's law). Because H200 finishes each decode step far faster than a 4090 D, all-reduce synchronization consumes a larger fraction of step time — even over NVLink 4.0. For dense models that fit on 1 chip, prefer **data-parallel replicas** over TP > 4.
