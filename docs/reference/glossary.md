# Glossary

Consistent terminology used across the AccelMark leaderboard, CLI, paper, and documentation:

| Term | Definition |
|---|---|
| **Serving Recipe** | A complete, auditable deployment tuple `(chip × chip_count × framework × version × precision × parallelism)`. Every row on AccelMark is a serving recipe, never an uncontextualized chip score. |
| **Suite (`A`–`H`)** | A pinned workload contract (model, prompt distribution, token lengths, accuracy baseline, and scenarios) designed to isolate one Roofline bottleneck. |
| **Scenario** | One measurement mode inside a suite (`accuracy`, `offline`, `online`, `interactive`, `sustained`, `burst`, `speculative`). |
| **Runner (`{platform}_{name}_{hash8}`)** | A self-contained Python adapter in `runners/` whose 8-char suffix is the SHA-256 prefix of `runner.py`. |
| **Closed vs Open Division (Runner Tier)** | **Closed (`reference`)** = default out-of-the-box engine stacks (shown by default on the leaderboard). **Open (`optimized` / `experimental`)** = heavily tuned or custom-kernel forks, cleanly toggleable without contaminating reference baselines. |
| **Submission Tier (`community` → `verified`)** | **`community`** = CI-validated PR (immediately published). **`verified`** = independently reproduced by a second submitter on matching hardware & runner within **±5%**. |
| **MMLU Accuracy Gate** | Mandatory 100-question MMLU check run on the exact loaded model instance before throughput benchmarking (`Δ ≤ 0.03` vs baseline). |
| **SLA Fail** | Shown on `Online QPS` when an accelerator cannot sustain even the lowest Poisson arrival rate under the `p99 TTFT ≤ 500 ms` latency SLA. |
