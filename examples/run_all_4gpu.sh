#!/bin/bash
# AccelMark full benchmark run — 4 GPU
#
# ── Scheduling overview ───────────────────────────────────────────────────────
#
# Stage 1  — Suite B (all 4 GPUs, sequential, ~91 min)
#   70B model requires all 4 GPUs. Scenarios run one at a time.
#   Now includes B/burst as a new extra scenario.
#
# Stage 2a — Accuracy gates for A/D/F in parallel (~3 min)
#
# Stage 2b — Global scheduler: A/C/D/F single-GPU scenarios (~69 min)
#   18 scenarios assigned to 4 GPUs using longest-first greedy.
#   Includes 3 new scenarios vs original: A/speculative, A/burst, D/speculative.
#
#   GPU 0 (~68m): D/online(38m) → D/offline(18m) → A/burst(8m) → A/offline(4m)
#   GPU 1 (~69m): A/sustained(32m) → C/online(20m) → A/interactive(8m) → F/interactive(6m) → F/online(3m)
#   GPU 2 (~67m): D/sustained(31m) → C/offline(20m) → F/sustained(15m) → F/offline(1m)
#   GPU 3 (~69m): D/interactive(27m) → C/sustained(25m) → A/online(8m) → D/speculative(5m) → A/speculative(4m)
#
#   Makespan: ~69 min   GPU utilization: ~99%
#
# Stage 2c — Merge pass for A/C/D/F (~1 min)
#
# Stage 2.5 — Suite G / Mixtral-8x7B (all 4 GPUs, sequential, ~80 min)
#   Mixtral-8x7B (~90GB BF16) needs 4×A100-40GB (160GB total).
#   required_chips=auto — runner uses all CUDA_VISIBLE_DEVICES.
#   Scenarios run sequentially since all share the same 4-GPU pool.
#   G/accuracy(3m) → G/offline(10m) → G/online(15m) →
#   G/interactive(20m) → G/sustained(32m)
#
# Stage 3  — Suite E chip-count sweep (all 4 GPUs, ~30 min)
#   Sweeps 1×/2×/4× chip counts in a single orchestrated run.
#
# ── Total wall-clock time ─────────────────────────────────────────────────────
#   Stage 1:    ~91 min
#   Stage 2:    ~73 min  (2a + 2b + 2c)
#   Stage 2.5:  ~80 min
#   Stage 3:    ~30 min
#   Total:     ~274 min  (~4.6 h)
#
# ── Notes ─────────────────────────────────────────────────────────────────────
# Suite C: each scenario runs all 5 precision formats internally.
#   No separate accuracy gate or merge_suite needed for Suite C.
#
# D/speculative: expected near-zero speedup (prefill-dominated at 28K tokens).
#   This is the intended finding — placed at end of GPU 3 sequence.
#
# All scenarios run to completion even if some fail (no set -e).

RUNNER_ID='nvidia_vllm_47f5d58e'
PYTHON_BIN="${1:-python}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_scenario() {
    local suite=$1 scenario=$2 gpus=$3
    log "START  $suite/$scenario (GPU $gpus)"
    CUDA_VISIBLE_DEVICES=$gpus \
        $PYTHON_BIN run.py \
            --runner "$RUNNER_ID" \
            --suite "$suite" \
            --tier verified \
            --scenario "$scenario" \
        && log "OK     $suite/$scenario" \
        || log "FAILED $suite/$scenario (exit $?)"
}

merge_suite() {
    local suite=$1 gpus=$2
    log "MERGE  $suite"
    CUDA_VISIBLE_DEVICES=$gpus \
        $PYTHON_BIN run.py \
            --runner "$RUNNER_ID" \
            --suite "$suite" \
            --tier verified \
            --scenario all \
            --skip-accuracy-gate \
        && log "OK     $suite merge" \
        || log "FAILED $suite merge (exit $?)"
}

# ── Stage 1: Suite B ──────────────────────────────────────────────────────────
# 70B model needs all 4 GPUs per scenario — strictly sequential.
# burst is a new extra scenario added to Suite B.
log "===== Stage 1: Suite B (GPUs 0-3, ~91 min) ====="

run_scenario suite_B accuracy    0,1,2,3   # ~3m
run_scenario suite_B offline     0,1,2,3   # ~6m
run_scenario suite_B online      0,1,2,3   # ~14m
run_scenario suite_B interactive 0,1,2,3   # ~26m
run_scenario suite_B sustained   0,1,2,3   # ~32m
run_scenario suite_B burst       0,1,2,3   # ~10m  NEW
merge_suite  suite_B             0,1,2,3

log "Stage 1 complete."

# ── Stage 2a: Accuracy gates ──────────────────────────────────────────────────
log "===== Stage 2a: Accuracy gates (A/D/F in parallel, ~3 min) ====="

run_scenario suite_A accuracy 0 &   # ~2m
run_scenario suite_D accuracy 1 &   # ~3m
run_scenario suite_F accuracy 2 &   # ~1m
wait

# ── Stage 2b: Global scheduler — 4 GPUs, ~69 min ─────────────────────────────
# 18 scenarios (15 original + A/speculative + A/burst + D/speculative).
log "===== Stage 2b: Benchmark scenarios — global schedule (GPUs 0-3, ~69 min) ====="

(                                               # GPU 0 (~68m)
    run_scenario suite_D online       0   #   38m
    run_scenario suite_D offline      0   #   18m
    run_scenario suite_A burst        0   #    8m  NEW
    run_scenario suite_A offline      0   #    4m
) &

(                                               # GPU 1 (~69m)
    run_scenario suite_A sustained    1   #   32m
    run_scenario suite_C online       1   #   20m
    run_scenario suite_A interactive  1   #    8m
    run_scenario suite_F interactive  1   #    6m
    run_scenario suite_F online       1   #    3m
) &

(                                               # GPU 2 (~67m)
    run_scenario suite_D sustained    2   #   31m
    run_scenario suite_C offline      2   #   20m
    run_scenario suite_F sustained    2   #   15m
    run_scenario suite_F offline      2   #    1m
) &

(                                               # GPU 3 (~69m)
    run_scenario suite_D interactive  3   #   27m
    run_scenario suite_C sustained    3   #   25m
    run_scenario suite_A online       3   #    8m
    run_scenario suite_D speculative  3   #    5m  NEW
    run_scenario suite_A speculative  3   #    4m  NEW
) &

wait

# ── Stage 2c: Final merge ─────────────────────────────────────────────────────
log "===== Stage 2c: Final merge ====="

merge_suite suite_A 0 &
merge_suite suite_D 1 &
merge_suite suite_F 2 &
wait

log "Stage 2 complete."

# ── Stage 2.5: Suite G (Mixtral-8x7B, all 4 GPUs) ────────────────────────────
# Mixtral-8x7B requires ~90GB BF16 — needs all 4×A100-40GB (160GB total).
# required_chips=auto means the runner uses all CUDA_VISIBLE_DEVICES.
# Scenarios run sequentially since all share the same 4-GPU pool.
log "===== Stage 2.5: Suite G / Mixtral-8x7B (GPUs 0-3, ~80 min) ====="

run_scenario suite_G accuracy    0,1,2,3   # ~3m
run_scenario suite_G offline     0,1,2,3   # ~10m
run_scenario suite_G online      0,1,2,3   # ~15m
run_scenario suite_G interactive 0,1,2,3   # ~20m
run_scenario suite_G sustained   0,1,2,3   # ~32m
merge_suite  suite_G             0,1,2,3

log "Stage 2.5 complete."

# ── Stage 3: Suite E ──────────────────────────────────────────────────────────
# Chip-count sweep 1×/2×/4× — needs all 4 GPUs, orchestrated internally.
log "===== Stage 3: Suite E (GPUs 0-3, chip-count sweep, ~30 min) ====="

run_scenario suite_E accuracy 0,1,2,3   # ~2m

CUDA_VISIBLE_DEVICES=0,1,2,3 \
    $PYTHON_BIN run.py \
        --runner "$RUNNER_ID" \
        --suite suite_E \
        --tier verified \
        --scenario offline \
        --max-chips 4 \
    && log "OK     suite_E/offline (all chip counts)" \
    || log "FAILED suite_E/offline (exit $?)"

merge_suite suite_E 0,1,2,3

log "Stage 3 complete."
log "===== All Done (~274 min total) ====="
