#!/bin/bash
# AccelMark full benchmark run — 8 GPU
#
# ── Scheduling overview ───────────────────────────────────────────────────────
#
# Stage 1  — Suite B (all 8 GPUs, sequential, ~91 min)
#   70B model requires all 8 GPUs. Scenarios run one at a time.
#   Now includes B/burst as a new extra scenario.
#
# Stage 2a — Accuracy gates for A/D/F in parallel (~3 min)
#   Suite C accuracy runs inside its own scenario subprocesses (per precision);
#   no separate accuracy gate is needed here.
#
# Stage 2b — Global scheduler: A/C/D/F single-GPU scenarios (~38 min)
#   18 scenarios assigned to 8 GPUs using longest-first greedy.
#   Includes 3 new scenarios vs original: A/speculative, A/burst, D/speculative.
#
#   GPU 0 (~38m): D/online(38m)
#   GPU 1 (~35m): A/sustained(32m) → F/online(3m)
#   GPU 2 (~35m): D/sustained(31m) → A/speculative(4m)
#   GPU 3 (~33m): D/interactive(27m) → F/interactive(6m)
#   GPU 4 (~33m): C/sustained(25m) → A/burst(8m)
#   GPU 5 (~33m): C/offline(20m) → A/interactive(8m) → D/speculative(5m)
#   GPU 6 (~33m): C/online(20m) → A/online(8m) → A/offline(4m) → F/offline(1m)
#   GPU 7 (~33m): D/offline(18m) → F/sustained(15m)
#
#   Makespan: ~38 min  (same as original despite 3 new scenarios)
#   GPU utilization: ~90%
#
# Stage 2c — Merge pass for A/C/D/F (~1 min)
#
# Stage 2.5 — Suite G / Mixtral-8x7B (4 GPUs, sequential, ~80 min)
#   Mixtral-8x7B (~90GB BF16) requires 4×A100-40GB.
#   Uses GPUs 0-3 (GPUs 4-7 idle during this stage).
#   Scenarios run sequentially since all share the same 4-GPU pool.
#   G/accuracy(3m) → G/offline(10m) → G/online(15m) →
#   G/interactive(20m) → G/sustained(32m)
#
# Stage 3  — Suite E chip-count sweep (all 8 GPUs, ~30 min)
#   Sweeps 1×/2×/4×/8× chip counts in a single orchestrated run.
#
# ── Total wall-clock time ─────────────────────────────────────────────────────
#   Stage 1:    ~91 min
#   Stage 2:    ~42 min  (2a + 2b + 2c)
#   Stage 2.5:  ~80 min
#   Stage 3:    ~30 min
#   Total:     ~243 min  (~4.1 h)
#
# ── Notes ─────────────────────────────────────────────────────────────────────
# Suite C: each scenario runs all 5 precision formats internally
#   (BF16/FP8/W8A8/W8A16/W4A16). merge_suite is not called for Suite C.
#
# Merge pass: parallel scenarios within the same suite may race on writing
#   the suite-level result.json. The Stage 2c merge_suite calls run after
#   Stage 2b completes — all subdirs already have result.json so the framework
#   skips re-execution and does one clean uncontested merge.
#
# D/speculative: expected to show near-zero speedup (prefill-dominated workload
#   at 28K tokens). This is the intended finding — speculative decoding does not
#   help compute-bound workloads.
#
# All scenarios run to completion even if some fail (no set -e).

RUNNER_ID='ascend_vllm_ascend_d4aa9fda'
PYTHON_BIN="${1:-python}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_scenario() {
    local suite=$1 scenario=$2 gpus=$3
    log "START  $suite/$scenario (GPU $gpus)"
    ASCEND_RT_VISIBLE_DEVICES=$gpus \
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
    ASCEND_RT_VISIBLE_DEVICES=$gpus \
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
# 70B model needs all 8 GPUs per scenario — strictly sequential.
# burst is a new extra scenario added to Suite B.
log "===== Stage 1: Suite B (GPUs 0-7, ~91 min) ====="

run_scenario suite_B accuracy    0,1,2,3,4,5,6,7   # ~3m
run_scenario suite_B offline     0,1,2,3,4,5,6,7   # ~6m
run_scenario suite_B online      0,1,2,3,4,5,6,7   # ~14m
run_scenario suite_B interactive 0,1,2,3,4,5,6,7   # ~26m
run_scenario suite_B sustained   0,1,2,3,4,5,6,7   # ~32m
run_scenario suite_B burst       0,1,2,3,4,5,6,7   # ~10m  NEW
merge_suite  suite_B             0,1,2,3,4,5,6,7

log "Stage 1 complete."

# ── Stage 2a: Accuracy gates ──────────────────────────────────────────────────
# A/D/F accuracy in parallel. Suite C skips — accuracy runs per-precision
# inside its scenario subprocesses.
log "===== Stage 2a: Accuracy gates (A/D/F in parallel, ~3 min) ====="

run_scenario suite_A accuracy 0 &   # ~2m
run_scenario suite_D accuracy 1 &   # ~3m
run_scenario suite_F accuracy 2 &   # ~1m
wait

# ── Stage 2b: Global scheduler — 8 GPUs, ~38 min ─────────────────────────────
# 18 scenarios (15 original + A/speculative + A/burst + D/speculative)
# assigned to 8 GPUs using longest-first greedy. Makespan unchanged at ~38m.
log "===== Stage 2b: Benchmark scenarios — global schedule (GPUs 0-7, ~38 min) ====="

(                                           # GPU 0 (~38m)
    run_scenario suite_D online       0    #   38m
) &

(                                           # GPU 1 (~35m)
    run_scenario suite_A sustained    1    #   32m
    run_scenario suite_F online       1    #    3m
) &

(                                           # GPU 2 (~35m)
    run_scenario suite_D sustained    2    #   31m
    run_scenario suite_A speculative  2    #    4m  NEW
) &

(                                           # GPU 3 (~33m)
    run_scenario suite_D interactive  3    #   27m
    run_scenario suite_F interactive  3    #    6m
) &

(                                           # GPU 4 (~33m)
    run_scenario suite_C sustained    4    #   25m
    run_scenario suite_A burst        4    #    8m  NEW
) &

(                                           # GPU 5 (~33m)
    run_scenario suite_C offline      5    #   20m
    run_scenario suite_A interactive  5    #    8m
    run_scenario suite_D speculative  5    #    5m  NEW
) &

(                                           # GPU 6 (~33m)
    run_scenario suite_C online       6    #   20m
    run_scenario suite_A online       6    #    8m
    run_scenario suite_A offline      6    #    4m
    run_scenario suite_F offline      6    #    1m
) &

(                                           # GPU 7 (~33m)
    run_scenario suite_D offline      7    #   18m
    run_scenario suite_F sustained    7    #   15m
) &

wait

# ── Stage 2c: Final merge ─────────────────────────────────────────────────────
# Rebuilds suite-level result.json from completed scenario subdirs.
# Suite C is exempt — it has its own merge logic.
log "===== Stage 2c: Final merge ====="

merge_suite suite_A 0 &
merge_suite suite_D 1 &
merge_suite suite_F 2 &
wait

log "Stage 2 complete."

# ── Stage 2.5: Suite G (Mixtral-8x7B, 4 GPUs) ────────────────────────────────
# Mixtral-8x7B requires ~90GB BF16 — needs 4×A100-40GB (160GB total).
# required_chips=auto means the runner uses all ASCEND_RT_VISIBLE_DEVICES,
# so we expose exactly 4 GPUs. GPUs 4-7 are idle during this stage.
# All scenarios share the same 4-GPU pool and run sequentially.
log "===== Stage 2.5: Suite G / Mixtral-8x7B (GPUs 0-3, ~80 min) ====="

run_scenario suite_G accuracy    0,1,2,3,4,5,6,7   # ~3m
run_scenario suite_G offline     0,1,2,3,4,5,6,7   # ~10m
run_scenario suite_G online      0,1,2,3,4,5,6,7  # ~15m
run_scenario suite_G interactive 0,1,2,3,4,5,6,7   # ~20m
run_scenario suite_G sustained   0,1,2,3,4,5,6,7   # ~32m
merge_suite  suite_G             0,1,2,3,4,5,6,7

log "Stage 2.5 complete."

# ── Stage 3: Suite E ──────────────────────────────────────────────────────────
# Chip-count sweep 1×/2×/4×/8× — needs all 8 GPUs, orchestrated internally.
log "===== Stage 3: Suite E (GPUs 0-7, chip-count sweep, ~30 min) ====="

run_scenario suite_E accuracy 0,1,2,3,4,5,6,7   # ~2m

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    $PYTHON_BIN run.py \
        --runner "$RUNNER_ID" \
        --suite suite_E \
        --tier verified \
        --scenario offline \
        --max-chips 8 \
    && log "OK     suite_E/offline (all chip counts)" \
    || log "FAILED suite_E/offline (exit $?)"

merge_suite suite_E 0,1,2,3,4,5,6,7

log "Stage 3 complete."
log "===== All Done (~243 min total) ====="
