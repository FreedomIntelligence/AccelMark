#!/bin/bash
# AccelMark full benchmark run — 8 GPU
#
# Stage 2 uses a global GPU pool scheduler rather than per-suite GPU blocks.
# Each GPU is assigned a sequence of scenarios (possibly from different suites)
# so no GPU sits idle while another suite's long task is running.
#
# Scheduling is longest-first greedy across all 15 Stage 2 benchmark scenarios.
# Each GPU runs its assigned scenarios sequentially; all 8 GPUs run in parallel.
#
# Stage 2 GPU assignment (A100-80GB timing estimates):
#   GPU 0 (~38m): D/online(38m)
#   GPU 1 (~35m): A/interactive(35m)
#   GPU 2 (~33m): D/sustained(31m) → F/online(3m)
#   GPU 3 (~34m): A/sustained(30m) → A/offline(3m) → F/offline(1m)
#   GPU 4 (~33m): D/interactive(27m) → F/interactive(6m)
#   GPU 5 (~33m): C/sustained(25m) → A/online(8m)
#   GPU 6 (~38m): C/offline(20m) → D/offline(18m)
#   GPU 7 (~35m): C/online(20m) → F/sustained(15m)
#
#   Makespan: ~38.5 min  (vs ~68 min with per-suite blocking)
#   GPU utilization: ~91%
#
# Suite C note: each scenario (offline/online/sustained) runs all 5 precision
#   formats internally (BF16/FP8/W8A8/W8A16/W4A16) — no separate accuracy gate.
#   Suite C has its own merge logic; merge_suite is not called for it.
#
# Merge note: parallel scenarios within the same suite race on writing the
#   suite-level result.json. merge_suite runs --scenario all after Stage 2
#   completes — all subdirs already have result.json so the framework skips
#   execution and does one clean uncontested merge.
#
# All scenarios run to completion even if some fail (no set -e).

RUNNER_ID='nvidia_vllm_47f5d58e'

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_scenario() {
    local suite=$1 scenario=$2 gpus=$3
    log "START  $suite/$scenario (GPU $gpus)"
    CUDA_VISIBLE_DEVICES=$gpus \
        python run.py \
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
        python run.py \
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
log "===== Stage 1: Suite B (GPUs 0-7, ~76 min) ====="

run_scenario suite_B accuracy    0,1,2,3,4,5,6,7
run_scenario suite_B offline     0,1,2,3,4,5,6,7   # ~6m
run_scenario suite_B online      0,1,2,3,4,5,6,7   # ~14m
run_scenario suite_B interactive 0,1,2,3,4,5,6,7   # ~26m
run_scenario suite_B sustained   0,1,2,3,4,5,6,7   # ~31m
merge_suite  suite_B             0,1,2,3,4,5,6,7

log "Stage 1 complete."

# ── Stage 2a: Accuracy gates ──────────────────────────────────────────────────
# A/D/F accuracy in parallel. Suite C skips — accuracy runs per-precision
# inside its scenario subprocesses.
log "===== Stage 2a: Accuracy gates (A/D/F in parallel) ====="

run_scenario suite_A accuracy 0 &
run_scenario suite_D accuracy 1 &
run_scenario suite_F accuracy 2 &
wait

# ── Stage 2b: Global scheduler — all 8 GPUs, ~38.5 min ───────────────────────
# Each GPU runs its sequence of scenarios independently.
# Different suites can share a GPU — they just run sequentially on that GPU.
log "===== Stage 2b: Benchmark scenarios — global schedule (GPUs 0-7, ~38.5 min) ====="

( run_scenario suite_D online       0 ) &   # GPU 0: ~38m

( run_scenario suite_A interactive  1 ) &   # GPU 1: ~35m

(                                           # GPU 2: ~33m
    run_scenario suite_D sustained  2
    run_scenario suite_F online     2
) &

(                                           # GPU 3: ~34m
    run_scenario suite_A sustained  3
    run_scenario suite_A offline    3
    run_scenario suite_F offline    3
) &

(                                           # GPU 4: ~33m
    run_scenario suite_D interactive 4
    run_scenario suite_F interactive 4
) &

(                                           # GPU 5: ~33m
    run_scenario suite_C sustained  5
    run_scenario suite_A online     5
) &

(                                           # GPU 6: ~38m
    run_scenario suite_C offline    6
    run_scenario suite_D offline    6
) &

(                                           # GPU 7: ~35m
    run_scenario suite_C online     7
    run_scenario suite_F sustained  7
) &

wait

# Final clean merge for each suite (no-op on scenarios, just rebuilds result.json).
# Suite C is exempt — it has its own merge logic.
log "===== Stage 2c: Final merge ====="
merge_suite suite_A 0 &
merge_suite suite_D 1 &
merge_suite suite_F 2 &
wait

log "Stage 2 complete."

# ── Stage 3: Suite E ──────────────────────────────────────────────────────────
log "===== Stage 3: Suite E (GPUs 0-7, chip-count sweep 1x/2x/4x/8x) ====="

run_scenario suite_E accuracy 0,1,2,3,4,5,6,7

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python run.py \
        --runner "$RUNNER_ID" \
        --suite suite_E \
        --tier verified \
        --scenario offline \
        --max-chips 8 \
    && log "OK     suite_E/offline (all chip counts)" \
    || log "FAILED suite_E/offline (exit $?)"

merge_suite suite_E 0,1,2,3,4,5,6,7

log "Stage 3 complete."
log "===== All Done ====="
