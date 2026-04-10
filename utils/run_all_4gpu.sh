#!/bin/bash
# AccelMark full benchmark run — 4 GPU
#
# Stage 2 uses a global GPU pool scheduler rather than per-suite GPU blocks.
#
# Stage 2 GPU assignment (A100-80GB timing estimates):
#   GPU 0 (~68m): D/online(38m) → C/online(20m) → F/interactive(6m) → A/offline(3m) → F/offline(1m)
#   GPU 1 (~73m): A/interactive(35m) → C/offline(20m) → D/offline(18m)
#   GPU 2 (~71m): D/sustained(31m) → C/sustained(25m) → F/sustained(15m)
#   GPU 3 (~68m): A/sustained(30m) → D/interactive(27m) → A/online(8m) → F/online(3m)
#
#   Makespan: ~73 min  (vs ~114 min with per-suite blocking)
#   GPU utilization: ~96%
#
# Suite C note: each scenario runs all 5 precision formats internally.
#   No separate accuracy gate or merge_suite needed for Suite C.
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
log "===== Stage 1: Suite B (GPUs 0-3, ~76 min) ====="

run_scenario suite_B accuracy    0,1,2,3
run_scenario suite_B offline     0,1,2,3   # ~6m
run_scenario suite_B online      0,1,2,3   # ~14m
run_scenario suite_B interactive 0,1,2,3   # ~26m
run_scenario suite_B sustained   0,1,2,3   # ~31m
merge_suite  suite_B             0,1,2,3

log "Stage 1 complete."

# ── Stage 2a: Accuracy gates ──────────────────────────────────────────────────
log "===== Stage 2a: Accuracy gates (A/D/F in parallel) ====="

run_scenario suite_A accuracy 0 &
run_scenario suite_D accuracy 1 &
run_scenario suite_F accuracy 2 &
wait

# ── Stage 2b: Global scheduler — all 4 GPUs, ~73 min ─────────────────────────
log "===== Stage 2b: Benchmark scenarios — global schedule (GPUs 0-3, ~73 min) ====="

(                                               # GPU 0: ~68m
    run_scenario suite_D online       0   # ~38m
    run_scenario suite_C online       0   # ~20m
    run_scenario suite_F interactive  0   # ~6m
    run_scenario suite_A offline      0   # ~3m
    run_scenario suite_F offline      0   # ~1m
) &

(                                               # GPU 1: ~73m
    run_scenario suite_A interactive  1   # ~35m
    run_scenario suite_C offline      1   # ~20m
    run_scenario suite_D offline      1   # ~18m
) &

(                                               # GPU 2: ~71m
    run_scenario suite_D sustained    2   # ~31m
    run_scenario suite_C sustained    2   # ~25m
    run_scenario suite_F sustained    2   # ~15m
) &

(                                               # GPU 3: ~68m
    run_scenario suite_A sustained    3   # ~30m
    run_scenario suite_D interactive  3   # ~27m
    run_scenario suite_A online       3   # ~8m
    run_scenario suite_F online       3   # ~3m
) &

wait

# Final clean merge (no-op on scenarios, just rebuilds result.json).
# Suite C is exempt — it has its own merge logic.
log "===== Stage 2c: Final merge ====="
merge_suite suite_A 0 &
merge_suite suite_D 1 &
merge_suite suite_F 2 &
wait

log "Stage 2 complete."

# ── Stage 3: Suite E ──────────────────────────────────────────────────────────
log "===== Stage 3: Suite E (GPUs 0-3, chip-count sweep 1x/2x/4x) ====="

run_scenario suite_E accuracy 0,1,2,3

CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python run.py \
        --runner "$RUNNER_ID" \
        --suite suite_E \
        --tier verified \
        --scenario offline \
        --max-chips 4 \
    && log "OK     suite_E/offline (all chip counts)" \
    || log "FAILED suite_E/offline (exit $?)"

merge_suite suite_E 0,1,2,3

log "Stage 3 complete."
log "===== All Done ====="
