#!/usr/bin/env python3
"""
Batch driver for running the full per-card benchmark bundle.

Runs SGLang all suites → vLLM all suites → profiling in order,
writing results to results/community/. Idempotent/resumable — skips
suites whose result.json already exists (unless --force is passed).

Usage::

    # Run P1.1 (1× A100-40G) — the anchor pair
    python tools/batch_driver.py --card a100_40g

    # Run with force (re-run even if results exist)
    python tools/batch_driver.py --card a100_40g --force

    # Dry-run: print what would be executed
    python tools/batch_driver.py --card a100_40g --dry-run

    # Resume from a specific phase
    python tools/batch_driver.py --card a100_40g --resume-from profiling

Card configs are defined inline below.  Add a new card by adding an
entry to ``CARD_CONFIGS``.

Environment selection
---------------------
Each runner may need a different conda/pip environment. The driver
uses ``conda run -n <env>`` when a ``conda_env`` is configured for
that runner. If unset it runs in whatever Python is on PATH.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = _REPO_ROOT / "results" / "community"


def _detect_chip_slug() -> str:
    """Best-effort detection of the current GPU for filtering existing results.

    Returns a slug like ``nvidia_a100_sxm4_40gb`` or ``unknown``.
    """
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        )
        chip = out.strip().splitlines()[0].strip()
        slug = __import__("re").sub(r"[^a-z0-9_]", "", __import__("re").sub(r"[ /\-]+", "_", chip.lower()))
        return slug or "unknown"
    except Exception:
        return "unknown"

# ── Card configs ──────────────────────────────────────────────────────────────
# Each card is a list of phases executed in order.  A phase is:
#   {"type": "benchmark", "runner": "<runner_id>", "suite": "<suite_id>",
#    "scenario": "all", "conda_env": "<env_or_null>"}
#   {"type": "profiling", "suite": "<suite_id>",
#    "conda_env": "<env_or_null>"}
#   {"type": "cooldown", "minutes": <float>}

@dataclass
class CardConfig:
    """Per-card configuration for the batch driver."""
    name: str
    description: str
    phases: list[dict] = field(default_factory=list)


# Standard single-chip bundle: SGLang suites A/C/D/F → vLLM suites A/C/D/F → profiling A/D
def _single_chip_bundle(
    sglang_env: str | None = None,
    vllm_env: str | None = None,
    profiling_env: str | None = None,
    *,
    sglang_runner: str = "nvidia_sglang_c43a8309",
    vllm_runner: str = "nvidia_vllm_47f5d58e",
) -> list[dict]:
    """Build the standard single-chip bundle phases."""
    suites = ["suite_A", "suite_C", "suite_D", "suite_F"]
    phases: list[dict] = []

    # ── SGLang ──
    for suite in suites:
        phases.append({
            "type": "benchmark",
            "runner": sglang_runner,
            "suite": suite,
            "scenario": "all",
            "conda_env": sglang_env,
        })
        phases.append({"type": "cooldown", "minutes": 2.0})

    # ── vLLM (energy) ──
    for suite in suites:
        phases.append({
            "type": "benchmark",
            "runner": vllm_runner,
            "suite": suite,
            "scenario": "all",
            "conda_env": vllm_env,
        })
        phases.append({"type": "cooldown", "minutes": 2.0})

    # ── Profiling ──
    for suite in ["suite_A", "suite_D"]:
        phases.append({
            "type": "profiling",
            "suite": suite,
            "conda_env": profiling_env,
        })
        phases.append({"type": "cooldown", "minutes": 1.0})

    return phases


# Standard multi-chip bundle: SGLang + vLLM on suites B, E, G
def _multi_chip_bundle(
    sglang_env: str | None = None,
    vllm_env: str | None = None,
    *,
    sglang_runner: str = "nvidia_sglang_c43a8309",
    vllm_runner: str = "nvidia_vllm_47f5d58e",
) -> list[dict]:
    """Build the standard 8-GPU multi-chip bundle phases."""
    suites = ["suite_B", "suite_E", "suite_G"]
    phases: list[dict] = []
    for suite in suites:
        phases.append({
            "type": "benchmark",
            "runner": sglang_runner,
            "suite": suite,
            "scenario": "all",
            "conda_env": sglang_env,
        })
        phases.append({"type": "cooldown", "minutes": 2.0})
    for suite in suites:
        phases.append({
            "type": "benchmark",
            "runner": vllm_runner,
            "suite": suite,
            "scenario": "all",
            "conda_env": vllm_env,
        })
        phases.append({"type": "cooldown", "minutes": 2.0})
    return phases


# ── Registered cards ───────────────────────────────────────────────────────────
# To add a new card, add an entry here and provision the GPU.
CARD_CONFIGS: dict[str, CardConfig] = {
    "a100_40g": CardConfig(
        name="A100-SXM4-40GB",
        description="P1.1 — anchor pair: SGLang vs vLLM on A100-40G",
        phases=_single_chip_bundle(sglang_env="sglang"),
    ),
    "a100_80g": CardConfig(
        name="A100-SXM4-80GB",
        description="P1.5 — Cycle 1 profiling partner",
        phases=_single_chip_bundle(),
    ),
    "h100": CardConfig(
        name="H100-80GB",
        description="P1.4 — SLA champion, policy relevance",
        phases=_single_chip_bundle(),
    ),
    "h20": CardConfig(
        name="H20-3e",
        description="P1.2 — framework-reversal candidate",
        phases=_single_chip_bundle(),
    ),
    "rtx5090": CardConfig(
        name="RTX 5090",
        description="P1.3 — SLA/framework story",
        phases=_single_chip_bundle(),
    ),
    "a100_40g_8x": CardConfig(
        name="8× A100-SXM4-40GB",
        description="P1.6 — framework diff at scale (70B + MoE)",
        phases=_multi_chip_bundle(),
    ),
    "h20_8x": CardConfig(
        name="8× H20-3e",
        description="P1.7 — MoE at scale",
        phases=_multi_chip_bundle(),
    ),
    "h200_8x": CardConfig(
        name="8× H200",
        description="P1.8 — offline-vs-SLA inversion at scale",
        phases=_multi_chip_bundle(),
    ),
}


# ── Phase execution ──────────────────────────────────────────────────────────

def _find_existing_result(runner_id: str, suite_id: str, chip_slug: str | None = None) -> list[Path]:
    """Return any existing result directories matching this runner+suite+chip combo.

    Looks for directories whose name contains the runner_id and suite_id.
    When ``chip_slug`` is provided (recommended), only matches directories
    whose name starts with that slug — this prevents confusing an A800 run
    with an A100 run when they share the same runner.

    Returns a list (usually 0 or 1 entries).
    """
    if not RESULTS_DIR.exists():
        return []
    matches = []
    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if runner_id in name and suite_id in name:
            if chip_slug and not name.startswith(chip_slug):
                continue
            matches.append(d)
    return matches


def _is_suite_complete(result_dir: Path, scenario: str = "all") -> bool:
    """Check if a result directory has completed benchmark results with power data.

    For ``scenario="all"``, checks both the top-level result.json and
    per-scenario subdirectories for evidence of completion.
    Also verifies that at least one scenario has ``power_watts_avg`` set —
    results without power data were produced before the energy collection
    feature and should be re-run.
    """
    result_json = result_dir / "result.json"
    if not result_json.exists():
        return False
    try:
        data = json.loads(result_json.read_text())
        metrics = data.get("metrics", {})
        if not metrics:
            return False

        # Check that at least one scenario has power data.
        # Results without power were produced before the energy feature;
        # they need to be re-run.
        has_power = _has_power_data(metrics)
        if not has_power:
            return False

        return True
    except Exception:
        return False


def _has_power_data(metrics: dict) -> bool:
    """Recursively check if any power_watts_avg is present in the metrics tree."""
    if isinstance(metrics, dict):
        if "power_watts_avg" in metrics and metrics["power_watts_avg"] is not None:
            return True
        # Check nested structures: list of dicts, sub-dicts
        for key, value in metrics.items():
            if key == "power_watts_avg" and value is not None:
                return True
            if isinstance(value, (dict, list)):
                if _has_power_data(value):
                    return True
    elif isinstance(metrics, list):
        for item in metrics:
            if isinstance(item, (dict, list)):
                if _has_power_data(item):
                    return True
    return False


def _find_profiling_result(suite_id: str) -> Optional[Path]:
    """Return the path to an existing profiling JSON for the given suite."""
    if not RESULTS_DIR.exists():
        return None
    for f in sorted(RESULTS_DIR.glob("*_intensity.json")):
        if suite_id in f.name:
            return f
    return None


def _chip_slug_for_card(card_config: CardConfig) -> str:
    """Derive a chip slug from the card config for profiling output naming."""
    return card_config.name.lower().replace(" ", "_").replace("×", "x").replace("-", "_")


def run_benchmark_phase(
    phase: dict,
    card: CardConfig,
    *,
    chip_slug: str = "unknown",
    force: bool = False,
    dry_run: bool = False,
) -> bool:
    """Execute a single benchmark phase. Returns True on success."""
    runner_id = phase["runner"]
    suite_id = phase["suite"]
    scenario = phase.get("scenario", "all")
    conda_env = phase.get("conda_env")

    # Check if already done
    existing = _find_existing_result(runner_id, suite_id, chip_slug=chip_slug)
    already_done = any(_is_suite_complete(d) for d in existing)
    existing_but_stale = bool(existing) and not already_done

    if already_done and not force:
        print(f"  ⏭  SKIP: {runner_id} / {suite_id} — already completed")
        for d in existing:
            if _is_suite_complete(d):
                print(f"       → {d.name}")
        return True

    if dry_run:
        if existing_but_stale:
            status = "will run (existing results lack power data → auto --force)"
        elif existing and force:
            status = "FORCE re-run"
        else:
            status = "will run"
        print(f"  📋 {status}: {runner_id} / {suite_id} / {scenario}")
        return True

    if existing_but_stale:
        print(f"  🔄 Re-run (existing results lack power/energy data): {runner_id} / {suite_id}")
        force = True  # auto-enable --force to overwrite stale results
    elif already_done and force:
        print(f"  🔄 FORCE re-run: {runner_id} / {suite_id}")

    # Build command
    cmd_parts = [
        "python", "run.py",
        "--runner", runner_id,
        "--suite", suite_id,
        "--scenario", scenario,
        "--tier", "community",
    ]
    if force:
        cmd_parts.append("--force")

    cmd_str = " ".join(cmd_parts)

    print(f"  ▶ Running: {cmd_str}")
    print(f"    Card: {card.name}  |  Runner: {runner_id}  |  Suite: {suite_id}")

    start = time.perf_counter()
    try:
        if conda_env:
            result = subprocess.run(
                ["conda", "run", "-n", conda_env, "--no-capture-output"] + cmd_parts,
                cwd=str(_REPO_ROOT),
                check=False,
            )
        else:
            result = subprocess.run(
                cmd_parts,
                cwd=str(_REPO_ROOT),
                check=False,
                shell=False,  # Use list form — safer for paths with spaces
            )
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False

    elapsed = time.perf_counter() - start
    if result.returncode == 0:
        print(f"  ✅ OK ({elapsed / 60:.1f} min)")
        return True
    else:
        print(f"  ❌ FAILED (exit code {result.returncode}, {elapsed / 60:.1f} min)")
        return False


def run_profiling_phase(
    phase: dict,
    card: CardConfig,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> bool:
    """Execute a profiling phase. Returns True on success."""
    suite_id = phase["suite"]
    conda_env = phase.get("conda_env")
    chip_slug = _chip_slug_for_card(card)

    # Resolve output path
    out_name = f"{chip_slug}_{suite_id}_intensity.json"
    out_path = RESULTS_DIR / out_name

    # Check if already done
    existing = _find_profiling_result(suite_id)
    if existing and not force:
        print(f"  ⏭  SKIP: profiling {suite_id} — already completed ({existing.name})")
        return True

    if dry_run:
        status = "FORCE re-run" if (existing and force) else "will run"
        print(f"  📋 {status}: profiling {suite_id} → {out_name}")
        return True

    if existing and force:
        print(f"  🔄 FORCE re-run: profiling {suite_id}")

    cmd_parts = [
        "python", "tools/profile_intensity.py",
        "--suite", suite_id,
        "--out", str(out_path),
    ]
    cmd_str = " ".join(cmd_parts)

    print(f"  ▶ Running: {cmd_str}")
    print(f"    Card: {card.name}  |  Suite: {suite_id}  |  Output: {out_name}")

    start = time.perf_counter()
    try:
        if conda_env:
            result = subprocess.run(
                ["conda", "run", "-n", conda_env, "--no-capture-output"] + cmd_parts,
                cwd=str(_REPO_ROOT),
                check=False,
            )
        else:
            result = subprocess.run(
                cmd_parts,
                cwd=str(_REPO_ROOT),
                check=False,
            )
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False

    elapsed = time.perf_counter() - start
    if result.returncode == 0:
        print(f"  ✅ OK ({elapsed / 60:.1f} min)")
        return True
    else:
        print(f"  ❌ FAILED (exit code {result.returncode}, {elapsed / 60:.1f} min)")
        return False


def run_cooldown(minutes: float, dry_run: bool = False) -> None:
    """Sleep to let the GPU cool down between suites."""
    if dry_run:
        print(f"  🕐 Cooldown: {minutes} min (skipped in dry-run)")
        return
    print(f"  🕐 Cooldown: {minutes} min — letting GPU cool...")
    time.sleep(minutes * 60)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="AccelMark batch driver — run the full per-card bundle",
    )
    parser.add_argument(
        "--card", default=None,
        help="Card config key (e.g. a100_40g, h20, rtx5090)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run phases even if results already exist",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be executed without running",
    )
    parser.add_argument(
        "--resume-from", default=None,
        help="Phase label to resume from (format: <type>:<runner>:<suite>, "
             "e.g. 'benchmark:nvidia_vllm_47f5d58e:suite_A'). "
             "Phases before this are skipped.",
    )
    parser.add_argument(
        "--list-cards", action="store_true",
        help="List available card configs and exit",
    )
    args = parser.parse_args()

    if args.list_cards:
        print("Available card configs:\n")
        for key, cfg in sorted(CARD_CONFIGS.items()):
            print(f"  {key:15s}  {cfg.name}")
            print(f"  {'':15s}  {cfg.description}")
            print()
        return 0

    if not args.card:
        print("Error: --card is required (use --list-cards to see options)")
        return 1

    card = CARD_CONFIGS.get(args.card)
    if card is None:
        print(f"Error: unknown card '{args.card}'")
        print(f"Available: {', '.join(sorted(CARD_CONFIGS))}")
        return 1

    phases = card.phases
    if not phases:
        print(f"Card '{args.card}' ({card.name}) has no phases defined.")
        return 0

    # Detect the actual chip for filtering existing results
    chip_slug = _detect_chip_slug()
    print(f"(detected chip: {chip_slug})")
    print()

    # ── Resolve resume-from ──────────────────────────────────────────────────
    skip_until = None
    if args.resume_from:
        resume_id = args.resume_from
        skip_until = resume_id
        print(f"⏩ Resuming from phase: {resume_id}")
        print()

    # ── Execute phases ───────────────────────────────────────────────────────
    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  Card: {card.name:52s} ║")
    print(f"║  Phases: {len(phases):3d}                                          ║")
    if args.dry_run:
        print(f"║  MODE: DRY-RUN                                              ║")
    elif args.force:
        print(f"║  MODE: FORCE                                                ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")
    print()

    total = len(phases)
    passed = 0
    failed = 0
    skipped = 0

    for i, phase in enumerate(phases):
        ptype = phase.get("type", "benchmark")

        # Build phase label for resume matching
        if ptype == "benchmark":
            label = f"{ptype}:{phase['runner']}:{phase['suite']}"
        elif ptype == "profiling":
            label = f"{ptype}:{phase['suite']}"
        elif ptype == "cooldown":
            label = f"cooldown:{phase.get('minutes', 0)}min"
        else:
            label = f"{ptype}"

        # Resume logic
        if skip_until is not None:
            if label == skip_until:
                skip_until = None  # found resume point, start executing
                print(f"[{i+1}/{total}] {label}  ← RESUMING HERE")
            else:
                print(f"[{i+1}/{total}] {label}  ⏩ skipped (before resume point)")
                skipped += 1
                continue

        print(f"[{i+1}/{total}] {label}")

        try:
            if ptype == "benchmark":
                ok = run_benchmark_phase(phase, card, chip_slug=chip_slug, force=args.force, dry_run=args.dry_run)
            elif ptype == "profiling":
                ok = run_profiling_phase(phase, card, force=args.force, dry_run=args.dry_run)
            elif ptype == "cooldown":
                run_cooldown(phase.get("minutes", 2.0), dry_run=args.dry_run)
                ok = True  # cooldowns always pass
            else:
                print(f"  ⚠ Unknown phase type: {ptype}")
                ok = True
        except KeyboardInterrupt:
            print(f"\n\n⏸  Interrupted after phase [{i+1}/{total}] {label}")
            print(f"   Resume with: --resume-from '{label}'")
            return 130

        if ok:
            passed += 1
        else:
            failed += 1
            print(f"\n  ⚠  Phase failed. Continuing with next phase...")
            print(f"     (Resume later with: --resume-from '{label}')")

        print()

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  DONE — {passed} passed, {failed} failed, {skipped} skipped"
          + f"{' (dry-run)' if args.dry_run else ''}")
    print(f"╚══════════════════════════════════════════════════════════════╝")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
