"""
AccelMark Submission Validator
Validates all files in a submission directory before opening a PR.

Usage:
    python runners/validate_submission.py --dir results/community/a100x1_llama3-8b_suite-A_2026-03-22
    Exit code 0 = ready to submit. Exit code 1 = fix required.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import jsonschema


# Repo root — runners/validate_submission.py is two levels down from root
REPO_ROOT = Path(__file__).parent.parent

SCHEMA_DIR = REPO_ROOT / "schema"
MAX_VARIANCE_PCT = 15.0
ANOMALY_MULTIPLIER = 2.0


def load_schema(name: str) -> dict:
    with open(SCHEMA_DIR / name) as f:
        return json.load(f)


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def validate_schema(data: dict, schema: dict, label: str) -> list[str]:
    errors = []
    validator = jsonschema.Draft7Validator(schema)
    for error in sorted(validator.iter_errors(data), key=str):
        errors.append(f"[{label}] {error.path}: {error.message}")
    return errors


def check_hard_failures(result: dict, result_dir: Path) -> list[str]:
    failures = []

    # num_runs >= 3
    suite_id = result.get("suite_id")
    suite_path = REPO_ROOT / f"suites/{suite_id}/suite.json"
    min_runs = 3
    if suite_path.exists():
        with open(suite_path) as f:
            suite_config = json.load(f)
        min_runs = suite_config.get("num_runs", 3)

    if result.get("task", {}).get("num_runs", 0) < min_runs:
        failures.append(f"task.num_runs must be >= {min_runs} (required by {suite_id})")

    # accuracy.valid for inference suites
    scenario = result.get("task", {}).get("scenario")
    if scenario != "training":
        accuracy = result.get("accuracy")
        # Suite C stores accuracy per-format in metrics.quantization; top-level accuracy is null
        if accuracy is not None:
            if not accuracy.get("valid", False) and not accuracy.get("notes"):
                failures.append(
                    "accuracy.valid is false but accuracy.notes is empty. "
                    "Explain why accuracy is below threshold, or fix the precision configuration."
                )

    # referenced files must exist — search recursively under result_dir
    meta = result.get("meta", {})
    for field in ["reproduce_script", "env_info_file", "log_file"]:
        ref = meta.get(field)
        if not ref:
            continue
        ref_path = Path(ref)
        ref_name = ref_path.name
        if ref_path.exists() or any(result_dir.rglob(ref_name)):
            continue
        failures.append(f"meta.{field} references '{ref}' which does not exist")

    # model_revision must not be placeholder
    model_revision = result.get("model", {}).get("model_revision", "")
    if "TO_BE_LOCKED" in model_revision or model_revision == "":
        failures.append("model.model_revision must be set to an actual git commit hash")

    return failures


def check_soft_warnings(result: dict) -> list[str]:
    warnings = []

    meta = result.get("meta", {})
    if not meta.get("samples_file"):
        warnings.append("meta.samples_file is null — samples.jsonl is optional but strongly recommended")

    # Check power data availability
    metrics = result.get("metrics", {})
    offline = metrics.get("offline")
    if offline:
        has_power = any(
            row.get("power_watts_avg") is not None
            for row in (offline.get("results_by_concurrency") or offline.get("results_by_batch_size", []))
        )
        if not has_power:
            warnings.append(
                "No power data in offline metrics — "
                "tokens_per_sec_per_watt cannot be computed. "
                "Consider adding power measurement to your script."
            )

    # implementation_id check (soft — old results won't have this)
    impl_id = result.get("implementation_id")
    if not impl_id:
        warnings.append(
            "implementation_id not set. For new submissions, use a runner from "
            "runners/ so results are reproducible."
        )
    else:
        runner_dir = REPO_ROOT / "runners" / impl_id
        if not runner_dir.exists():
            warnings.append(
                f"implementation_id '{impl_id}' does not match any folder in runners/. "
                "Check the ID or submit your runner first."
            )

    return warnings


def check_run_id_integrity(result: dict) -> list[str]:
    """
    Verify meta.run_id matches a recomputation from the result's own fields.
    Catches manual edits to result.json that forget to update run_id.
    Silently skips results that don't have run_id (older submissions).
    """
    meta          = result.get("meta", {})
    stored_run_id = meta.get("run_id")

    if not stored_run_id:
        return []  # Old result without run_id — skip

    chip       = result.get("chip", {})
    sw         = result.get("software", {})
    model      = result.get("model", {})
    chip_count = chip.get("count", 1)

    key = {
        "chip_name":         chip.get("name", "unknown"),
        "chip_memory_gb":    chip.get("memory_gb_per_chip", 0),
        "chip_count":        chip_count,
        "interconnect":      chip.get("interconnect_intra_node") if chip_count > 1 else None,
        "runner_id":         result.get("implementation_id", "unknown"),
        "framework_version": sw.get("framework_version", "unknown"),
        "suite_id":          result.get("suite_id", "unknown"),
        "model_id":          model.get("model_id", "unknown"),
        "precision":         model.get("precision", "BF16"),
        "submitted_by":      meta.get("submitted_by", "unknown"),
    }

    raw      = json.dumps(key, sort_keys=True)
    expected = hashlib.sha256(raw.encode()).hexdigest()[:8]

    if stored_run_id != expected:
        return [
            f"meta.run_id mismatch — stored '{stored_run_id}' "
            f"but recomputed '{expected}'. "
            f"result.json may have been manually edited after the run."
        ]
    return []


def compute_derived(result: dict) -> dict:
    """Compute and inject derived metrics."""
    metrics = result.setdefault("metrics", {})
    derived = metrics.setdefault("derived", {})

    # tokens_per_sec_per_watt from offline metrics
    offline = metrics.get("offline")
    if offline:
        rows = offline.get("results_by_concurrency") or offline.get("results_by_batch_size", [])
        valid_rows = [r for r in rows if not r.get("oom") and r.get("power_watts_avg")]
        if valid_rows:
            best = max(valid_rows, key=lambda r: r.get("throughput_tokens_per_sec", 0))
            thr = best.get("throughput_tokens_per_sec", 0)
            pwr = best.get("power_watts_avg", 0)
            if pwr > 0:
                derived["tokens_per_sec_per_watt"] = round(thr / pwr, 4)

    # tokens_per_sec_per_chip
    chip_count = result.get("chip", {}).get("count", 1)
    training = metrics.get("training")
    if training and training.get("tokens_per_sec"):
        derived["tokens_per_sec_per_chip"] = round(
            training["tokens_per_sec"] / chip_count, 2
        )

    return result


def check_env_info(submission_dir: Path) -> list[str]:
    errors = []
    env_info_path = submission_dir / "env_info.json"
    if not env_info_path.exists():
        errors.append(
            "env_info.json not found — it must be in the task root directory"
        )
    return errors


def check_suite_e(submission_dir: Path, result: dict) -> list[str]:
    """Check Suite E specific requirements."""
    errors = []

    if result.get("suite_id") != "suite_E":
        return errors

    # Load suite config to get required counts
    suite_path = REPO_ROOT / "suites/suite_E/suite.json"
    if not suite_path.exists():
        return errors

    with open(suite_path) as f:
        suite = json.load(f)

    required_counts = suite.get("chip_counts_required", [1, 2])

    # Check chip_counts_run in task
    counts_run = result.get("task", {}).get("chip_counts_run", [])
    missing = [c for c in required_counts if c not in counts_run]
    if missing:
        errors.append(
            f"Suite E: required chip counts {missing} not found in task.chip_counts_run. "
            f"Re-run with at least --max-chips {max(required_counts)}."
        )

    # Check subdirectories exist
    for count in counts_run:
        count_dir = submission_dir / f"{count}x"
        if not count_dir.exists():
            errors.append(
                f"Suite E: expected subdirectory '{count}x/' not found in submission."
            )
        else:
            result_path = count_dir / "result.json"
            if not result_path.exists():
                errors.append(
                    f"Suite E: {count}x/result.json not found."
                )

    return errors


def find_env_info(submission_dir: Path) -> Path | None:
    """Return path to env_info.json at the task root, or None if not found."""
    top = submission_dir / "env_info.json"
    return top if top.exists() else None


def check_accuracy(submission_dir: Path, result: dict) -> list[str]:
    errors = []

    # Find all accuracy.json files anywhere under the submission directory
    candidates = list(submission_dir.rglob("accuracy.json"))

    if not candidates:
        errors.append("accuracy.json not found — run --scenario accuracy first")
        return errors

    for acc_path in candidates:
        with open(acc_path) as f:
            acc = json.load(f)
        if not acc.get("valid"):
            rel = acc_path.relative_to(submission_dir)
            errors.append(
                f"{rel}: accuracy.valid is false (score={acc.get('subset_score')}) "
                f"— model output quality check failed"
            )
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()

    result_dir = Path(args.dir)
    result_path = result_dir / "result.json"

    all_errors = []
    all_warnings = []

    # --- Validate result.json ---
    if not result_path.exists():
        print(f"ERROR: result.json not found in {result_dir}")
        sys.exit(1)

    result = load_json(result_path)
    result_schema = load_schema("result.schema.json")
    all_errors.extend(validate_schema(result, result_schema, "result.json"))

    if not all_errors:
        all_errors.extend(check_hard_failures(result, result_dir))
        all_warnings.extend(check_soft_warnings(result))
        all_errors.extend(check_run_id_integrity(result))
        all_errors.extend(check_suite_e(result_dir, result))
        result = compute_derived(result)
        # Write back computed fields
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print("Derived metrics computed and written to result.json")

    # --- Validate env_info.json ---
    env_errors = check_env_info(result_dir)
    all_errors.extend(env_errors)
    if not env_errors:
        env_path = find_env_info(result_dir)
        env = load_json(env_path)
        env_schema = load_schema("env.schema.json")
        all_errors.extend(validate_schema(env, env_schema, "env_info.json"))

    # --- Validate accuracy.json ---
    all_errors.extend(check_accuracy(result_dir, result))

    # --- Report ---
    if all_warnings:
        print("\nWARNINGS (submission will be accepted but flagged):")
        for w in all_warnings:
            print(f"  ⚠  {w}")

    if all_errors:
        print("\nERRORS (fix before submitting PR):")
        for e in all_errors:
            print(f"  ✗  {e}")
        print(f"\n{len(all_errors)} error(s) found. Fix all errors before opening a PR.")
        sys.exit(1)
    else:
        print("\n✓ Validation passed. Ready to submit PR.")
        print(f"  Submission directory: {result_dir}")
        print(f"  Copy to: results/community/{{your_submission_name}}/")
        sys.exit(0)


if __name__ == "__main__":
    main()
