"""
AccelMark Submission Validator
Validates all files in a submission directory before opening a PR.

Usage:
    python scripts/validate_submission.py --dir results/community/a100x1_llama3-8b_suite-A_2026-03-22
    Exit code 0 = ready to submit. Exit code 1 = fix required.
"""

import argparse
import json
import sys
from pathlib import Path

import jsonschema


SCHEMA_DIR = Path("schema")
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
    if result.get("task", {}).get("num_runs", 0) < 3:
        failures.append("task.num_runs must be >= 3")

    # accuracy.valid for inference suites
    scenario = result.get("task", {}).get("scenario")
    if scenario != "training":
        accuracy = result.get("accuracy", {})
        if not accuracy.get("valid", False) and not accuracy.get("notes"):
            failures.append(
                "accuracy.valid is false but accuracy.notes is empty. "
                "Explain why accuracy is below threshold, or fix the precision configuration."
            )

    # referenced files must exist
    meta = result.get("meta", {})
    scenario_subdirs = ["offline", "online", "interactive", "training"]
    for field in ["reproduce_script", "env_info_file", "log_file"]:
        ref = meta.get(field)
        if not ref:
            continue
        ref_name = Path(ref).name
        if (Path(ref).exists()
                or (result_dir / ref_name).exists()
                or any((result_dir / s / ref_name).exists() for s in scenario_subdirs)):
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
            for row in offline.get("results_by_batch_size", [])
        )
        if not has_power:
            warnings.append(
                "No power data in offline metrics — "
                "tokens_per_sec_per_watt cannot be computed. "
                "Consider adding power measurement to your script."
            )

    return warnings


def compute_derived(result: dict) -> dict:
    """Compute and inject derived metrics."""
    metrics = result.setdefault("metrics", {})
    derived = metrics.setdefault("derived", {})

    # tokens_per_sec_per_watt from offline metrics
    offline = metrics.get("offline")
    if offline:
        rows = offline.get("results_by_batch_size", [])
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

    # Check top-level first
    env_info_path = submission_dir / "env_info.json"
    if env_info_path.exists():
        return errors  # found at top level, OK

    # For suite-level submissions, check inside scenario subdirs
    for scenario in ["offline", "online", "interactive", "training"]:
        scenario_env = submission_dir / scenario / "env_info.json"
        if scenario_env.exists():
            return errors  # found in scenario subdir, OK

    errors.append(
        "env_info.json not found — run scripts/collect_env.py first, "
        "or ensure it exists in a scenario subdirectory"
    )
    return errors


def find_env_info(submission_dir: Path) -> Path | None:
    """Return path to env_info.json, checking top-level then scenario subdirs."""
    top = submission_dir / "env_info.json"
    if top.exists():
        return top
    for scenario in ["offline", "online", "interactive", "training"]:
        candidate = submission_dir / scenario / "env_info.json"
        if candidate.exists():
            return candidate
    return None


def check_accuracy(submission_dir: Path, result: dict) -> list[str]:
    errors = []

    # Check top-level first
    acc_path = submission_dir / "accuracy.json"
    if not acc_path.exists():
        # For suite-level, check scenario subdirs
        for scenario in ["offline", "online", "interactive"]:
            scenario_acc = submission_dir / scenario / "accuracy.json"
            if scenario_acc.exists():
                acc_path = scenario_acc
                break

    if not acc_path.exists():
        errors.append("accuracy.json not found — run scripts/run_accuracy.py first")
        return errors

    with open(acc_path) as f:
        acc = json.load(f)

    if not acc.get("valid"):
        errors.append(
            f"accuracy.valid is false (score={acc.get('subset_score')}) "
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
