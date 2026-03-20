"""
AccelMark Accuracy Checker
Runs the 100-question MMLU accuracy subset on a given model/platform script.

Usage:
    python scripts/run_accuracy.py \
        --suite suite_A \
        --script scripts/nvidia/run_vllm.py \
        --output my_submission/accuracy.json
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path


SCHEMA_DIR = Path("schema")


def load_accuracy_subset() -> list[dict]:
    subset_path = SCHEMA_DIR / "accuracy_subset.jsonl"
    if not subset_path.exists():
        print(f"ERROR: {subset_path} not found.")
        sys.exit(1)
    questions = []
    with open(subset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def load_baselines() -> dict:
    baselines_path = SCHEMA_DIR / "accuracy_baselines.json"
    if not baselines_path.exists():
        return {}
    with open(baselines_path) as f:
        return json.load(f)


def load_script_module(script_path: str):
    spec = importlib.util.spec_from_file_location("platform_script", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def format_question(q: dict) -> str:
    choices = q["choices"]
    return (
        f"Question: {q['question']}\n"
        f"A) {choices[0]}\n"
        f"B) {choices[1]}\n"
        f"C) {choices[2]}\n"
        f"D) {choices[3]}\n"
        f"Answer:"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--script", required=True, help="Path to platform benchmark script")
    parser.add_argument("--output", required=True, help="Output accuracy.json path")
    parser.add_argument("--model-id", help="Override model ID (default: from suite.json)")
    parser.add_argument("--model-revision", help="Override model revision")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    args = parser.parse_args()

    # Load suite to get model info
    suite_path = Path(f"suites/{args.suite}/suite.json")
    with open(suite_path) as f:
        suite = json.load(f)

    model_id = args.model_id or suite["model_id"]
    model_revision = args.model_revision or suite["model_revision"]

    # Load platform script
    print(f"Loading platform script: {args.script}")
    module = load_script_module(args.script)

    # Load model
    print(f"Loading model {model_id}...")
    module.load_model(model_id, model_revision, args.tensor_parallel_size)
    print("Model loaded.")

    # Load accuracy subset
    questions = load_accuracy_subset()
    print(f"Running accuracy on {len(questions)} questions...")

    correct = 0
    for i, q in enumerate(questions):
        prompt = format_question(q)
        results = module.inference_fn([prompt])
        r = results[0]

        if not r.success:
            print(f"  Question {i}: FAILED ({r.error})")
            continue

        # The model's output should start with A, B, C, or D
        output_text = getattr(r, "output_text", "").strip().upper()
        predicted = output_text[0] if output_text and output_text[0] in "ABCD" else None

        if predicted == q["answer"]:
            correct += 1

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(questions)}, correct so far: {correct}")

    subset_score = correct / len(questions) if questions else 0.0
    print(f"\nAccuracy: {correct}/{len(questions)} = {subset_score:.4f}")

    # Compare against baseline
    baselines = load_baselines()
    baseline_info = baselines.get(model_id, {})
    baseline_score = baseline_info.get("bf16_baseline_score")
    threshold = suite.get("accuracy_threshold_delta", 0.03)

    baseline_delta = None
    valid = False
    notes = None

    if baseline_score is not None:
        baseline_delta = round(abs(subset_score - baseline_score), 4)
        valid = baseline_delta <= threshold
        if not valid:
            notes = (
                f"Score {subset_score:.4f} deviates from baseline {baseline_score:.4f} "
                f"by {baseline_delta:.4f} (threshold: {threshold})"
            )
    else:
        notes = f"No baseline available for {model_id}. Cannot determine validity."

    accuracy = {
        "subset_score": round(subset_score, 4),
        "baseline_delta": baseline_delta,
        "valid": valid,
        "notes": notes,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(accuracy, f, indent=2)
    print(f"Accuracy results written to {out_path}")

    if valid:
        print("✓ Accuracy check passed.")
    else:
        print("✗ Accuracy check failed. See notes in accuracy.json.")


if __name__ == "__main__":
    main()
