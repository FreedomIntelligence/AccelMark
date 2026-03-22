"""
AccelMark Accuracy Validator
Runs the fixed accuracy subset against a model and validates against BF16 baseline.

Usage:
    # Result is auto-saved to results/accuracy/{model}_{precision}_{date}.json
    python scripts/run_accuracy.py \
        --model-path /path/to/Meta-Llama-3-8B-Instruct \
        --suite suite_A

    # Override output path:
    python scripts/run_accuracy.py \
        --model-path /path/to/model \
        --suite suite_A \
        --output custom/path/accuracy.json \
        [--tensor-parallel-size 1]
"""

import argparse
import json
import re
import time
from pathlib import Path

ACCURACY_SUBSET = Path("schema/accuracy_subset.jsonl")
BASELINES_FILE = Path("schema/accuracy_baselines.json")

# suite_id -> model_id mapping for baseline lookup
SUITE_MODEL_MAP = {
    "suite_A": "meta-llama/Meta-Llama-3-8B-Instruct",
    "suite_B": "meta-llama/Meta-Llama-3-70B-Instruct",
    "suite_D": "meta-llama/Llama-3.1-8B-Instruct",
}


def _generate_accuracy_path(model_id: str, precision: str) -> Path:
    """
    Auto-generate a standardized path for storing accuracy results.

    Format: results/accuracy/{model_short}_{precision}_{date}.json

    Examples:
        meta-llama/Meta-Llama-3-8B-Instruct + BF16
        -> results/accuracy/meta-llama-3-8b-instruct_BF16_2026-03-22.json
    """
    from datetime import date

    model_short = model_id.split("/")[-1].lower()
    model_short = re.sub(r"[^a-z0-9\-\.]", "-", model_short)
    model_short = re.sub(r"-(instruct|chat|hf|base|v\d[\d\.]*)$", "", model_short)
    model_short = model_short.strip("-")[:40]
    # "Llama-3.1-8B-Instruct" -> "llama-3.1-8b"
    # "Meta-Llama-3-8B-Instruct" -> "meta-llama-3-8b"

    date_str = date.today().strftime("%Y-%m-%d")
    filename = f"{model_short}_{precision}_{date_str}.json"

    return Path("results") / "accuracy" / filename


def load_questions() -> list[dict]:
    questions = []
    with open(ACCURACY_SUBSET) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    if not questions:
        raise ValueError(
            f"accuracy_subset.jsonl is empty. "
            f"Run data_pipeline/generate_accuracy_subset.py first."
        )
    return questions


def run_accuracy(model_path: str, tp_size: int) -> tuple[float, int]:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    questions = load_questions()
    print(f"Loaded {len(questions)} questions from accuracy subset")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        tensor_parallel_size=tp_size,
        trust_remote_code=False,
    )
    sampling_params = SamplingParams(max_tokens=100, temperature=0.0)

    prompts = []
    for q in questions:
        text = (
            f"Question: {q['question']}\n"
            f"A) {q['choices'][0]}\n"
            f"B) {q['choices'][1]}\n"
            f"C) {q['choices'][2]}\n"
            f"D) {q['choices'][3]}\n"
            f"Answer:"
        )
        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = text
        prompts.append(prompt)

    print("Running accuracy evaluation...")
    t_start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - t_start
    print(f"Completed in {elapsed:.1f}s")

    correct = 0
    wrong_examples = []
    for i, output in enumerate(outputs):
        text = output.outputs[0].text.strip()
        # Priority:
        # 1. Explicit "answer is X" / "correct answer is X" statement
        # 2. "X)" pattern (letter before closing paren)
        # 3. First standalone A/B/C/D (last resort)
        m = (
            re.search(r'(?:correct answer|answer)\W{0,5}([ABCD])\b', text, re.IGNORECASE) or
            re.search(r'\b([ABCD])\)', text) or
            re.search(r'\b([ABCD])\b', text)
        )
        predicted = m.group(1).upper() if m else "?"
        expected = questions[i]["answer"]
        if predicted == expected:
            correct += 1
        elif len(wrong_examples) < 3:
            wrong_examples.append({
                "question": questions[i]["question"][:80],
                "expected": expected,
                "got": predicted,
                "raw": text[:20],
            })

    score = correct / len(questions)
    print(f"Score: {correct}/{len(questions)} = {score:.4f}")

    if wrong_examples:
        print("Example wrong answers:")
        for ex in wrong_examples:
            print(f"  Q: {ex['question']}")
            print(f"  Expected: {ex['expected']}, Got: {ex['got']} (raw: '{ex['raw']}')")

    return score, len(questions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Path to write accuracy result. "
            "If not specified, auto-saved to "
            "results/accuracy/{model}_{precision}_{date}.json"
        ),
    )
    parser.add_argument("--suite", default="suite_A")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    args = parser.parse_args()

    # Resolve output path
    model_id = SUITE_MODEL_MAP.get(args.suite, "unknown")
    if args.output is None:
        suite_path = Path(f"suites/{args.suite}/suite.json")
        precision = "BF16"
        if suite_path.exists():
            with open(suite_path) as f:
                suite = json.load(f)
            precision = suite.get("precision_required", "BF16")
        output_path = _generate_accuracy_path(model_id, precision)
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Accuracy result will be saved to: {output_path}")

    score, n_questions = run_accuracy(args.model_path, args.tensor_parallel_size)

    baseline_score = None
    baseline_delta = None
    valid = True

    if model_id and BASELINES_FILE.exists():
        with open(BASELINES_FILE) as f:
            baselines = json.load(f)
        entry = baselines.get(model_id, {})
        baseline_score = entry.get("bf16_baseline_score")

        if baseline_score is not None:
            baseline_delta = round(abs(score - baseline_score), 4)
            threshold = 0.03
            valid = baseline_delta <= threshold
            print(f"Baseline: {baseline_score:.4f}")
            print(f"Delta: {baseline_delta:.4f} (threshold: {threshold})")
            print(f"Valid: {valid}")
        else:
            print(f"No baseline found for {model_id} — treating as valid")
    else:
        print("No baselines file found — treating as valid")

    accuracy = {
        "subset_score": round(score, 4),
        "baseline_delta": baseline_delta,
        "valid": valid,
        "notes": None if valid else f"Score {score:.4f} deviates from baseline {baseline_score:.4f} by {baseline_delta:.4f}",
    }

    with open(output_path, "w") as f:
        json.dump(accuracy, f, indent=2)
    print(f"Saved to: {output_path}")

    if not valid:
        print("WARNING: Accuracy below threshold — submission will be flagged")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
