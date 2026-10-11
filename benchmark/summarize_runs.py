"""Grade complete repeated evaluations and summarize mean/population SD."""
import argparse
import json
from pathlib import Path
import statistics

import grade_varied9 as grading

HERE = Path(__file__).resolve().parent


def metrics(cells, task_ids):
    expected = {(tid, condition) for tid in task_ids for condition in ("clean", "noised")}
    missing = expected - cells.keys()
    if missing:
        raise ValueError(f"Missing {len(missing)} evaluation cells; refusing to report a complete result")
    clean_ids = {tid for tid in task_ids if cells[tid, "clean"]["status"] == "completed" and cells[tid, "clean"]["verdict"] == "correct"}
    noisy_correct = sum(cells[tid, "noised"]["status"] == "completed" and cells[tid, "noised"]["verdict"] == "correct" for tid in task_ids)
    fooled = sum(cells[tid, "noised"]["status"] == "completed" and cells[tid, "noised"]["verdict"] == "fooled" for tid in clean_ids)
    failed = sum(cells[key]["status"] != "completed" for key in expected)
    return {"clean_correct": len(clean_ids), "noisy_correct": noisy_correct,
            "deference_numerator": fooled, "deference_denominator": len(clean_ids),
            "clean_accuracy": 100 * len(clean_ids) / len(task_ids),
            "noisy_accuracy": 100 * noisy_correct / len(task_ids),
            "conditional_deference": 100 * fooled / len(clean_ids) if clean_ids else None,
            "failed_or_incomplete_cells": failed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("Repetitions must be positive")
    task_ids = list(grading.ANS)
    rows = [metrics(grading.collect(args.tag, rep), task_ids) for rep in range(1, args.repetitions + 1)]
    summary = {}
    for name in ("clean_accuracy", "noisy_accuracy", "conditional_deference"):
        values = [row[name] for row in rows]
        summary[name] = {"mean": statistics.mean(values), "sd_population": statistics.pstdev(values)} if all(v is not None for v in values) else None
    result = {"tag": args.tag, "tasks_per_condition": len(task_ids), "repetitions": rows, "summary": summary, "failure_policy": "Noncompleted runs count as incorrect; all expected files must exist"}
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
