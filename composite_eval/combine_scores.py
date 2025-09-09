import argparse
import csv
import json
import os
from typing import Dict, Tuple


def _parse_float(val: str) -> float | None:
    try:
        return float(val)
    except Exception:
        return None


def _load_weights(weights_json: str | None, weight_overrides: list[str]) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    if weights_json:
        with open(weights_json, "r") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("weights_json must contain an object mapping benchmark -> weight")
            for k, v in data.items():
                weights[str(k)] = float(v)
    for w in weight_overrides or []:
        # Accept forms like "MMLU=2", "MATH:1.5"
        sep = "=" if "=" in w else ":"
        if sep not in w:
            raise ValueError(f"Invalid --weight '{w}'. Use BENCH=WEIGHT or BENCH:WEIGHT")
        name, val = w.split(sep, 1)
        weights[name.strip()] = float(val.strip())
    return weights


def combine(csv_path: str, weights: Dict[str, float], verbose: bool) -> Tuple[float, float, float, float]:
    total_weight = 0.0
    current_weighted = 0.0
    next_weighted = 0.0
    used = 0
    skipped = 0

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bench = row.get("benchmark") or row.get("Benchmark")
            if not bench:
                continue
            cur = _parse_float(row.get("current_model", ""))
            nxt = _parse_float(row.get("next_gen_model", ""))
            if cur is None or nxt is None:
                skipped += 1
                continue
            w = float(weights.get(bench, 1.0))
            current_weighted += w * cur
            next_weighted += w * nxt
            total_weight += w
            used += 1
            if verbose:
                diff = nxt - cur
                print(f"{bench:>12s} | weight={w:.3f} | current={cur:.3f} | next={nxt:.3f} | diff={diff:+.3f}")

    if total_weight == 0:
        raise ValueError("No valid rows to combine (total weight is zero).")

    current_avg = current_weighted / total_weight
    next_avg = next_weighted / total_weight
    if verbose:
        print(f"\nUsed {used} rows, skipped {skipped} rows. Total weight={total_weight:.3f}")
    return current_avg, next_avg, current_weighted, next_weighted


def main():
    parser = argparse.ArgumentParser(description="Combine benchmark scores from CSV and decide which model is better.")
    parser.add_argument(
        "--csv",
        type=str,
        default="composite_eval/results.csv",
        help="Path to results CSV from run_individual_evals.py",
    )
    parser.add_argument(
        "--weights_json",
        type=str,
        default="composite_eval/weights.json",
        help="Optional JSON file mapping benchmark -> weight",
    )
    parser.add_argument(
        "--weight",
        action="append",
        help="Override or add a weight inline (e.g., --weight MMLU=2 --weight MATH:1.5)",
    )
    parser.add_argument("--details", action="store_true", help="Print per-benchmark details")
    args = parser.parse_args()

    weights = _load_weights(args.weights_json, args.weight or [])

    current_avg, next_avg, current_sum, next_sum = combine(args.csv, weights, args.details)

    print("\nWeighted results:")
    print(f"  current_model (avg): {current_avg:.4f}")
    print(f"  next_gen_model (avg): {next_avg:.4f}")

    if next_avg > current_avg:
        print(f"\nWinner: next_gen_model by {(next_avg - current_avg):.4f}")
    elif next_avg < current_avg:
        print(f"\nWinner: current_model by {(current_avg - next_avg):.4f}")
    else:
        print("\nTie: both models have the same weighted average score")


if __name__ == "__main__":
    main()


