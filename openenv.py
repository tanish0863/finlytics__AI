"""CLI wrapper for OpenEnv commands in the Finlytics submission."""

from __future__ import annotations

import argparse

from openenv.baseline import run_baseline
from openenv.validate import validate_environment


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenEnv CLI wrapper for Finlytics.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("validate", help="Run environment validation.")
    inference_parser = subparsers.add_parser("inference", help="Run the baseline inference script.")
    inference_parser.add_argument(
        "--provider",
        choices=["openai", "heuristic"],
        default="openai",
        help="Action provider to use for inference.",
    )

    args = parser.parse_args()

    if args.command == "validate":
        result = validate_environment()
        print(result)
        return 0

    if args.command == "inference":
        results = run_baseline(provider=args.provider)
        for result in results:
            print(f"{result.task_id}: {result.task_title} -> score={result.score:.4f} steps={result.steps}")
        if results:
            mean_score = sum(result.score for result in results) / len(results)
            print(f"mean_score={mean_score:.4f}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
