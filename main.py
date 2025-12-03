"""
Semantic Bundle Training - Main entry point.

This package implements experiments for training LLMs with overlapping
low-rank bundle adapters, where different loss types (SSL, RL components,
error signals) flow through different neural circuits.

Quick start:
    # Run smoke test on laptop
    uv run python -m src.smoke_test --quick

    # Full smoke test
    uv run python -m src.smoke_test

    # Training (local)
    uv run python -m src.training --model gpt2 --config ssl_rl_separate --epochs 1

    # Training (Modal - cloud)
    modal run modal_train.py --config ssl_rl_separate --model gpt2-medium
"""

import sys


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nAvailable commands:")
        print("  smoke-test  - Run smoke test on small models")
        print("  train       - Run training")
        print("  eval        - Run evaluation")
        print("\nExamples:")
        print("  uv run python -m src.smoke_test --quick")
        print("  uv run python -m src.training --model gpt2 --config ssl_rl_separate")
        return

    command = sys.argv[1]

    if command == "smoke-test":
        from src.smoke_test import main as smoke_main
        sys.argv = sys.argv[1:]  # Shift args
        smoke_main()
    elif command == "train":
        from src.training import main as train_main
        sys.argv = sys.argv[1:]
        train_main()
    else:
        print(f"Unknown command: {command}")
        print("Available: smoke-test, train")


if __name__ == "__main__":
    main()
