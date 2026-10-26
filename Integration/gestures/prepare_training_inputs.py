"""Build or inspect the shared, content-addressed (N,T,43) training cache."""

from __future__ import annotations

import argparse
from pathlib import Path

import training_data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="explicit training_config.json path")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="read-only preflight; this is the default")
    mode.add_argument("--build", action="store_true", help="build the shared memmap from verified raw landmark caches")
    args = parser.parse_args(argv)
    try:
        if args.build:
            result = training_data.build_feature_cache(args.config)
            print(f"Shared feature cache ready: {result}")
            return 0
        result = training_data.inspect_inputs(args.config, require_features=False)
        print(f"Sequence length: {result['config']['_sequence_length']}")
        print(f"Retained labeled clips: {len(result['source'].get('retained_ids', []))}")
        if result["blockers"]:
            print("Blockers:")
            for blocker in result["blockers"]:
                print(f"  - {blocker}")
            return 2
        print("Input source ready; run with --build to create the shared cache.")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
