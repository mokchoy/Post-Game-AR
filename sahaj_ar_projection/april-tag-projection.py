"""Thin legacy entry point for the bounded AR pipeline."""
from ar_projection.cli import main

if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
