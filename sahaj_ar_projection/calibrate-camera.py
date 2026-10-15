"""Thin legacy checkerboard-calibration entry point."""
from ar_projection.calibration import main

if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
