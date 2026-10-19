"""Paired MSE/Huber reconstruction comparison against independent image-plane GT."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from reconstruction import ANNOTATION_TO_VIDEO, SOURCE_COMMIT, project_point, validate_pixel_loss
from run_projection import (
    DEFAULT_CHECKPOINT,
    DEFAULT_HOMOGRAPHY,
    DEFAULT_VIDEO,
    HERE,
    assign_paths,
    fit_hit_flights,
    fit_terminal_tail,
    inside_test,
    load_calibration,
    load_player_positions,
    player_references,
    prepare_inputs,
    repo_path,
    reserve_output_files,
    track_shuttle,
    write_tracked_csv,
)


def load_ground_truth(
    csv_path: Path, frame_offset: int, frame_count: int, width: int, height: int
) -> list[dict]:
    required = {"Frame", "Visibility", "X", "Y"}
    with csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Ground-truth CSV is missing columns: {sorted(required - set(reader.fieldnames or []))}")
        source_rows = list(reader)
    rows = []
    frame_ids = []
    for row_number, row in enumerate(source_rows, start=2):
        try:
            gt_frame = int(row["Frame"])
            visibility = int(row["Visibility"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Ground-truth row {row_number} has an invalid frame or visibility") from error
        if gt_frame < 0:
            raise ValueError(f"Ground-truth row {row_number} has a negative Frame ID")
        if visibility not in (0, 1):
            raise ValueError(f"Ground-truth row {row_number} visibility must be 0 or 1")
        clean_frame = gt_frame + int(frame_offset)
        if not 0 <= clean_frame < frame_count:
            raise ValueError(f"Ground-truth row {row_number} maps outside the cleaned video: frame {clean_frame}")
        frame_ids.append(gt_frame)
        if visibility:
            try:
                x_gt, y_gt = float(row["X"]), float(row["Y"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Visible ground-truth row {row_number} has invalid coordinates") from error
            if not np.isfinite([x_gt, y_gt]).all():
                raise ValueError(f"Visible ground-truth row {row_number} has non-finite coordinates")
            x_clean, y_clean = x_gt * ANNOTATION_TO_VIDEO, y_gt * ANNOTATION_TO_VIDEO
            if not (0 <= x_clean < width and 0 <= y_clean < height):
                raise ValueError(f"Visible ground-truth row {row_number} maps outside the cleaned video image")
        else:
            x_gt = y_gt = x_clean = y_clean = None
        rows.append({
            "gt_frame": gt_frame,
            "clean_frame": clean_frame,
            "visibility": visibility,
            "x_gt_px": x_gt,
            "y_gt_px": y_gt,
            "x_clean_px": x_clean,
            "y_clean_px": y_clean,
        })
    if not frame_ids or frame_ids != sorted(set(frame_ids)):
        raise ValueError("Ground-truth Frame IDs must be non-empty, unique, and strictly increasing")
    return rows


def fit_chain(
    hits: list[dict],
    players: list[dict],
    observations: list[tuple[int, int] | None],
    projection: np.ndarray,
    fps: float,
    calibration: dict,
    geometry,
    pixel_loss: str,
    huber_delta_px: float,
) -> dict:
    """Fit to TrackNet pixels and player priors, excluding shuttle endpoint labels."""
    coordinates, phases, segment_ids, _, hit_diagnostics, last_path = fit_hit_flights(
        hits, players, observations, projection, fps, pixel_loss, huber_delta_px
    )
    tail_paths, terminal_diagnostics, _ = fit_terminal_tail(
        calibration, geometry, hits, observations, projection, fps, last_path, pixel_loss, huber_delta_px
    )
    assign_paths(coordinates, phases, segment_ids, tail_paths, tail=True)
    return {
        "coordinates": coordinates,
        "phases": phases,
        "hit_diagnostics": hit_diagnostics,
        "terminal_diagnostics": terminal_diagnostics,
    }


def fit_loss_pair(
    hits: list[dict],
    players: list[dict],
    observations: list[tuple[int, int] | None],
    projection: np.ndarray,
    fps: float,
    calibration: dict,
    geometry,
    huber_delta_px: float,
) -> dict:
    return {
        method: fit_chain(
            hits, players, observations, projection, fps, calibration, geometry, method, huber_delta_px
        )
        for method in ("mse", "huber")
    }


def _reproject(coordinates: np.ndarray, projection: np.ndarray) -> np.ndarray:
    result = np.full((len(coordinates), 2), np.nan, dtype=np.float64)
    for frame, point in enumerate(coordinates):
        if np.isfinite(point).all():
            result[frame] = project_point(projection, point)
    return result


def _metrics(errors: list[float]) -> dict:
    values = np.asarray(errors, dtype=np.float64)
    if not len(values):
        return {"count": 0, "rmse_px": None, "mean_euclidean_px": None, "median_px": None, "p95_px": None, "maximum_px": None}
    return {
        "count": int(len(values)),
        "rmse_px": float(np.sqrt(np.mean(np.square(values)))),
        "mean_euclidean_px": float(np.mean(values)),
        "median_px": float(np.median(values)),
        "p95_px": float(np.percentile(values, 95)),
        "maximum_px": float(np.max(values)),
    }


def evaluate_trajectories(
    ground_truth: list[dict],
    mse_fit: dict,
    huber_fit: dict,
    projection: np.ndarray,
    first_hit_frame: int,
    final_hit_frame: int,
    floor_contact_frame: int | None,
) -> tuple[list[dict], dict, dict]:
    reprojected = {
        "mse": _reproject(mse_fit["coordinates"], projection),
        "huber": _reproject(huber_fit["coordinates"], projection),
    }
    fits = {"mse": mse_fit, "huber": huber_fit}
    rows = []
    for gt in ground_truth:
        clean_frame = gt["clean_frame"]
        if clean_frame < first_hit_frame:
            reason = "pre_hit_padding"
        elif floor_contact_frame is not None and clean_frame == floor_contact_frame:
            reason = "manual_floor_contact_evaluation_boundary"
        elif floor_contact_frame is not None and clean_frame > floor_contact_frame:
            reason = "post_contact"
        elif not gt["visibility"]:
            reason = "invisible_ground_truth"
        else:
            reason = "eligible"
        eligible = reason == "eligible"
        gt_point = np.asarray([gt["x_clean_px"], gt["y_clean_px"]], dtype=np.float64) if gt["visibility"] else None
        errors = {}
        positions = {}
        for method in ("mse", "huber"):
            position = reprojected[method][clean_frame]
            positions[method] = position if np.isfinite(position).all() else None
            errors[method] = (
                float(np.linalg.norm(position - gt_point))
                if eligible and positions[method] is not None else None
            )
        paired = eligible and errors["mse"] is not None and errors["huber"] is not None
        rows.append({
            **gt,
            "mse_reprojected_x_px": None if positions["mse"] is None else float(positions["mse"][0]),
            "mse_reprojected_y_px": None if positions["mse"] is None else float(positions["mse"][1]),
            "mse_error_px": errors["mse"],
            "mse_flight_identity": fits["mse"]["phases"][clean_frame],
            "huber_reprojected_x_px": None if positions["huber"] is None else float(positions["huber"][0]),
            "huber_reprojected_y_px": None if positions["huber"] is None else float(positions["huber"][1]),
            "huber_error_px": errors["huber"],
            "huber_flight_identity": fits["huber"]["phases"][clean_frame],
            "eligibility_reason": reason,
            "comparison_eligible": eligible,
            "mse_scored": errors["mse"] is not None,
            "huber_scored": errors["huber"] is not None,
            "paired_comparison_eligible": bool(paired),
        })

    eligible_rows = [row for row in rows if row["comparison_eligible"]]

    def summarize(candidate_rows: list[dict]) -> dict:
        paired_rows = [row for row in candidate_rows if row["paired_comparison_eligible"]]
        individual = {
            method: _metrics([row[f"{method}_error_px"] for row in candidate_rows if row[f"{method}_error_px"] is not None])
            for method in ("mse", "huber")
        }
        paired = {method: _metrics([row[f"{method}_error_px"] for row in paired_rows]) for method in ("mse", "huber")}
        count = len(candidate_rows)
        return {
            "coverage": {
                "eligible_visible_gt_frames": count,
                "mse_available_frames": individual["mse"]["count"],
                "huber_available_frames": individual["huber"]["count"],
                "paired_available_frames": len(paired_rows),
                "mse_fraction": individual["mse"]["count"] / count if count else None,
                "huber_fraction": individual["huber"]["count"] / count if count else None,
                "paired_fraction": len(paired_rows) / count if count else None,
                "mse_missing_estimate_clean_frames": [row["clean_frame"] for row in candidate_rows if not row["mse_scored"]],
                "huber_missing_estimate_clean_frames": [row["clean_frame"] for row in candidate_rows if not row["huber_scored"]],
            },
            **{
                method: {"individual_metrics": individual[method], "paired_metrics": paired[method]}
                for method in ("mse", "huber")
            },
        }

    baseline_rows = [row for row in eligible_rows if row["clean_frame"] <= final_hit_frame]
    final_flight_rows = [row for row in eligible_rows if row["clean_frame"] > final_hit_frame]
    overall_summary = summarize(eligible_rows)
    coverage = overall_summary["coverage"]

    per_flight = {}
    flight_ids = sorted({
        row[f"{method}_flight_identity"]
        for row in eligible_rows
        for method in ("mse", "huber")
        if row[f"{method}_flight_identity"]
    })
    for flight_id in flight_ids:
        per_flight[flight_id] = {}
        paired_flight_rows = [
            row for row in eligible_rows
            if row["paired_comparison_eligible"]
            and row["mse_flight_identity"] == flight_id
            and row["huber_flight_identity"] == flight_id
        ]
        for method in ("mse", "huber"):
            available_rows = [
                row for row in eligible_rows
                if row[f"{method}_flight_identity"] == flight_id and row[f"{method}_error_px"] is not None
            ]
            per_flight[flight_id][method] = {
                "individual_metrics": _metrics([row[f"{method}_error_px"] for row in available_rows]),
                "paired_metrics": _metrics([row[f"{method}_error_px"] for row in paired_flight_rows]),
            }
    metrics = {
        "overall": overall_summary,
        "hit_to_hit_baseline": summarize(baseline_rows),
        "final_flight_extension": summarize(final_flight_rows),
        "per_flight": per_flight,
    }
    return rows, coverage, metrics


def _matplotlib_pyplot():
    try:
        import matplotlib
    except ImportError as error:
        raise RuntimeError("Matplotlib is required to write the requested PNG and is not installed") from error
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def write_error_plot(path: Path, rows: list[dict], plt) -> None:
    frames = [row["clean_frame"] for row in rows]
    for method, label in (("mse", "MSE"), ("huber", "Huber")):
        errors = [row[f"{method}_error_px"] if row["comparison_eligible"] else None for row in rows]
        plt.plot(frames, [np.nan if error is None else error for error in errors], label=label)
    plt.xlabel("Cleaned video frame")
    plt.ylabel("Euclidean error (cleaned-video pixels)")
    plt.title("Trajectory error against visible shuttle ground truth")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _new_output_dir(value: Path | None) -> Path:
    if value is None:
        value = HERE / f"korea_loss_comparison_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    path = inside_test(repo_path(value))
    path.mkdir(parents=True, exist_ok=False)
    return path


def run_comparison(args) -> dict:
    _, delta = validate_pixel_loss("huber", args.huber_delta_px)
    player_csv = getattr(args, "player_positions_csv", None)
    video, hits_csv, homography_csv, metadata, hits, geometry = prepare_inputs(
        args.video, args.rally_csv, args.homography_csv, require_player_positions=player_csv is None
    )
    player_tracks, player_provenance = load_player_positions(player_csv, video, metadata, hits, geometry)
    players = player_references(hits, geometry, player_tracks)
    calibration_path = inside_test(repo_path(args.calibration))
    if not calibration_path.is_file():
        raise FileNotFoundError(f"Calibration JSON not found: {calibration_path}")
    calibration = load_calibration(calibration_path, geometry, metadata)
    floor_event = calibration["terminal_events"]["floor_contact"]
    if floor_event.get("status") != "manual":
        raise ValueError("The paired comparison requires a manually selected first floor-contact frame and pixel")
    floor_frame = int(floor_event["frame"])
    if floor_frame <= hits[-1]["clip_frame_num"]:
        raise ValueError("The manually selected floor-contact frame must follow the final annotated hit")
    gt_path = repo_path(args.gt_csv)
    if not gt_path.is_file():
        raise FileNotFoundError(gt_path)
    ground_truth = load_ground_truth(
        gt_path, args.gt_frame_offset, metadata["frame_count"], metadata["width"], metadata["height"]
    )
    plt = _matplotlib_pyplot()
    output_dir = _new_output_dir(args.output_dir)
    stem = video.stem
    output_paths = {
        "predictions_csv": output_dir / f"{stem}_tracked_pixels.csv",
        "comparison_csv": output_dir / f"{stem}_loss_comparison.csv",
        "diagnostics_json": output_dir / f"{stem}_loss_comparison.json",
        "error_plot_png": output_dir / f"{stem}_loss_comparison.png",
    }
    created = reserve_output_files(list(output_paths.values()))
    complete = False
    try:
        observations, device = track_shuttle(video, metadata["frame_count"], repo_path(args.checkpoint))
        projection = np.asarray(calibration["camera_calibration"]["projection_matrix"], dtype=np.float64)
        # Ground truth is deliberately not passed into either fitting call.
        fits = fit_loss_pair(hits, players, observations, projection, metadata["fps"], calibration, geometry, delta)
        mse_fit, huber_fit = fits["mse"], fits["huber"]
        rows, coverage, metrics = evaluate_trajectories(
            ground_truth, mse_fit, huber_fit, projection,
            hits[0]["clip_frame_num"], hits[-1]["clip_frame_num"], floor_frame,
        )
        write_tracked_csv(output_paths["predictions_csv"], observations)
        with output_paths["comparison_csv"].open("w", newline="", encoding="utf-8") as stream:
            fields = [
                "gt_frame", "clean_frame", "visibility", "x_gt_px", "y_gt_px", "x_clean_px", "y_clean_px",
                "mse_reprojected_x_px", "mse_reprojected_y_px", "mse_error_px", "mse_flight_identity",
                "huber_reprojected_x_px", "huber_reprojected_y_px", "huber_error_px", "huber_flight_identity",
                "eligibility_reason", "comparison_eligible", "mse_scored", "huber_scored", "paired_comparison_eligible",
            ]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        write_error_plot(output_paths["error_plot_png"], rows, plt)
        fit_unavailable = {
            method: [
                detail for detail in fit["hit_diagnostics"] + fit["terminal_diagnostics"]["phases"]
                if detail.get("status") not in ("available", "failed")
            ]
            for method, fit in (("mse", mse_fit), ("huber", huber_fit))
        }
        fit_failures = {
            method: [
                detail for detail in fit["hit_diagnostics"] + fit["terminal_diagnostics"]["phases"]
                if detail.get("status") == "failed"
            ]
            for method, fit in (("mse", mse_fit), ("huber", huber_fit))
        }
        diagnostics = {
            "configuration": {
                "video": str(video),
                "rally_csv": str(hits_csv),
                "homography_csv": str(homography_csv),
                "homography_row_id": geometry.row_id,
                "calibration_json": str(calibration_path),
                "checkpoint": str(repo_path(args.checkpoint)),
                "input_metadata": metadata,
                "annotated_hits": len(hits),
                "huber_delta_px": delta,
                "pixel_losses": ["mse", "huber"],
                "monotrack_source_commit": SOURCE_COMMIT,
                "frame_timing": {"fps": metadata["fps"], "frame_time": "frame / fps"},
                "gt_csv": str(gt_path),
                "gt_frame_offset": int(args.gt_frame_offset),
                "gt_coordinate_scale_to_clean_video": ANNOTATION_TO_VIDEO,
                "ground_truth_rows": len(ground_truth),
                "visible_ground_truth_rows": sum(row["visibility"] for row in ground_truth),
                "evaluation_window": {
                    "first_scored_clean_frame": hits[0]["clip_frame_num"],
                    "manual_floor_contact_frame_excluded": floor_frame,
                    "frames_at_or_after_floor_contact_excluded": floor_frame is not None,
                },
            },
            "provenance": {
                "player_positions": player_provenance,
                "tracknet_run_count": 1,
                "ground_truth_used_only_for_evaluation": True,
                "ground_truth_used_for_fitting_or_initialization": False,
                "manual_calibration_shared_by_both_losses": True,
                "prediction_csv": str(output_paths["predictions_csv"]),
                "fit_loss_objective_is_not_an_accuracy_metric": True,
            },
            "tracker": {
                "script": str(HERE.parent / "TrackNetv2-pretrained" / "track_video.py"),
                "checkpoint": str(repo_path(args.checkpoint)),
                "device": device,
            },
            "manual_calibration": {
                "camera_calibration": calibration["camera_calibration"],
                "terminal_events": calibration["terminal_events"],
            },
            "outputs": {name: str(path) for name, path in output_paths.items()},
            "fit_residuals_against_tracknet": {
                "interpretation": "Per-flight residuals and optimizer objectives are fit diagnostics against TrackNet pixels; they are not ground-truth accuracy metrics.",
                "mse": {"hit_to_hit": mse_fit["hit_diagnostics"], "terminal": mse_fit["terminal_diagnostics"]},
                "huber": {"hit_to_hit": huber_fit["hit_diagnostics"], "terminal": huber_fit["terminal_diagnostics"]},
            },
            "fit_failures": fit_failures,
            "unavailable_flights": fit_unavailable,
            "coverage": coverage,
            "evaluation_errors_against_ground_truth": metrics,
        }
        with output_paths["diagnostics_json"].open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(_json_safe(diagnostics), stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        complete = True
    finally:
        if not complete:
            for path in created:
                path.unlink(missing_ok=True)
            try:
                output_dir.rmdir()
            except OSError:
                pass
    print(f"TrackNet predictions: {output_paths['predictions_csv']}")
    print(f"Per-frame comparison: {output_paths['comparison_csv']}")
    print(f"Metrics and diagnostics: {output_paths['diagnostics_json']}")
    print(f"Error plot: {output_paths['error_plot_png']}")
    for method in ("mse", "huber"):
        result = metrics["overall"][method]
        rmse = result["paired_metrics"]["rmse_px"]
        print(f"{method.upper()} paired RMSE: {rmse}; paired frames: {coverage['paired_available_frames']}")
        final_rmse = metrics["final_flight_extension"][method]["paired_metrics"]["rmse_px"]
        print(f"{method.upper()} final-flight paired RMSE: {final_rmse}")
    print(f"Coverage: {coverage}")
    print(f"Fit failures: {fit_failures}")
    print(f"Unavailable flight phases: {fit_unavailable}")
    return diagnostics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--rally-csv", type=Path, default=None)
    parser.add_argument("--homography-csv", type=Path, default=DEFAULT_HOMOGRAPHY)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="New output directory inside monotrack_test; defaults to a timestamped path")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--player-positions-csv", type=Path, help="Saved smoothed keypoint CSV with review metadata; otherwise use annotated player positions")
    parser.add_argument("--gt-csv", type=Path, required=True)
    parser.add_argument("--gt-frame-offset", type=int, required=True, help="Cleaned frame = GT frame + offset (Korea default: 28)")
    parser.add_argument("--huber-delta-px", type=float, default=10.0, help="Huber threshold in cleaned-video pixels (default: 10)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.rally_csv is None:
        args.rally_csv = args.video.with_suffix(".csv") if args.video != DEFAULT_VIDEO else DEFAULT_VIDEO.with_suffix(".csv")
    try:
        run_comparison(args)
    except Exception as error:
        parser.exit(1, f"compare_losses: error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
