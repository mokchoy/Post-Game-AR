"""Static comparison figures and a Markdown report, from exported rows only."""
from __future__ import annotations

import os
import textwrap

import numpy as np

METHODS = ("detector", "keypoint")
VARIANTS = ("raw", "smoothed")
COLORS = {"detector": "#0072B2", "keypoint": "#D55E00"}
LABELS = {"detector": "Faster R-CNN", "keypoint": "YOLO heatmap"}


def eligible_cdf(errors, field, limit):
    values = sorted(row[field] for row in errors if row[field] is not None)
    return ([0, *values, limit],
            [0, *[i / len(errors) for i in range(1, len(values) + 1)],
             len(values) / len(errors) if errors else 0])


def write_graphs_and_report(output, rallies, records, timings, results):
    os.environ.setdefault("MPLCONFIGDIR", str(output.parent / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import runner as detector

    graphs = output / "graphs"
    graphs.mkdir(exist_ok=False)
    saved = []

    def save(fig, name, caption):
        for extension in ("png", "pdf"):
            fig.savefig(graphs / f"{name}.{extension}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append((name, caption))

    def label(method, variant):
        return f"{LABELS[method]} / {variant}"

    def legend(fig, ax):
        handles, names = ax.get_legend_handles_labels()
        fig.legend(handles, names, loc="lower center", ncol=2, fontsize=9)
        fig.tight_layout(rect=(0, .07, 1, .94))

    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
        for ax, field, units in zip(axes, ("error_px", "error_m"), ("native pixels", "metres")):
            all_values = [row[field] for entry in records.values() for row in entry["errors"]
                          if row[field] is not None]
            limit = max(all_values, default=1) * 1.03 or 1
            for method in METHODS:
                for variant in VARIANTS:
                    errors = [row for rally in rallies for row in records[method, variant, rally["id"]]["errors"]]
                    x, y = eligible_cdf(errors, field, limit)
                    ax.step(x, np.asarray(y) * 100, where="post", color=COLORS[method],
                            linestyle="--" if variant == "raw" else "-", label=label(method, variant))
            ax.set(xlabel=f"Error ({units})", ylabel="Eligible references within error (%)", ylim=(0, 102), xlim=(0, limit))
            ax.grid(alpha=.2)
        fig.suptitle("Error distributions — missing predictions stay in the denominator")
        legend(fig, axes[0])
        save(fig, "error_distributions", "Pooled error distributions; a curve's ceiling is its annotation-time coverage.")

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        scopes = [(None, "Pooled"), *[(r["id"], f"Rally {r['key'][-1]} · {r['id']}") for r in rallies]]
        for ax, (rally_id, title) in zip(axes.flat, scopes):
            for index, (method, variant) in enumerate((m, v) for m in METHODS for v in VARIANTS):
                summary = results["methods"][method][variant]
                metrics = (summary if rally_id is None else summary["per_rally"][rally_id])["combined"]
                values = [metrics["coverage_percentage"], 100 * metrics["position_coverage"],
                          metrics["within_0_5_m"]["percentage"], metrics["within_1_m"]["percentage"]]
                ax.bar(np.arange(4) + (index - 1.5) * .19, values, width=.18, color=COLORS[method],
                       alpha=.5 if variant == "raw" else 1, hatch="//" if variant == "raw" else None,
                       label=label(method, variant))
            ax.set(title=title, ylabel="Percent", ylim=(0, 105))
            ax.set_xticks(range(4), ["Annotated\ncoverage", "All-frame\ncoverage", "Within\n0.5 m", "Within\n1 m"])
            ax.grid(axis="y", alpha=.2)
        fig.suptitle("Coverage and distance success — raw and smoothed")
        legend(fig, axes.flat[0])
        save(fig, "coverage_success", "Smoothed coverage includes interpolated/held positions. Missing references fail distance-success tests.")

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        names = [LABELS[m] for m in METHODS]
        values = [[r["milliseconds"] for r in timings if r["method"] == m] for m in METHODS]
        boxes = axes[0].boxplot(values, tick_labels=names, showfliers=False, patch_artist=True)
        for box, method in zip(boxes["boxes"], METHODS):
            box.set_facecolor(COLORS[method])
        axes[0].set(ylabel="Milliseconds/frame", title="Inference pipeline latency\nbox: quartiles, line: median")
        for i, method in enumerate(METHODS):
            timing = results["methods"][method]["timing"]
            axes[0].scatter(i + 1, timing["p95_ms"], marker="D", color="black", s=24,
                            label="p95" if i == 0 else None)
        axes[0].legend()
        axes[1].bar(names, [results["methods"][m]["timing"]["fps"] for m in METHODS], color=[COLORS[m] for m in METHODS])
        axes[1].set(ylabel="Frames/second", title="Inference pipeline throughput")
        axes[2].bar(names, [1000 * results["methods"][m]["smoothing_seconds"] /
                           results["methods"][m]["timing"]["frames"] for m in METHODS], color=[COLORS[m] for m in METHODS])
        axes[2].set(ylabel="Milliseconds/source frame", title="Offline smoothing computation\nseparate from inference")
        for ax in axes:
            ax.tick_params(axis="x", labelsize=9)
            ax.grid(axis="y", alpha=.2)
        fig.suptitle("Batch 1 · FP32 · synchronized timing · no video I/O")
        fig.tight_layout(rect=(0, 0, 1, .92))
        save(fig, "runtime", "Inference includes preprocessing, transfers, decoding predictions and court projection. Boxes hide outliers; p95 is marked.")

        for rally in rallies:
            title = textwrap.fill(f"{rally['key'][1].replace('_', ' ')} — rally {rally['key'][-1]}", 85)
            fig, axes = plt.subplots(2, 2, figsize=(10, 11))
            for column, method in enumerate(METHODS):
                for row_index, player in enumerate(("A", "B")):
                    ax = axes[row_index, column]
                    for start, end in detector.court_world_lines():
                        ax.plot([start[0], end[0]], [start[1], end[1]], color=".65", linewidth=.8)
                    outside = 0
                    for variant in VARIANTS:
                        positions = [r for r in records[method, variant, rally["id"]]["positions"] if r["player"] == player]
                        points = np.asarray([[r["x_court_m"], r["y_court_m"]] if r["available"] else [np.nan, np.nan] for r in positions])
                        ax.plot(points[:, 0], points[:, 1], color=COLORS[method],
                                alpha=.4 if variant == "raw" else 1, linewidth=.8 if variant == "raw" else 1.5,
                                linestyle="--" if variant == "raw" else "-", label=variant)
                        available = np.isfinite(points).all(axis=1)
                        outside += int(np.sum(available & ((points[:, 0] < -3) | (points[:, 0] > detector.COURT_WIDTH_M + 3)
                                       | (points[:, 1] < -3) | (points[:, 1] > detector.COURT_LENGTH_M + 3))))
                        filled = np.asarray([r.get("gap_filled", False) for r in positions])
                        if np.any(filled):
                            ax.scatter(points[filled, 0], points[filled, 1], marker="^", s=15, facecolors="none",
                                       edgecolors=COLORS[method], label="interpolated/held", zorder=4)
                    refs = np.asarray([refs[player]["world"] for refs in rally["annotations"].values()])
                    ax.scatter(refs[:, 0], refs[:, 1], marker="x", color="black", s=25,
                               label="annotated references", zorder=5)
                    ax.set(title=f"{LABELS[method]} · {player} ({rally['halves'][player]})",
                           xlabel="Court x (m)", ylabel="Court y (m)", xlim=(-3, detector.COURT_WIDTH_M + 3),
                           ylim=(detector.COURT_LENGTH_M + 3, -3), aspect="equal")
                    if outside:
                        ax.text(.02, .02, f"{outside} raw/smoothed positions outside view", transform=ax.transAxes, fontsize=8)
                    ax.legend(fontsize=8, loc="upper right")
            fig.suptitle(title + "\nCourt paths; crosses are sparse references, not a ground-truth path", fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, .92))
            save(fig, f"trajectories_{rally['id']}", f"{title.replace(chr(10), ' ')}. Triangles identify filled points; court view includes a 3 m margin.")

            fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, sharey=True)
            for ax, player in zip(axes, ("A", "B")):
                any_available = False
                for method in METHODS:
                    for variant in VARIANTS:
                        errors = [r for r in records[method, variant, rally["id"]]["errors"] if r["player"] == player]
                        times = np.asarray([r["clip_time_seconds"] for r in errors])
                        errors_m = np.asarray([r["error_m"] if r["error_m"] is not None else np.nan for r in errors])
                        any_available |= bool(np.isfinite(errors_m).any())
                        ax.plot(times, errors_m, marker=".", linewidth=1, color=COLORS[method],
                                linestyle="--" if variant == "raw" else "-", label=label(method, variant))
                        missing = ~np.isfinite(errors_m)
                        if missing.any():
                            offset = .03 + .04 * (2 * METHODS.index(method) + VARIANTS.index(variant))
                            ax.scatter(times[missing], np.full(missing.sum(), offset), transform=ax.get_xaxis_transform(),
                                       marker="x", color=COLORS[method], s=35, label=f"missing: {label(method, variant)}")
                if not any_available:
                    ax.text(.5, .5, "No available annotated predictions", ha="center", transform=ax.transAxes)
                    ax.set_ylim(0, 1)
                ax.set(title=f"Player {player} ({rally['halves'][player]})", ylabel="Metre error", ylim=(0, None))
                # All-NaN lines do not autoscale x, including the missing-marker transform.
                ax.set_xlim(0, max((rally["frames"] - 1) / rally["fps"], 1 / rally["fps"]))
                ax.grid(alpha=.2)
                ax.legend(fontsize=8, ncol=2)
            axes[-1].set_xlabel("Clip time (seconds); errors exist only at annotated frames")
            fig.suptitle(title, fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, .92))
            save(fig, f"errors_time_{rally['id']}", "Annotation-time errors only. Crosses along the lower edge indicate missing predictions; line gaps are retained.")

    def fmt(value):
        return "—" if value is None else f"{value:.3f}"

    lines = ["# Matched player-localisation comparison", "", results["diagnostic_scope"], "",
             "This compares an unadapted COCO detector with a badminton-trained heatmap model. "
             "It does not isolate architecture superiority or establish precise foot-position accuracy.", "",
             "Accuracy reuses the validated saved keypoint positions. Runtime is newly measured for both models. "
             "Error statistics condition on available predictions; coverage and success use every eligible reference. "
             "Continuous smoothing uses future frames and may fill long gaps or hold endpoints.", "",
             "## Clips", ""]
    lines.extend(f"- `{r['id']}`: {r['key'][1]}, rally {r['key'][-1]}; {r['frames']} frames, "
                 f"{2*len(r['annotations'])} references; A {r['halves']['A']}, B {r['halves']['B']}." for r in rallies)
    scopes = []
    for method in METHODS:
        for variant in VARIANTS:
            summary = results["methods"][method][variant]
            scopes.append((label(method, variant), "Pooled", summary["combined"]))
            scopes.extend((label(method, variant), f"Pooled player {p}", summary["per_player"][p]) for p in ("A", "B"))
            for rally in rallies:
                subset = summary["per_rally"][rally["id"]]
                scopes.append((label(method, variant), rally["id"], subset["combined"]))
                scopes.extend((label(method, variant), f"{rally['id']} player {p}", subset["per_player"][p]) for p in ("A", "B"))
    for field, units in (("metre_error", "metres"), ("pixel_error", "native pixels")):
        lines += ["", f"## Position error ({units})", "", "| Approach | Scope | Available | Mean | Median | p95 | RMSE |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for approach, scope, metrics in scopes:
            stats = metrics[field]
            lines.append(f"| {approach} | {scope} | {stats['count']} | " + " | ".join(
                fmt(stats[key]) for key in ("mean", "median", "p95", "euclidean_rmse")) + " |")
    lines += ["", "## Coverage and distance success", "",
              "| Approach | Scope | References | Annotated coverage % | All-frame coverage % | Within 0.5 m % | Within 1 m % | Filled positions |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for approach, scope, metrics in scopes:
        lines.append(f"| {approach} | {scope} | {metrics['eligible_references']} | {fmt(metrics['coverage_percentage'])} | "
                     f"{fmt(100*metrics['position_coverage'])} | {fmt(metrics['within_0_5_m']['percentage'])} | "
                     f"{fmt(metrics['within_1_m']['percentage'])} | {metrics['filled_points']} |")
    lines += ["", "## Runtime", "", "Batch 1, FP32, 10 warm-up calls per clip, same device, models run sequentially. "
              "Synchronized timing includes preprocessing, device transfers, prediction decoding/selection and projection. "
              "It excludes model loading, video decoding, rendering, encoding and smoothing. "
              "Smoothing cost includes filtering and inverse projection. These are offline throughput measurements, not live latency guarantees.", "",
              "| Approach | Scope | Frames | Mean ms | Median ms | p95 ms | FPS |", "|---|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        timing = results["methods"][method]["timing"]
        for scope, metric in [("Pooled", timing), *timing.get("per_rally", {}).items()]:
            lines.append(f"| {LABELS[method]} | {scope} | {metric['frames']} | " + " | ".join(
                fmt(metric[key]) for key in ("mean_ms", "median_ms", "p95_ms", "fps")) + " |")
        lines += ["", f"{LABELS[method]} offline smoothing: {fmt(results['methods'][method]['smoothing_seconds'])} seconds total.", ""]
    lines += ["", "## Graphs", ""]
    for name, caption in saved:
        lines += [f"![{caption}](graphs/{name}.png)", "", f"{caption} [Vector PDF](graphs/{name}.pdf).", ""]
    lines += ["[Full metrics and provenance](comparison.json) · [Input hashes and settings](provenance.json) · "
              "[Frozen references and geometry](references.json) · [Per-frame timings](timings.csv) · "
              "[Smoothing timings](smoothing_times.csv)", ""]
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    results["graphs"] = [f"graphs/{name}.{ext}" for name, _ in saved for ext in ("png", "pdf")]
