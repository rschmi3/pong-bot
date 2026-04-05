"""
vision/plots.py - pong-plot-cv: generate CV analysis graphs from cv_shots.jsonl.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import click


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

COLOR_HIT = "tab:green"
COLOR_MISS = "tab:red"
COLOR_TIMEOUT = "tab:gray"
FIG_DPI = 200
FIG_SIZE = (12, 8)
FIG_SIZE_WIDE = (16, 10)

# CV model comparison colours and labels (V1/V2/V3)
COLOR_V1 = "tab:blue"
COLOR_V2 = "tab:orange"
COLOR_V3 = "tab:green"
MODEL_COLORS = {
    "aim_model_v1.onnx": COLOR_V1,
    "aim_model_v2.onnx": COLOR_V2,
    "aim_model_v3.onnx": COLOR_V3,
}
MODEL_LABELS = {
    "aim_model_v1.onnx": "V1 (516-dim)",
    "aim_model_v2.onnx": "V2 (518-dim)",
    "aim_model_v3.onnx": "V3 (split)",
}
# Only primary models in comparison plots (exclude lowtemp variants)
PRIMARY_MODELS = ["aim_model_v1.onnx", "aim_model_v2.onnx", "aim_model_v3.onnx"]

# Threshold for counting a prediction as a simulated hit (steps, both axes)
SIM_HIT_THRESHOLD = 400


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_cv_shots(data_dir: Path) -> list[dict]:
    """Load CV shot records from cv_shots.jsonl."""
    from utils.data_dir import load_jsonl

    return load_jsonl(data_dir / "cv_shots.jsonl")


# ---------------------------------------------------------------------------
# CV helpers
# ---------------------------------------------------------------------------


def _load_ground_truth(
    data_dir: Path,
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    """Compute per-cell median winning motor positions from rl_shots.jsonl."""
    import statistics

    from utils.data_dir import load_jsonl

    rl_records = load_jsonl(data_dir / "rl_shots.jsonl")

    cell_wx: dict[tuple[int, int], list[float]] = defaultdict(list)
    cell_wy: dict[tuple[int, int], list[float]] = defaultdict(list)
    for r in rl_records:
        if r.get("type") != "session_end":
            continue
        if r.get("outcome") != "hit":
            continue
        cx, cy = r.get("cup_x"), r.get("cup_y")
        if cx is None or cy is None:
            continue
        cell_wx[(cx, cy)].append(r["winning_x"])
        cell_wy[(cx, cy)].append(r["winning_y"])

    median_wx = {c: statistics.median(vs) for c, vs in cell_wx.items()}
    median_wy = {c: statistics.median(vs) for c, vs in cell_wy.items()}
    return median_wx, median_wy


def _filter_primary_cv_shots(cv_shots: list[dict]) -> dict[str, list[dict]]:
    """Filter cv_shots to primary models only, grouped by model name.

    Excludes dry runs, shots without hit feedback, and non-primary models
    (e.g. lowtemp variants).

    Returns dict: model_name -> list of shot records.
    """
    by_model: dict[str, list[dict]] = defaultdict(list)
    for s in cv_shots:
        if s.get("dry_run"):
            continue
        if s.get("hit") is None:
            continue
        model = s.get("model")
        if model not in PRIMARY_MODELS:
            continue
        by_model[model].append(s)
    return dict(by_model)


def _iou(pred: tuple[float, ...], gt: tuple[float, ...]) -> float:
    """
    Compute IoU between two bounding boxes in (cx, cy, w, h) normalised format.
    """
    px1, py1 = pred[0] - pred[2] / 2, pred[1] - pred[3] / 2
    px2, py2 = pred[0] + pred[2] / 2, pred[1] + pred[3] / 2
    gx1, gy1 = gt[0] - gt[2] / 2, gt[1] - gt[3] / 2
    gx2, gy2 = gt[0] + gt[2] / 2, gt[1] + gt[3] / 2

    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = pred[2] * pred[3] + gt[2] * gt[3] - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# CV model comparison graphs (V1/V2/V3)
# ---------------------------------------------------------------------------


def _plot_cv_model_hit_rate(cv_shots: list[dict], plots_dir: Path) -> None:
    """Bar chart comparing live hit rates across aim model versions."""
    import matplotlib.pyplot as plt
    import numpy as np

    by_model = _filter_primary_cv_shots(cv_shots)
    if not by_model:
        logger.warning("No primary model CV data - skipping cv-model-hit-rate")
        return

    models = [m for m in PRIMARY_MODELS if m in by_model]
    labels = [MODEL_LABELS.get(m, m) for m in models]
    colors = [MODEL_COLORS.get(m, "tab:gray") for m in models]

    rates = []
    annotations = []
    for m in models:
        shots = by_model[m]
        hits = sum(1 for s in shots if s.get("hit") is True)
        total = len(shots)
        rate = 100.0 * hits / total if total else 0
        rates.append(rate)
        annotations.append(f"{hits}/{total}")

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    x = np.arange(len(models))
    bars = ax.bar(x, rates, color=colors, edgecolor="white", width=0.6)

    for bar, ann in zip(bars, annotations):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            ann,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel("Hit Rate (%)", fontsize=13)
    ax.set_ylim(0, max(rates) * 1.25 if rates else 100)
    ax.set_title("CV Live Hit Rate by Model Version", fontsize=15, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(plots_dir / "cv_model_hit_rate.png", dpi=FIG_DPI)
    plt.close(fig)
    click.echo("  cv_model_hit_rate.png")


def _plot_cv_model_sim_live_gap(
    cv_shots: list[dict],
    data_dir: Path,
    plots_dir: Path,
) -> None:
    """Grouped bar chart: sim hit rate vs live hit rate per model, with gap annotation."""
    import matplotlib.pyplot as plt
    import numpy as np

    by_model = _filter_primary_cv_shots(cv_shots)
    median_wx, median_wy = _load_ground_truth(data_dir)
    if not by_model or not median_wx or not median_wy:
        logger.warning("Insufficient data for sim-live gap plot")
        return

    models = [m for m in PRIMARY_MODELS if m in by_model]
    labels = [MODEL_LABELS.get(m, m) for m in models]

    live_rates = []
    sim_rates = []
    for m in models:
        shots = by_model[m]
        # Live hit rate
        hits = sum(1 for s in shots if s.get("hit") is True)
        total = len(shots)
        live_rates.append(100.0 * hits / total if total else 0)
        # Sim hit rate (predictions within +/-SIM_HIT_THRESHOLD of ground truth on both axes)
        sim_hits = 0
        sim_total = 0
        for s in shots:
            cx, cy = s.get("cup_x"), s.get("cup_y")
            if cx is None or cy is None:
                continue
            if (cx, cy) not in median_wx:
                continue

            sim_total += 1
            x_err = abs(s["predicted_x"] - median_wx[(cx, cy)])
            y_err = abs(s["predicted_y"] - median_wy[(cx, cy)])
            if x_err <= SIM_HIT_THRESHOLD and y_err <= SIM_HIT_THRESHOLD:
                sim_hits += 1
        sim_rates.append(100.0 * sim_hits / sim_total if sim_total else 0)

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    x = np.arange(len(models))
    width = 0.3
    bars_sim = ax.bar(
        x - width / 2,
        sim_rates,
        width,
        label="Sim Hit Rate",
        color=[MODEL_COLORS.get(m, "tab:gray") for m in models],
        alpha=0.5,
        edgecolor="white",
    )
    bars_live = ax.bar(
        x + width / 2,
        live_rates,
        width,
        label="Live Hit Rate",
        color=[MODEL_COLORS.get(m, "tab:gray") for m in models],
        edgecolor="white",
    )

    # Annotate gap
    for i, (sim, live) in enumerate(zip(sim_rates, live_rates)):
        gap = live - sim
        y_pos = max(sim, live) + 2
        ax.text(
            x[i],
            y_pos,
            f"gap: {gap:+.0f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="darkred" if abs(gap) > 10 else "darkgreen",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel("Hit Rate (%)", fontsize=13)
    ax.set_ylim(0, max(max(sim_rates), max(live_rates)) * 1.35)
    ax.set_title("Sim vs Live Hit Rate by Model", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(plots_dir / "cv_model_sim_live_gap.png", dpi=FIG_DPI)
    plt.close(fig)
    click.echo("  cv_model_sim_live_gap.png")


def _plot_cv_model_hit_grid(cv_shots: list[dict], plots_dir: Path) -> None:
    """Multi-panel 6x5 grid showing which cups each model hits."""
    import matplotlib.pyplot as plt
    import numpy as np

    by_model = _filter_primary_cv_shots(cv_shots)
    if not by_model:
        logger.warning("No primary model CV data - skipping cv-model-hit-grid")
        return

    models = [m for m in PRIMARY_MODELS if m in by_model]
    n_models = len(models)
    if n_models == 0:
        return

    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 8))
    if n_models == 1:
        axes = [axes]

    for ax, m in zip(axes, models):
        shots = by_model[m]
        hits = sum(1 for s in shots if s.get("hit") is True)
        total = len(shots)

        # Build grid: 0=not attempted, 1=attempted miss, 2=hit
        grid = np.zeros((5, 6))  # Y rows 1-5, X cols 1-6
        attempted = set()
        hit_cups = set()
        for s in shots:
            cx, cy = s.get("cup_x"), s.get("cup_y")
            if cx is None or cy is None:
                continue
            attempted.add((cx, cy))
            if s.get("hit") is True:
                hit_cups.add((cx, cy))

        for cy in range(1, 6):
            for cx in range(1, 7):
                if (cx, cy) in hit_cups:
                    grid[cy - 1, cx - 1] = 2
                elif (cx, cy) in attempted:
                    grid[cy - 1, cx - 1] = 1

        # Custom colormap: white=0, red=1, green=2
        from matplotlib.colors import ListedColormap

        cmap = ListedColormap(["white", COLOR_MISS, COLOR_HIT])

        # Flip horizontally so X=6 (positive steps, left cup) is on the left
        # Y rows stay as-is: Y=1 (far) at top, Y=5 (near) at bottom
        display_grid = np.fliplr(grid)
        ax.imshow(display_grid, cmap=cmap, vmin=0, vmax=2, aspect="auto")

        # Grid lines and labels: X=6..1 left-to-right, Y=1..5 top-to-bottom
        ax.set_xticks(np.arange(6))
        ax.set_xticklabels([f"X={i}" for i in range(6, 0, -1)], fontsize=10)
        ax.set_yticks(np.arange(5))
        ax.set_yticklabels([f"Y={i}" for i in range(1, 6)], fontsize=10)

        # Cell text: col index in display_grid = 5 - (cx-1) = 6 - cx
        for cy in range(5):
            for cx in range(6):
                val = display_grid[cy, cx]
                if val == 2:
                    ax.text(
                        cx,
                        cy,
                        "HIT",
                        ha="center",
                        va="center",
                        fontsize=9,
                        fontweight="bold",
                        color="white",
                    )
                elif val == 1:
                    ax.text(
                        cx,
                        cy,
                        "miss",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white",
                    )

        label = MODEL_LABELS.get(m, m)
        rate = 100.0 * hits / total if total else 0
        ax.set_title(
            f"{label}\n{hits}/{total} ({rate:.0f}%) - {len(hit_cups)}/30 cups",
            fontsize=13,
            fontweight="bold",
        )

    fig.suptitle(
        "Cup Hit Grid by Model Version", fontsize=16, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    fig.savefig(plots_dir / "cv_model_hit_grid.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    click.echo("  cv_model_hit_grid.png")


def _plot_cv_model_error_by_column(
    cv_shots: list[dict],
    data_dir: Path,
    plots_dir: Path,
) -> None:
    """Line plot: mean signed X error by cup column for each model."""
    import matplotlib.pyplot as plt

    by_model = _filter_primary_cv_shots(cv_shots)
    median_wx, _ = _load_ground_truth(data_dir)
    if not by_model or not median_wx:
        logger.warning("Insufficient data for error-by-column plot")
        return

    models = [m for m in PRIMARY_MODELS if m in by_model]

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # +/-SIM_HIT_THRESHOLD band
    ax.axhspan(
        -SIM_HIT_THRESHOLD,
        SIM_HIT_THRESHOLD,
        alpha=0.1,
        color="green",
        label=f"+-{SIM_HIT_THRESHOLD} threshold",
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    columns = list(range(1, 7))
    for m in models:
        shots = by_model[m]
        col_errors: dict[int, list[float]] = defaultdict(list)
        for s in shots:
            cx, cy = s.get("cup_x"), s.get("cup_y")
            if cx is None or cy is None:
                continue
            if (cx, cy) not in median_wx:
                continue
            col_errors[cx].append(s["predicted_x"] - median_wx[(cx, cy)])

        means = []
        for col in columns:
            errs = col_errors.get(col, [])
            means.append(sum(errs) / len(errs) if errs else float("nan"))

        label = MODEL_LABELS.get(m, m)
        color = MODEL_COLORS.get(m, "tab:gray")
        ax.plot(
            columns,
            means,
            marker="o",
            linewidth=2.5,
            markersize=8,
            label=label,
            color=color,
        )

    ax.set_xlabel("Cup Column (X=6 left, X=1 right)", fontsize=13)
    ax.set_ylabel("Mean Signed X Error (steps)", fontsize=13)
    ax.set_xticks(columns)
    ax.set_xticklabels([f"X={c}" for c in columns], fontsize=11)
    ax.invert_xaxis()
    ax.set_title("X Prediction Error by Column", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "cv_model_error_by_column.png", dpi=FIG_DPI)
    plt.close(fig)
    click.echo("  cv_model_error_by_column.png")


def _plot_cv_model_error_by_row(
    cv_shots: list[dict],
    data_dir: Path,
    plots_dir: Path,
) -> None:
    """Line plot: mean signed Y error by cup row for each model."""
    import matplotlib.pyplot as plt

    by_model = _filter_primary_cv_shots(cv_shots)
    _, median_wy = _load_ground_truth(data_dir)
    if not by_model or not median_wy:
        logger.warning("Insufficient data for error-by-row plot")
        return

    models = [m for m in PRIMARY_MODELS if m in by_model]

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # +/-SIM_HIT_THRESHOLD band
    ax.axhspan(
        -SIM_HIT_THRESHOLD,
        SIM_HIT_THRESHOLD,
        alpha=0.1,
        color="green",
        label=f"+-{SIM_HIT_THRESHOLD} threshold",
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    rows = list(range(1, 6))
    for m in models:
        shots = by_model[m]
        row_errors: dict[int, list[float]] = defaultdict(list)
        for s in shots:
            cx, cy = s.get("cup_x"), s.get("cup_y")
            if cx is None or cy is None:
                continue
            if (cx, cy) not in median_wy:
                continue
            row_errors[cy].append(s["predicted_y"] - median_wy[(cx, cy)])

        means = []
        for row in rows:
            errs = row_errors.get(row, [])
            means.append(sum(errs) / len(errs) if errs else float("nan"))

        label = MODEL_LABELS.get(m, m)
        color = MODEL_COLORS.get(m, "tab:gray")
        ax.plot(
            rows,
            means,
            marker="o",
            linewidth=2.5,
            markersize=8,
            label=label,
            color=color,
        )

    ax.set_xlabel("Cup Row (Y=1 far, Y=5 near)", fontsize=13)
    ax.set_ylabel("Mean Signed Y Error (steps)", fontsize=13)
    ax.set_xticks(rows)
    ax.set_xticklabels([f"Y={r}" for r in rows], fontsize=11)
    ax.invert_xaxis()
    ax.set_title("Y Prediction Error by Row", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "cv_model_error_by_row.png", dpi=FIG_DPI)
    plt.close(fig)
    click.echo("  cv_model_error_by_row.png")


def _plot_cv_model_scatter(
    cv_shots: list[dict],
    data_dir: Path,
    plots_dir: Path,
) -> None:
    """Two-panel scatter: predicted vs ground truth for X and Y axes."""
    import matplotlib.pyplot as plt

    by_model = _filter_primary_cv_shots(cv_shots)
    median_wx, median_wy = _load_ground_truth(data_dir)
    if not by_model or not median_wx:
        logger.warning("Insufficient data for model scatter plot")
        return

    models = [m for m in PRIMARY_MODELS if m in by_model]

    fig, (ax_x, ax_y) = plt.subplots(1, 2, figsize=FIG_SIZE_WIDE)

    for m in models:
        shots = by_model[m]
        gt_xs, pred_xs = [], []
        gt_ys, pred_ys = [], []
        for s in shots:
            cx, cy = s.get("cup_x"), s.get("cup_y")
            if cx is None or cy is None:
                continue
            if (cx, cy) not in median_wx:
                continue
            gt_xs.append(median_wx[(cx, cy)])
            pred_xs.append(s["predicted_x"])
            gt_ys.append(median_wy[(cx, cy)])
            pred_ys.append(s["predicted_y"])

        label = MODEL_LABELS.get(m, m)
        color = MODEL_COLORS.get(m, "tab:gray")
        ax_x.scatter(gt_xs, pred_xs, c=color, s=30, alpha=0.6, label=label)
        ax_y.scatter(gt_ys, pred_ys, c=color, s=30, alpha=0.6, label=label)

    # X panel
    all_x = []
    for m in models:
        for s in by_model[m]:
            cx, cy = s.get("cup_x"), s.get("cup_y")
            if cx is None or cy is None:
                continue
            if (cx, cy) in median_wx:
                all_x.extend([median_wx[(cx, cy)], s["predicted_x"]])
    if all_x:
        x_min, x_max = min(all_x) - 500, max(all_x) + 500
        ax_x.plot(
            [x_min, x_max],
            [x_min, x_max],
            "k--",
            linewidth=1,
            alpha=0.5,
            label="Perfect",
        )
        ax_x.fill_between(
            [x_min, x_max],
            [x_min - SIM_HIT_THRESHOLD, x_max - SIM_HIT_THRESHOLD],
            [x_min + SIM_HIT_THRESHOLD, x_max + SIM_HIT_THRESHOLD],
            alpha=0.08,
            color="green",
        )
        ax_x.set_xlim(x_min, x_max)
        ax_x.set_ylim(x_min, x_max)

    ax_x.set_xlabel("Ground Truth X (steps)", fontsize=12)
    ax_x.set_ylabel("Predicted X (steps)", fontsize=12)
    ax_x.set_title("X: Predicted vs Ground Truth", fontsize=14, fontweight="bold")
    ax_x.legend(fontsize=11)
    ax_x.grid(True, alpha=0.3)
    ax_x.set_aspect("equal", adjustable="box")
    ax_x.invert_xaxis()
    ax_x.invert_yaxis()

    # Y panel
    all_y = []
    for m in models:
        for s in by_model[m]:
            cx, cy = s.get("cup_x"), s.get("cup_y")
            if cx is None or cy is None:
                continue
            if (cx, cy) in median_wy:
                all_y.extend([median_wy[(cx, cy)], s["predicted_y"]])
    if all_y:
        y_min, y_max = min(all_y) - 500, max(all_y) + 500
        ax_y.plot(
            [y_min, y_max],
            [y_min, y_max],
            "k--",
            linewidth=1,
            alpha=0.5,
            label="Perfect",
        )
        ax_y.fill_between(
            [y_min, y_max],
            [y_min - SIM_HIT_THRESHOLD, y_max - SIM_HIT_THRESHOLD],
            [y_min + SIM_HIT_THRESHOLD, y_max + SIM_HIT_THRESHOLD],
            alpha=0.08,
            color="green",
        )
        ax_y.set_xlim(y_min, y_max)
        ax_y.set_ylim(y_min, y_max)

    ax_y.set_xlabel("Ground Truth Y (steps)", fontsize=12)
    ax_y.set_ylabel("Predicted Y (steps)", fontsize=12)
    ax_y.set_title("Y: Predicted vs Ground Truth", fontsize=14, fontweight="bold")
    ax_y.legend(fontsize=11)
    ax_y.grid(True, alpha=0.3)
    ax_y.set_aspect("equal", adjustable="box")
    ax_y.invert_xaxis()
    ax_y.invert_yaxis()

    fig.suptitle(
        "Predicted vs Ground Truth by Model", fontsize=16, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    fig.savefig(plots_dir / "cv_model_scatter.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    click.echo("  cv_model_scatter.png")


# ---------------------------------------------------------------------------
# Detector test visualisation
# ---------------------------------------------------------------------------


def _plot_detect_test(
    data_dir: Path,
    plots_dir: Path,
    score_threshold: float = 0.5,
) -> None:
    """
    Run the trained detector on test images, draw bboxes, save annotated
    images, and report IoU / accuracy statistics.

    Requires torch - lazy-imported inside this function.
    """
    import cv2
    import torch

    from vision.transforms import preprocess

    # Try aim model checkpoints in order of preference (v3 > v2 > v1)
    # All share the same PongDetector backbone - we only need it for detection.
    ckpt_path = None
    for name in ("aim_model_v3.pt", "aim_model_v2.pt", "aim_model_v1.pt"):
        candidate = data_dir / "checkpoints" / name
        if candidate.exists():
            ckpt_path = candidate
            break
    if ckpt_path is None:
        logger.warning(
            "No aim_model_v*.pt found in %s - skipping detect-test",
            data_dir / "checkpoints",
        )
        return

    test_image_dir = data_dir / "test" / "images"
    test_label_dir = data_dir / "test" / "labels"
    if not test_image_dir.exists():
        logger.warning("No test images at %s - skipping detect-test", test_image_dir)
        return

    images = sorted(test_image_dir.glob("*.jpg"))
    if not images:
        logger.warning("No .jpg files in %s - skipping detect-test", test_image_dir)
        return

    # Load backbone from aim model checkpoint (any version - backbone is identical)
    from vision.models.pong_model import PongAimModelV3

    aim_model = PongAimModelV3()
    aim_model.load_state_dict(
        torch.load(str(ckpt_path), map_location="cpu", weights_only=True),
        strict=False,
    )
    model = aim_model.backbone
    model.eval()
    click.echo(f"  Loaded detector backbone from {ckpt_path}")

    out_dir = plots_dir / "detect_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Accumulators
    ious: list[float] = []
    pos_scores: list[float] = []
    neg_scores: list[float] = []
    tp = 0  # true positive: positive image, score >= threshold
    fn = 0  # false negative: positive image, score < threshold
    tn = 0  # true negative: negative image, score < threshold
    fp = 0  # false positive: negative image, score >= threshold

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        # Run inference
        chw = preprocess(frame)
        with torch.no_grad():
            out = model(torch.from_numpy(chw).unsqueeze(0))
        pred_box = out["boxes"][0].tolist()  # (cx, cy, w, h)
        score = out["scores"][0].item()

        # Check for ground truth label
        label_path = test_label_dir / (img_path.stem + ".txt")
        is_positive = label_path.exists()
        gt_box = None

        if is_positive:
            try:
                parts = label_path.read_text().strip().split()
                if len(parts) >= 5:
                    gt_box = tuple(float(p) for p in parts[1:5])
            except (ValueError, OSError):
                pass

        # Compute IoU for positive images
        iou_val = None
        if gt_box is not None:
            iou_val = _iou(tuple(pred_box), gt_box)
            ious.append(iou_val)

        # Classify
        if is_positive:
            pos_scores.append(score)
            if score >= score_threshold:
                tp += 1
            else:
                fn += 1
        else:
            neg_scores.append(score)
            if score < score_threshold:
                tn += 1
            else:
                fp += 1

        # Draw annotated image
        ih, iw = frame.shape[:2]
        annotated = frame.copy()

        # Draw bboxes and text overlay
        cx, cy, w, h = pred_box
        x1 = int((cx - w / 2) * iw)
        y1 = int((cy - h / 2) * ih)
        x2 = int((cx + w / 2) * iw)
        y2 = int((cy + h / 2) * ih)

        if is_positive:
            # Positive image: always draw predicted bbox
            box_color = (0, 255, 0) if score >= score_threshold else (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

            # Ground truth bbox (blue)
            if gt_box is not None:
                gcx, gcy, gw, gh = gt_box
                gx1 = int((gcx - gw / 2) * iw)
                gy1 = int((gcy - gh / 2) * ih)
                gx2 = int((gcx + gw / 2) * iw)
                gy2 = int((gcy + gh / 2) * ih)
                cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)

            # Text
            label_text = f"score={score:.2f}"
            if iou_val is not None:
                label_text += f" IoU={iou_val:.2f}"
            cv2.putText(
                annotated,
                label_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                box_color,
                2,
            )
        else:
            # Negative image: only draw bbox if false positive
            if score >= score_threshold:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label_text = f"score={score:.2f} FALSE POSITIVE"
                text_color = (0, 0, 255)
            else:
                label_text = f"score={score:.2f} NEGATIVE (rejected)"
                text_color = (0, 180, 0)
            cv2.putText(
                annotated,
                label_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                text_color,
                2,
            )

        cv2.imwrite(str(out_dir / f"{img_path.stem}.jpg"), annotated)

    # Print summary
    n_pos = tp + fn
    n_neg = tn + fp
    click.echo(f"\n  Detector Test Results (threshold={score_threshold}):")
    click.echo(f"    Positive images: {n_pos}  (TP={tp}, FN={fn})")
    click.echo(f"    Negative images: {n_neg}  (TN={tn}, FP={fp})")
    if n_pos > 0:
        click.echo(f"    TP rate: {100.0 * tp / n_pos:.1f}%")
        click.echo(f"    Mean positive score: {sum(pos_scores) / len(pos_scores):.3f}")
    if n_neg > 0:
        click.echo(f"    TN rate: {100.0 * tn / n_neg:.1f}%")
        click.echo(f"    Mean negative score: {sum(neg_scores) / len(neg_scores):.3f}")
    if ious:
        click.echo(f"    Mean IoU: {sum(ious) / len(ious):.3f}")
        click.echo(
            f"    IoU > {score_threshold}: {sum(1 for v in ious if v > score_threshold)}/{len(ious)}"
        )

    # Generate IoU histogram
    if ious:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.hist(
            ious, bins=20, range=(0, 1), color=COLOR_HIT, edgecolor="white", alpha=0.8
        )
        ax.axvline(
            score_threshold,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"IoU={score_threshold} threshold",
        )
        mean_iou = sum(ious) / len(ious)
        ax.axvline(
            mean_iou,
            color="tab:blue",
            linestyle="--",
            linewidth=2,
            label=f"Mean IoU={mean_iou:.3f}",
        )
        ax.set_xlabel("IoU", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(
            f"Detection IoU Distribution (n={len(ious)} positive test images)",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(plots_dir / "detect_test_iou.png", dpi=FIG_DPI)
        plt.close(fig)
        click.echo("  detect_test_iou.png")

    click.echo(f"  Annotated images saved to {out_dir}/")


# ---------------------------------------------------------------------------
# Feature visualisation (score heatmap, encoder PCA, Grad-CAM)
# ---------------------------------------------------------------------------


def _plot_features(
    data_dir: Path,
    plots_dir: Path,
    score_threshold: float = 0.5,
) -> None:
    """
    For each test image produce a three-panel visualisation:

      Panel 1 - Score heatmap: sigmoid of the detection head's score channel
                overlaid on the original image, with predicted and GT bboxes.
      Panel 2 - Encoder PCA: the first 3 PCA components of the 256-dim
                block4 feature map projected to RGB.  Semantically distinct
                regions (cup, table, background) typically appear as distinct
                colours.
      Panel 3 - Grad-CAM: gradient of the best-cell score w.r.t. block4
                features, showing which spatial regions drove the prediction.

    Requires torch and sklearn - lazy-imported inside this function.
    """
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from sklearn.decomposition import PCA

    from vision.models.pong_model import PongAimModelV3, PongDetector
    from vision.transforms import preprocess

    # Try aim model checkpoints in order of preference (v3 > v2 > v1)
    ckpt_path = None
    for name in ("aim_model_v3.pt", "aim_model_v2.pt", "aim_model_v1.pt"):
        candidate = data_dir / "checkpoints" / name
        if candidate.exists():
            ckpt_path = candidate
            break
    if ckpt_path is None:
        logger.warning(
            "No aim_model_v*.pt found in %s - skipping features",
            data_dir / "checkpoints",
        )
        return

    aim_model = PongAimModelV3()
    aim_model.load_state_dict(
        torch.load(str(ckpt_path), map_location="cpu", weights_only=True),
        strict=False,
    )
    backbone = aim_model.backbone
    assert isinstance(backbone, PongDetector), (
        f"expected PongDetector, got {type(backbone)}"
    )

    click.echo(f"  Loaded backbone from {ckpt_path}")

    # --- Locate test images ---
    test_image_dir = data_dir / "test" / "images"
    test_label_dir = data_dir / "test" / "labels"
    if not test_image_dir.exists():
        logger.warning("No test images at %s - skipping features", test_image_dir)
        return

    images = sorted(test_image_dir.glob("*.jpg"))
    if not images:
        logger.warning("No .jpg files in %s - skipping features", test_image_dir)
        return

    out_dir = plots_dir / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    click.echo(f"  Processing {len(images)} test images...")

    top_cell_hits = 0
    total_positive = 0

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        ih, iw = frame.shape[:2]

        chw = preprocess(frame)  # [3, 640, 640] float32
        inp = torch.from_numpy(chw).unsqueeze(0)  # [1, 3, 640, 640]

        # Resize frame to 640x640 for blending with heatmaps/Grad-CAM
        # (preprocess already did this; reconstruct BGR uint8 version)
        frame640 = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
        ih, iw = 640, 640  # all panels work at 640x640

        # Read GT label if present
        label_path = test_label_dir / (img_path.stem + ".txt")
        gt_box = None
        if label_path.exists():
            try:
                parts = label_path.read_text().strip().split()
                if len(parts) >= 5:
                    gt_box = tuple(float(p) for p in parts[1:5])
            except (ValueError, OSError):
                pass

        # ----------------------------------------------------------------
        # Panel 1 + 2: score heatmap and PCA - no grad needed
        # ----------------------------------------------------------------
        backbone.eval()
        with torch.no_grad():
            feat = backbone._encode(inp)  # [1, 256, 80, 80]
            det_out = backbone(inp)

        raw = det_out["raw_grid"]  # [1, 5, 80, 80]
        score_map = torch.sigmoid(raw[0, 4]).numpy()  # [80, 80]
        pred_box = det_out["boxes"][0].tolist()  # (cx, cy, w, h)
        pred_score = det_out["scores"][0].item()

        # Score heatmap: upsample, colorise, blend
        score_up = cv2.resize(score_map, (640, 640), interpolation=cv2.INTER_LINEAR)
        score_color = cv2.applyColorMap(
            (score_up * 255).astype(np.uint8),
            cv2.COLORMAP_JET,
        )
        panel1 = cv2.addWeighted(frame640, 0.45, score_color, 0.55, 0)

        # Draw predicted bbox
        cx, cy, w, h = pred_box
        x1 = int((cx - w / 2) * iw)
        y1 = int((cy - h / 2) * ih)
        x2 = int((cx + w / 2) * iw)
        y2 = int((cy + h / 2) * ih)
        box_color = (0, 255, 0) if pred_score >= score_threshold else (0, 0, 255)
        cv2.rectangle(panel1, (x1, y1), (x2, y2), box_color, 2)

        # Draw GT bbox (blue) and compute IoU
        iou_val = None
        if gt_box is not None:
            gcx, gcy, gw, gh = gt_box
            gx1 = int((gcx - gw / 2) * iw)
            gy1 = int((gcy - gh / 2) * ih)
            gx2 = int((gcx + gw / 2) * iw)
            gy2 = int((gcy + gh / 2) * ih)
            cv2.rectangle(panel1, (gx1, gy1), (gx2, gy2), (255, 80, 0), 2)
            iou_val = _iou(tuple(pred_box), gt_box)

            # Top-cell accuracy: predicted best cell within 2 grid cells of GT
            best_idx = torch.sigmoid(raw[0, 4]).flatten().argmax().item()
            best_row_pred = best_idx // 80
            best_col_pred = best_idx % 80
            gt_row = int(gcy * 80)
            gt_col = int(gcx * 80)
            if abs(best_row_pred - gt_row) <= 2 and abs(best_col_pred - gt_col) <= 2:
                top_cell_hits += 1
            total_positive += 1

        # Encoder PCA: [256, 80, 80] -> PCA(3) -> RGB [80, 80, 3]
        f_np = feat[0].numpy()  # [256, 80, 80]
        spatial = f_np.reshape(256, -1).T  # [6400, 256]
        pca = PCA(n_components=3)
        rgb_pca = pca.fit_transform(spatial)  # [6400, 3]
        rgb_pca = rgb_pca.reshape(80, 80, 3)
        # Normalise each channel independently to [0, 1]
        for c in range(3):
            ch = rgb_pca[:, :, c]
            lo, hi = ch.min(), ch.max()
            rgb_pca[:, :, c] = (ch - lo) / (hi - lo + 1e-8)
        panel2 = cv2.resize(
            (rgb_pca * 255).astype(np.uint8),
            (640, 640),
            interpolation=cv2.INTER_NEAREST,
        )  # BGR order doesn't matter - displayed as-is in matplotlib RGB

        # ----------------------------------------------------------------
        # Panel 3: Grad-CAM w.r.t. block4 output
        # ----------------------------------------------------------------
        backbone.eval()

        feat_gc: list[torch.Tensor] = []
        grad_gc: list[torch.Tensor] = []

        def _fwd_hook(m, inp_h, outp):
            feat_gc.clear()
            feat_gc.append(outp.detach())

        def _bwd_hook(m, grad_in, grad_out):
            grad_gc.clear()
            grad_gc.append(grad_out[0].detach())

        h_fwd = backbone.block4.register_forward_hook(_fwd_hook)
        h_bwd = backbone.block4.register_full_backward_hook(_bwd_hook)

        inp_gc = inp.clone().requires_grad_(True)
        gc_out = backbone(inp_gc)
        target = gc_out["scores"][0]  # best-cell confidence, scalar
        target.backward()

        h_fwd.remove()
        h_bwd.remove()

        if feat_gc and grad_gc:
            # Per-channel weights: global average pool of gradients
            weights = grad_gc[0][0].mean(dim=(-2, -1))  # [256]
            cam = (weights[:, None, None] * feat_gc[0][0]).sum(0)  # [80, 80]
            cam = torch.relu(cam).numpy()
            cam_min, cam_max = cam.min(), cam.max()
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        else:
            cam = np.zeros((80, 80), dtype=np.float32)

        cam_up = cv2.resize(cam, (640, 640), interpolation=cv2.INTER_LINEAR)
        cam_color = cv2.applyColorMap(
            (cam_up * 255).astype(np.uint8),
            cv2.COLORMAP_JET,
        )
        panel3 = cv2.addWeighted(frame640, 0.45, cam_color, 0.55, 0)

        # ----------------------------------------------------------------
        # Compose 3-panel figure
        # ----------------------------------------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))

        axes[0].imshow(cv2.cvtColor(panel1, cv2.COLOR_BGR2RGB))
        thr_str = "above" if pred_score >= score_threshold else "below"
        axes[0].set_title(
            f"Score Heatmap  score={pred_score:.3f} ({thr_str} threshold)"
            + (f"  IoU={iou_val:.3f}" if iou_val is not None else "  (negative)"),
            fontsize=11,
        )
        axes[0].axis("off")

        # PCA panel - panel2 is already RGB order (matplotlib expects RGB)
        axes[1].imshow(panel2)
        axes[1].set_title(
            "Encoder Feature PCA  (block4, 256-dim -> 3 PCs)",
            fontsize=11,
        )
        axes[1].axis("off")

        axes[2].imshow(cv2.cvtColor(panel3, cv2.COLOR_BGR2RGB))
        axes[2].set_title(
            "Grad-CAM  (gradient of best-cell score w.r.t. block4)",
            fontsize=11,
        )
        axes[2].axis("off")

        fig.tight_layout()
        fig.savefig(str(out_dir / f"{img_path.stem}.jpg"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Summary
    click.echo(f"  Feature visualisations saved to {out_dir}/")
    if total_positive > 0:
        click.echo(
            f"  Top-cell accuracy (within 2 cells of GT): "
            f"{top_cell_hits}/{total_positive} "
            f"({100.0 * top_cell_hits / total_positive:.1f}%)"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.group("pong-plot-cv")
@click.option(
    "--elastics",
    required=True,
    type=int,
    help="Number of elastics - determines data directory.",
)
@click.pass_context
def cli(ctx: click.Context, elastics: int) -> None:
    """Generate CV analysis graphs from session data."""
    from utils.data_dir import elastics_data_dir

    data_dir = elastics_data_dir(elastics)
    if not data_dir.exists():
        raise click.ClickException(f"Data directory not found: {data_dir}")
    ctx.ensure_object(dict)
    ctx.obj["data_dir"] = data_dir
    ctx.obj["plots_dir"] = data_dir / "plots"


# --- CV graphs ---


@cli.command("cv-models")
@click.pass_context
def cv_models_cmd(ctx: click.Context) -> None:
    """Multi-model CV comparison plots (V1/V2/V3)."""
    data_dir = ctx.obj["data_dir"]
    plots_dir = ctx.obj["plots_dir"]
    plots_dir.mkdir(parents=True, exist_ok=True)
    cv_shots = _load_cv_shots(data_dir)
    if not cv_shots:
        click.echo("No CV shot data found - skipping cv-models plots.")
        return
    click.echo("Generating CV model comparison plots...")
    _plot_cv_model_hit_rate(cv_shots, plots_dir)
    _plot_cv_model_sim_live_gap(cv_shots, data_dir, plots_dir)
    _plot_cv_model_hit_grid(cv_shots, plots_dir)
    _plot_cv_model_error_by_column(cv_shots, data_dir, plots_dir)
    _plot_cv_model_error_by_row(cv_shots, data_dir, plots_dir)
    _plot_cv_model_scatter(cv_shots, data_dir, plots_dir)


# --- Detector test ---


@cli.command("detect-test")
@click.option(
    "--score-threshold",
    default=0.5,
    show_default=True,
    type=float,
    help="Minimum confidence for a positive detection.",
)
@click.pass_context
def detect_test_cmd(ctx: click.Context, score_threshold: float) -> None:
    """Run detector on test images, draw bboxes, report IoU stats."""
    data_dir = ctx.obj["data_dir"]
    plots_dir = ctx.obj["plots_dir"]
    plots_dir.mkdir(parents=True, exist_ok=True)
    _plot_detect_test(data_dir, plots_dir, score_threshold)


# --- Feature visualisation ---


@cli.command("features")
@click.option(
    "--score-threshold",
    default=0.5,
    show_default=True,
    type=float,
    help="Minimum confidence for a positive detection (affects bbox colour).",
)
@click.pass_context
def features_cmd(ctx: click.Context, score_threshold: float) -> None:
    """Score heatmap, encoder PCA, and Grad-CAM for each test image."""
    data_dir = ctx.obj["data_dir"]
    plots_dir = ctx.obj["plots_dir"]
    plots_dir.mkdir(parents=True, exist_ok=True)
    _plot_features(data_dir, plots_dir, score_threshold)


# --- Generate all ---


@cli.command("all")
@click.pass_context
def all_cmd(ctx: click.Context) -> None:
    """Generate all CV graphs."""
    data_dir = ctx.obj["data_dir"]
    plots_dir = ctx.obj["plots_dir"]
    plots_dir.mkdir(parents=True, exist_ok=True)

    # CV graphs
    cv_shots = _load_cv_shots(data_dir)
    if cv_shots:
        click.echo(f"\nLoaded {len(cv_shots)} CV records")
        click.echo("Generating CV model comparison graphs...")
        _plot_cv_model_hit_rate(cv_shots, plots_dir)
        _plot_cv_model_sim_live_gap(cv_shots, data_dir, plots_dir)
        _plot_cv_model_hit_grid(cv_shots, plots_dir)
        _plot_cv_model_error_by_column(cv_shots, data_dir, plots_dir)
        _plot_cv_model_error_by_row(cv_shots, data_dir, plots_dir)
        _plot_cv_model_scatter(cv_shots, data_dir, plots_dir)
    else:
        click.echo("\nNo CV shot data found - skipping CV graphs.")

    # Detector test (requires torch - graceful if unavailable)
    try:
        click.echo("\nGenerating detector test visualisation...")
        _plot_detect_test(data_dir, plots_dir)
    except Exception as e:
        click.echo(f"\nDetector test skipped: {e}")

    click.echo(f"\nAll graphs saved to {plots_dir}/")


if __name__ == "__main__":
    cli()
