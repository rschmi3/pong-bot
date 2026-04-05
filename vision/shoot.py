"""
vision/cv_shoot.py - pong-cv-shoot: CV inference + shoot on the Pi.

Runs entirely on the Pi with no server dependency:
  1. Captures a still image from picamera2
  2. Runs the ONNX aim model via AimPredictor (cv2.dnn, no PyTorch)
  3. Displays the prediction and prompts for confirmation
  4. Moves the robot to the predicted step values and fires
  5. Saves an annotated debug image with bbox and step values

Usage
-----
    pong-cv-shoot --elastics 1
    pong-cv-shoot --elastics 1 --dry-run
    pong-cv-shoot --elastics 1 --no-confirm
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import click
import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Debug image annotation
# ---------------------------------------------------------------------------


def _annotate_frame(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float, float] | None,
    x_steps: int | None,
    y_steps: int | None,
) -> np.ndarray:
    """
    Draw bounding box and step prediction on the frame.

    Parameters
    ----------
    frame : np.ndarray
        BGR image (will be copied, not modified in place).
    bbox : tuple (cx, cy, w, h, score) normalised [0,1], or None.
    x_steps, y_steps : predicted step values, or None.

    Returns
    -------
    Annotated BGR image.
    """
    annotated = frame.copy()
    h_img, w_img = annotated.shape[:2]

    if bbox is not None:
        cx, cy, w, h, score = bbox
        # Convert normalised bbox to pixel coordinates
        x1 = int((cx - w / 2) * w_img)
        y1 = int((cy - h / 2) * h_img)
        x2 = int((cx + w / 2) * w_img)
        y2 = int((cy + h / 2) * h_img)

        # Draw bbox rectangle (green)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Score label above the box
        label = f"cup {score:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # Step prediction text at the top of the image
    if x_steps is not None and y_steps is not None:
        step_text = f"X={x_steps:+d}  Y={y_steps:+d}"
        cv2.putText(
            annotated,
            step_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )

    return annotated


def _score_heatmap(
    frame: np.ndarray,
    raw_grid: np.ndarray,
) -> np.ndarray:
    """
    Generate a score heatmap blended over the frame.

    Uses only cv2 and numpy - no torch dependency, Pi-compatible.

    Parameters
    ----------
    frame : np.ndarray
        BGR image at any resolution.
    raw_grid : np.ndarray
        Raw detection head output shaped [1, 5, H, W] (H=W=80).
        Channel 4 contains raw score logits; sigmoid is applied here.

    Returns
    -------
    BGR image the same size as frame with the score heatmap blended at
    50% opacity using COLORMAP_JET (blue=low, red=high confidence).
    """
    # Sigmoid on score channel [80, 80] - manual, no torch needed
    score_raw = raw_grid[0, 4].astype(np.float32)  # [80, 80]
    score_map = 1.0 / (1.0 + np.exp(-score_raw))  # [80, 80] in [0, 1]

    # Resize to frame resolution with bilinear interpolation
    h, w = frame.shape[:2]
    score_up = cv2.resize(score_map, (w, h), interpolation=cv2.INTER_LINEAR)

    # Colorise: blue=low confidence, red=high confidence
    score_u8 = (score_up * 255).clip(0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(score_u8, cv2.COLORMAP_JET)

    return cv2.addWeighted(frame, 0.5, heatmap_color, 0.5, 0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("pong-cv-shoot")
@click.option(
    "--elastics",
    required=True,
    type=int,
    help="Number of elastics on the launcher (e.g. 1 or 2). "
    "Determines ONNX model path: data/{N}_elastics/checkpoints/aim_model.onnx",
)
@click.option(
    "--cup",
    default=None,
    type=str,
    help="Cup grid position as X,Y (e.g. --cup 3,2). Optional.",
)
@click.option(
    "--pi-port",
    default="/dev/ttyUSB0",
    show_default=True,
    help="Serial port for GRBL connection.",
)
@click.option(
    "--pi-baud",
    default=115200,
    show_default=True,
    type=int,
    help="Baud rate for GRBL connection.",
)
@click.option(
    "--resolution",
    default="1920x1080",
    show_default=True,
    help="Camera resolution as WIDTHxHEIGHT.",
)
@click.option(
    "--score-threshold",
    default=0.5,
    show_default=True,
    type=float,
    help="Minimum detection confidence to accept a prediction.",
)
@click.option(
    "--no-confirm",
    is_flag=True,
    default=False,
    help="Skip confirmation prompt - fire immediately after prediction.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Predict only - do not move the robot or fire.",
)
@click.option("--verbose", "-v", is_flag=True, default=False)
@click.option(
    "--model-path",
    default=None,
    type=click.Path(exists=True),
    help="Path to ONNX model file. Defaults to "
    "data/{N}_elastics/checkpoints/aim_model.onnx. "
    "Use this to test a specific named model without overwriting "
    "the default aim_model.onnx.",
)
def cli(
    elastics: int,
    cup: str | None,
    pi_port: str,
    pi_baud: int,
    resolution: str,
    score_threshold: float,
    no_confirm: bool,
    dry_run: bool,
    verbose: bool,
    model_path: str | None,
) -> None:
    """
    CV inference + shoot: capture an image, predict aim, and fire.

    Runs entirely on the Pi. Loads the ONNX aim model, captures a still
    from the camera, predicts where to shoot, and (after confirmation)
    moves the robot and fires.
    """
    logging.basicConfig(
        format="%(levelname)-8s %(name)s: %(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
        stream=sys.stdout,
    )

    from utils.data_dir import elastics_data_dir, parse_cup_arg

    # Parse cup grid position
    try:
        cup_x, cup_y = parse_cup_arg(cup)
    except ValueError as e:
        raise click.BadParameter(str(e))

    # Resolve paths
    data_dir = elastics_data_dir(elastics)
    if model_path is None:
        resolved_model_path = data_dir / "checkpoints" / "aim_model.onnx"
    else:
        resolved_model_path = Path(model_path)
    debug_dir = data_dir / "cv_debug"
    cv_shots_path = data_dir / "cv_shots.jsonl"

    if not resolved_model_path.exists():
        raise click.ClickException(
            f"ONNX model not found: {resolved_model_path}\n"
            f"Train with: python -m training.train_head --elastics {elastics} "
            f"--backbone models/cup_detector.pt"
        )

    # Parse resolution
    try:
        w_cam, h_cam = (int(x) for x in resolution.split("x"))
    except ValueError:
        raise click.ClickException(
            f"Invalid resolution format: {resolution!r}. Expected WIDTHxHEIGHT."
        )

    # --- Step 1: Load ONNX model ---
    from vision.detector import AimPredictor

    click.echo(f"Loading model: {resolved_model_path}")
    predictor = AimPredictor(str(resolved_model_path), score_threshold=score_threshold)
    model_name = resolved_model_path.name

    # --- Step 2: Capture image ---
    import time

    click.echo(f"Capturing image ({w_cam}x{h_cam})...")
    from picamera2 import Picamera2

    cam = Picamera2()
    cam.configure(
        cam.create_still_configuration(
            main={"size": (w_cam, h_cam), "format": "RGB888"}
        )
    )
    cam.start()
    # Let auto-exposure and white balance settle before capturing
    time.sleep(1.0)

    # RGB888 in picamera2 is actually BGR in memory - consistent with cv2
    frame = cam.capture_array()

    cam.stop()
    cam.close()

    # --- Step 3: Run inference (single forward pass) ---
    aim, bbox, raw_grid = predictor.predict_and_detect(frame)

    if aim is None:
        click.echo("No cup detected - aborting.")
        _save_debug_image(frame, None, None, None, debug_dir, raw_grid=raw_grid)
        return

    x_steps, y_steps = aim
    score = bbox[4] if bbox is not None else 0.0

    click.echo(f"  Predicted: X={x_steps:+d}  Y={y_steps:+d}  (score={score:.3f})")

    # --- Step 4: Save annotated debug image + score heatmap ---
    debug_path = _save_debug_image(
        frame, bbox, x_steps, y_steps, debug_dir, raw_grid=raw_grid
    )
    debug_rel = str(debug_path.relative_to(data_dir)) if debug_path else ""

    # --- Step 5: Confirm and fire ---
    if dry_run:
        click.echo("Dry run - not firing.")
        _log_cv_shot(
            cv_shots_path,
            cup_x,
            cup_y,
            x_steps,
            y_steps,
            score,
            bbox,
            hit=None,
            dry_run=True,
            debug_image=debug_rel,
            model_name=model_name,
        )
        return

    if not no_confirm:
        click.echo(f"  Fire at X={x_steps:+d} Y={y_steps:+d}? [y/N] : ", nl=False)
        ch = click.getchar().lower()
        click.echo(ch)
        if ch != "y":
            click.echo("Aborted.")
            return

    # --- Step 6: Move and fire ---
    click.echo("Connecting to robot...")
    from motor_control.grbl import GrblInterface
    from motor_control.robot import Robot

    with GrblInterface(port=pi_port, baud=pi_baud) as iface:
        robot = Robot(iface)

        click.echo("Homing...")
        robot.home_all_axes()

        click.echo(f"Moving to X={x_steps:+d} Y={y_steps:+d}...")
        robot.move_steps("X", x_steps)
        robot.move_steps("Y", y_steps)

        click.echo("Firing!")
        robot.fire()

        click.echo("Homing...")
        robot.home()

    # --- Step 7: Hit/miss feedback ---
    click.echo("  Did it hit? [y]es  [n]o  [s]kip : ", nl=False)
    ch = click.getchar().lower()
    click.echo(ch)
    if ch == "y":
        hit = True
    elif ch == "n":
        hit = False
    else:
        hit = None

    _log_cv_shot(
        cv_shots_path,
        cup_x,
        cup_y,
        x_steps,
        y_steps,
        score,
        bbox,
        hit=hit,
        dry_run=False,
        debug_image=debug_rel,
        model_name=model_name,
    )
    click.echo("Done.")


def _save_debug_image(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float, float] | None,
    x_steps: int | None,
    y_steps: int | None,
    debug_dir: Path,
    raw_grid: np.ndarray | None = None,
) -> Path:
    """
    Annotate the frame and save to the debug directory. Returns the path.

    If raw_grid is provided ([1, 5, 80, 80] from the ONNX model), also saves
    a score heatmap image alongside the annotated image as {timestamp}_heatmap.jpg.
    The heatmap uses only cv2/numpy - no torch dependency.
    """
    debug_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = debug_dir / f"{timestamp}.jpg"

    # Save raw un-annotated frame for potential retraining (full resolution)
    raw_path = debug_dir / f"{timestamp}_raw.jpg"
    cv2.imwrite(str(raw_path), frame)
    click.echo(f"  Raw image:   {raw_path}")

    annotated = _annotate_frame(frame, bbox, x_steps, y_steps)
    cv2.imwrite(str(out_path), annotated)
    click.echo(f"  Debug image: {out_path}")

    if raw_grid is not None:
        heatmap_path = debug_dir / f"{timestamp}_heatmap.jpg"
        heatmap = _score_heatmap(frame, raw_grid)
        cv2.imwrite(str(heatmap_path), heatmap)
        click.echo(f"  Heatmap:     {heatmap_path}")

    return out_path


def _log_cv_shot(
    cv_shots_path: Path,
    cup_x: int | None,
    cup_y: int | None,
    predicted_x: int,
    predicted_y: int,
    score: float,
    bbox: tuple[float, float, float, float, float] | None,
    hit: bool | None,
    dry_run: bool,
    debug_image: str,
    model_name: str = "aim_model.onnx",
) -> None:
    """Append a CV shot record to cv_shots.jsonl."""
    import json

    record = {
        "timestamp": datetime.now().isoformat(),
        "cup_x": cup_x,
        "cup_y": cup_y,
        "predicted_x": predicted_x,
        "predicted_y": predicted_y,
        "score": score,
        "bbox": list(bbox[:4]) if bbox else None,
        "hit": hit,
        "dry_run": dry_run,
        "debug_image": debug_image,
        "model": model_name,
    }
    cv_shots_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cv_shots_path, "a") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    cli()
