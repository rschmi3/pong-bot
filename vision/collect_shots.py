"""
vision/collect_shots.py - Data collection script for Pong-Bot.

Captures a frame from the Pi camera, moves to the given X/Y step targets,
fires, then prompts whether the shot scored. Scored (and optionally missed)
shots are appended to a JSONL dataset file.

Usage
-----
    pong-collect --x-steps 1200 --y-steps -300 --elastics 5
    pong-collect --x-steps 1200 --y-steps -300 --elastics 5 --cup 3,2
    pong-collect --x-steps 1200 --y-steps -300 --elastics 5 --dry-run
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import click
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _capture_to_file(path: Path) -> bool:
    """
    Capture a still image directly to a file using picamera2.

    Uses capture_file() which handles encoding internally - avoids any
    array format conversion issues. Returns True on success, False if
    picamera2 is not available (e.g. running outside the Pi).
    """
    try:
        from picamera2 import Picamera2

        _do_capture(Picamera2, str(path))
        return True
    except ImportError:
        logger.warning("picamera2 not available - camera capture skipped")
        return False


def _do_capture(Picamera2, path: str) -> None:
    """Run the picamera2 capture sequence."""
    import time

    cam = Picamera2()
    cam.configure(cam.create_still_configuration(main={"size": (1920, 1080)}))
    cam.start()
    time.sleep(2)  # allow auto-exposure to settle
    cam.capture_file(path)
    cam.stop()
    cam.close()


def _save_image(image_dir: Path, dry_run: bool) -> Path:
    """
    Capture and save an image. Returns the saved path.

    In dry_run mode writes a placeholder grey JPEG instead of using the camera.
    """
    image_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = image_dir / f"{timestamp}.jpg"

    if dry_run:
        placeholder = np.full((1080, 1920, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(path), placeholder)
    else:
        ok = _capture_to_file(path)
        if not ok:
            logger.warning("Camera unavailable - saving placeholder")
            placeholder = np.full((1080, 1920, 3), 128, dtype=np.uint8)
            cv2.imwrite(str(path), placeholder)

    logger.info("Image saved to %s", path)
    return path


def _append_record(output: Path, record: dict) -> None:
    """Append a single JSON record to the JSONL output file."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "a") as f:
        f.write(json.dumps(record) + "\n")
    logger.info("Record appended to %s", output)


@click.command()
@click.option(
    "--x-steps",
    "-x",
    type=int,
    required=True,
    help="Absolute X step target from home position.",
)
@click.option(
    "--y-steps",
    "-y",
    type=int,
    required=True,
    help="Absolute Y step target from home position.",
)
@click.option(
    "--elastics",
    required=True,
    type=int,
    help="Number of elastics on the launcher (e.g. 2 or 4). "
    "Determines data directory: data/{N}_elastics/",
)
@click.option(
    "--cup",
    default=None,
    type=str,
    help="Cup grid position as X,Y (e.g. --cup 3,2). Optional.",
)
@click.option(
    "--port",
    default="/dev/ttyUSB0",
    show_default=True,
    help="Serial port for the Arduino.",
)
@click.option("--baud", default=115200, show_default=True, help="Serial baud rate.")
@click.option(
    "--feed-x",
    default=None,
    show_default=True,
    type=float,
    help="Override X axis feed rate mm/min.",
)
@click.option(
    "--feed-y",
    default=None,
    show_default=True,
    type=float,
    help="Override Y axis feed rate mm/min.",
)
@click.option(
    "--feed-z",
    default=None,
    show_default=True,
    type=float,
    help="Override Z axis feed rate mm/min.",
)
@click.option(
    "--home-y/--no-home-y",
    default=True,
    show_default=True,
    help="Home Y on the physical limit switch before the shot. "
    "Use --no-home-y to opt out.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print GCode and skip camera/serial - safe without hardware.",
)
@click.option(
    "--verbose", "-v", is_flag=True, default=False, help="Enable debug logging."
)
def cli(
    x_steps: int,
    y_steps: int,
    elastics: int,
    cup: str | None,
    port: str,
    baud: int,
    feed_x: float | None,
    feed_y: float | None,
    feed_z: float | None,
    home_y: bool,
    dry_run: bool,
    verbose: bool,
) -> None:
    """
    Capture a shot for the training dataset.

    Captures an image, moves to the target step positions, fires, then
    prompts whether the shot scored.

    Scored shots are always saved. Missed shots prompt for optional saving
    (the image is still useful for object detection training).
    """
    logging.basicConfig(
        format="%(levelname)-8s %(name)s: %(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
        stream=sys.stderr,
    )

    from utils.data_dir import elastics_data_dir, parse_cup_arg

    try:
        cup_x, cup_y = parse_cup_arg(cup)
    except ValueError as e:
        raise click.BadParameter(str(e))

    data_dir_path = elastics_data_dir(elastics)
    data_dir_path.mkdir(parents=True, exist_ok=True)
    output_path = data_dir_path / "shots.jsonl"
    image_dir_path = data_dir_path / "images"

    # ------------------------------------------------------------------
    # 1. Capture frame before moving
    # ------------------------------------------------------------------
    click.echo("Capturing frame...")
    if dry_run:
        click.echo("  [dry-run] placeholder frame used")
    image_path = _save_image(image_dir_path, dry_run)
    click.echo(f"  Saved: {image_path}")

    # ------------------------------------------------------------------
    # 2. Move and fire
    # ------------------------------------------------------------------
    from motor_control.grbl import GrblInterface
    from motor_control.robot import Robot

    with GrblInterface(port=port, baud=baud, dry_run=dry_run) as iface:
        axis_feed = {
            k: v
            for k, v in {"X": feed_x, "Y": feed_y, "Z": feed_z}.items()
            if v is not None
        }
        robot = Robot(iface, axis_feed=axis_feed)

        # Home all axes atomically - Y on physical switch (if enabled),
        # then X and Z to soft home. This ensures a consistent start state.
        click.echo("Homing all axes...")
        robot.home_all_axes(home_y=home_y)

        click.echo(f"Moving to X={x_steps:+d} steps, Y={y_steps:+d} steps...")
        robot.move_steps("X", x_steps)
        robot.move_steps("Y", y_steps)

        click.echo("Firing...")
        try:
            robot.fire()
        finally:
            click.echo("Returning to home...")
            robot.home()

    # ------------------------------------------------------------------
    # 3. Prompt for result
    # ------------------------------------------------------------------
    click.echo("\n  Did it score? [y/n] : ", nl=False)
    ch = click.getchar().lower()
    click.echo(ch)
    scored = ch == "y"

    # Store image path relative to data_dir for portability
    try:
        rel_image = str(image_path.relative_to(data_dir_path))
    except ValueError:
        rel_image = str(image_path)  # fallback: absolute if not under data_dir

    if scored:
        record = {
            "image": rel_image,
            "x_steps": x_steps,
            "y_steps": y_steps,
            "scored": True,
            "cup_x": cup_x,
            "cup_y": cup_y,
        }
        _append_record(output_path, record)
        click.echo(f"Saved scored shot to {output_path}")
    else:
        click.echo("  Save anyway? (useful for detection training) [y/n] : ", nl=False)
        ch = click.getchar().lower()
        click.echo(ch)
        if ch == "y":
            record = {
                "image": rel_image,
                "x_steps": x_steps,
                "y_steps": y_steps,
                "scored": False,
                "cup_x": cup_x,
                "cup_y": cup_y,
            }
            _append_record(output_path, record)
            click.echo(f"Saved missed shot to {output_path}")
        else:
            click.echo("  Keep image file? [y/n] : ", nl=False)
            ch = click.getchar().lower()
            click.echo(ch)
            if ch == "y":
                click.echo("Record discarded - image kept.")
            else:
                if image_path.exists():
                    image_path.unlink()
                    click.echo("Record and image discarded.")
                else:
                    click.echo(
                        "Record discarded - image file not found, nothing to delete."
                    )


if __name__ == "__main__":
    cli()
