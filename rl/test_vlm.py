"""
vision/test_vlm.py - pong-vlm-test: test the VLM classifier on saved videos.

Useful for iterating on the prompt without firing live shots. Loads an MP4,
extracts frames the same way classify() does, runs the VLM, and prints the
full reasoning text and classification.

Usage
-----
    # Test a single video
    pong-vlm-test data/videos/20260325_shot01_MISS-RIGHT.mp4

    # Test all MP4s in a directory (batch mode)
    pong-vlm-test data/videos/

    # Save the 24 subsampled frames for inspection
    pong-vlm-test data/videos/shot01.mp4 --save-frames /tmp/frames

    # Batch mode with accuracy score (labels parsed from filenames)
    pong-vlm-test data/videos/ --score
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import click
import cv2

# Reduce CUDA memory fragmentation - required for Qwen3-VL-4B at 1280×720
# with 24 frames on 12GB VRAM. Must be set before any CUDA allocation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _load_frames(video_path: Path) -> list:
    """Load all frames from an MP4 file as BGR numpy arrays."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def _parse_label_from_filename(name: str) -> str | None:
    """
    Extract expected label from a filename like:
      20260325_021916_shot01_MISS-RIGHT.mp4
    Returns 'MISS:RIGHT', 'MISS:LEFT', 'HIT', etc. or None if not found.
    """
    stem = Path(name).stem.upper()
    for token in ("HIT", "MISS-LEFT", "MISS-RIGHT", "MISS-LONG", "MISS-SHORT"):
        if stem.endswith(token) or f"_{token}_" in stem:
            return token.replace("-", ":")
    return None


def _run_single(
    video_path: Path,
    classifier,
    save_frames_dir: Path | None,
    num_frames: int,
    target_w: int | None = None,
    target_h: int | None = None,
) -> tuple[str, str | None]:
    """
    Run the VLM on a single video. Returns (raw_response, parsed_token).

    If target_w / target_h are provided, every frame is resized to that
    resolution before frame selection and VLM inference. This lets you
    test old high-resolution videos without OOM.
    """
    from rl.vlm import ShotClassifier

    frames = _load_frames(video_path)
    if not frames:
        click.echo(f"  ERROR: no frames read from {video_path.name}", err=True)
        return "", None

    if target_w is not None and target_h is not None:
        frames = [cv2.resize(f, (target_w, target_h)) for f in frames]

    sampled = ShotClassifier._select_frames(frames, num_frames)

    if save_frames_dir:
        save_frames_dir.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(sampled):
            out = save_frames_dir / f"frame_{i:03d}.jpg"
            cv2.imwrite(str(out), f)
        click.echo(f"  Saved {len(sampled)} frames to {save_frames_dir}")

    result = classifier.classify(sampled)
    return result.raw_response, (
        "HIT" if result.hit else f"MISS:{result.direction.upper()}" if result.direction else "UNKNOWN"
    )


@click.command("pong-vlm-test")
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--model",
    default="Qwen/Qwen3-VL-4B-Instruct",
    show_default=True,
    help="HuggingFace VLM model ID.",
)
@click.option(
    "--device",
    default="auto",
    show_default=True,
    help="Torch device: auto, cuda, or cpu.",
)
@click.option(
    "--save-frames",
    default=None,
    type=click.Path(),
    help="Save the subsampled frames as JPEGs to this directory for inspection.",
)
@click.option(
    "--target-resolution",
    default="640x480",
    show_default=True,
    help="Resize frames to WIDTHxHEIGHT before passing to the VLM. "
         "Defaults to 640x480 to match the current stream defaults and "
         "avoid OOM when testing old high-resolution videos. "
         "Use 'none' to disable resizing.",
)
@click.option(
    "--score",
    is_flag=True,
    default=False,
    help="In directory mode, parse expected labels from filenames and print "
         "accuracy score. Labels are extracted from filename suffixes like "
         "_MISS-RIGHT or _HIT.",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Enable debug logging.",
)
def cli(
    path: str,
    model: str,
    device: str,
    save_frames: str | None,
    target_resolution: str,
    score: bool,
    verbose: bool,
) -> None:
    """
    Test the VLM classifier on one or more saved shot videos.

    PATH can be a single MP4 file or a directory of MP4 files.

    In single-file mode, the full VLM reasoning text and classification
    are printed. In directory mode, a summary table is printed and
    optionally an accuracy score if --score is set.
    """
    logging.basicConfig(
        format="%(levelname)-8s %(name)s: %(message)s",
        level=logging.DEBUG if verbose else logging.WARNING,
        stream=sys.stderr,
    )

    # Suppress noisy HF HTTP debug output
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    from rl.vlm import NUM_FRAMES, ShotClassifier

    save_frames_path = Path(save_frames) if save_frames else None

    # Parse --target-resolution
    target_w: int | None = None
    target_h: int | None = None
    if target_resolution.lower() != "none":
        try:
            target_w, target_h = (int(v) for v in target_resolution.lower().split("x"))
        except (ValueError, AttributeError):
            raise click.BadParameter(
                f"Expected WIDTHxHEIGHT or 'none', got {target_resolution!r}",
                param_hint="--target-resolution",
            )
        click.echo(f"Resizing frames to {target_w}x{target_h} before VLM inference.")

    click.echo(f"Loading model {model}...")
    classifier = ShotClassifier(model_id=model, device=device)
    click.echo("Model ready.\n")

    target = Path(path)

    if target.is_file():
        # ── Single file mode ─────────────────────────────────────────────
        if not target.suffix.lower() == ".mp4":
            raise click.BadParameter(f"Expected .mp4 file, got: {target}")

        click.echo(f"{'━' * 60}")
        click.echo(f"Video: {target.name}")
        click.echo(f"{'━' * 60}")

        raw, token = _run_single(
            target, classifier, save_frames_path, NUM_FRAMES, target_w, target_h
        )

        click.echo("VLM reasoning + classification:")
        click.echo()
        for line in raw.strip().splitlines():
            click.echo(f"  {line}")
        click.echo()
        click.echo(f"Parsed result: {token}")

    else:
        # ── Directory batch mode ─────────────────────────────────────────
        videos = sorted(target.glob("*.mp4"))
        if not videos:
            raise click.ClickException(f"No MP4 files found in {target}")

        click.echo(f"Found {len(videos)} videos in {target}\n")

        correct = 0
        total_scored = 0
        results = []

        for video in videos:
            click.echo(f"{'─' * 60}")
            click.echo(f"{video.name}")

            frames_dir = save_frames_path / video.stem if save_frames_path else None
            raw, token = _run_single(
                video, classifier, frames_dir, NUM_FRAMES, target_w, target_h
            )

            expected = _parse_label_from_filename(video.name) if score else None

            # Print abbreviated reasoning (first line only) + token
            first_line = raw.strip().splitlines()[0] if raw.strip() else ""
            if len(first_line) > 100:
                first_line = first_line[:97] + "..."
            click.echo(f"  Reasoning: {first_line}")
            click.echo(f"  Result:    {token}", nl=False)

            if score and expected is not None:
                match = token == expected
                correct += int(match)
                total_scored += 1
                marker = "✓" if match else f"✗ (expected {expected})"
                click.echo(f"  {marker}")
            else:
                click.echo()

            results.append((video.name, token, expected))

        click.echo(f"\n{'━' * 60}")
        click.echo(f"Processed {len(videos)} videos")

        if score and total_scored > 0:
            pct = 100 * correct / total_scored
            click.echo(f"Accuracy: {correct}/{total_scored} = {pct:.0f}%")
            click.echo()
            click.echo("Breakdown:")
            for name, got, exp in results:
                if exp is not None:
                    marker = "✓" if got == exp else "✗"
                    click.echo(f"  {marker} {name}")
                    if got != exp:
                        click.echo(f"      got={got}  expected={exp}")


if __name__ == "__main__":
    cli()
