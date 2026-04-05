"""
rl/tune.py - pong-tune: run one inner session (search for one cup).

Used for standalone data collection (heuristic and GRU inner sessions).
For multi-cup outer sessions, use pong-tune-outer.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import click

from motor_control.robot import AXIS_MAX_STEPS, AXIS_MIN_STEPS
from rl.policy import DEFAULT_START_X, DEFAULT_START_Y

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        format="%(levelname)-8s %(name)s: %(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
        stream=sys.stdout,
    )
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _make_result(hit: bool, direction: str | None, raw: str):
    """Build a lightweight result object duck-type compatible with ShotResult."""
    import types

    return types.SimpleNamespace(
        hit=hit,
        direction=direction,
        confidence=1.0,
        raw_response=raw,
    )


def _random_result():
    """Generate a random result for dry-run mode."""
    if random.random() < 0.15:
        return _make_result(True, None, "HIT")
    direction = random.choice(["left", "right", "long", "short"])
    return _make_result(False, direction, f"MISS:{direction.upper()}")


def _human_label(shot_num: int, x: int, y: int):
    """
    Prompt the operator for the shot outcome.

    Keys: h=hit  l=left  r=right  s=short  f=far/long  ?=skip
    Returns a result object or None (skip - no policy update).
    """
    click.echo(
        f"  Result? [h]it  [l]eft  [r]ight  [s]hort  [f]ar  [?]skip : ",
        nl=False,
    )
    while True:
        ch = click.getchar().lower()
        mapping = {
            "h": (True, None, "HIT"),
            "l": (False, "left", "MISS:LEFT"),
            "r": (False, "right", "MISS:RIGHT"),
            "s": (False, "short", "MISS:SHORT"),
            "f": (False, "long", "MISS:LONG"),
            "?": None,
        }
        if ch not in mapping:
            continue
        click.echo(ch)
        if mapping[ch] is None:
            return None
        hit, direction, raw = mapping[ch]
        return _make_result(hit, direction, f"human:{raw}")


def _pre_shot_prompt(shot_num: int, x: int, y: int):
    """
    Ask whether to fire the real shot or record a synthetic result.

    Shown before the shot executes so the operator can choose to simulate
    the outcome rather than actually fire - useful for quickly generating
    training trajectories without firing balls.

    Keys:
      g       - go: fire the real shot (default, also accepts Enter)
      h       - synthetic HIT
      l       - synthetic MISS:LEFT
      r       - synthetic MISS:RIGHT
      s       - synthetic MISS:SHORT
      f       - synthetic MISS:LONG (far)

    Returns the string "go" to fire, or a SimpleNamespace result to simulate.
    For synthetic shots the robot still moves to position and captures a still
    frame - no ball is fired.
    """
    click.echo(
        f"  Fire or simulate? [g]o  [h]it  [l]eft  [r]ight  [s]hort  [f]ar : ",
        nl=False,
    )
    while True:
        ch = click.getchar().lower()
        # Accept Enter (\r or \n) as "go"
        if ch in ("\r", "\n", "g"):
            click.echo("g")
            return "go"
        mapping = {
            "h": (True, None, "HIT"),
            "l": (False, "left", "MISS:LEFT"),
            "r": (False, "right", "MISS:RIGHT"),
            "s": (False, "short", "MISS:SHORT"),
            "f": (False, "long", "MISS:LONG"),
        }
        if ch not in mapping:
            continue
        click.echo(ch)
        hit, direction, raw = mapping[ch]
        return _make_result(hit, direction, f"synthetic:{raw}")


def _load_policy(policy_name: str, checkpoint: str | None, resume: bool):
    """Instantiate and optionally load an inner policy from checkpoint."""
    from rl.policy import GRUPolicy, HeuristicPolicy

    policy_map = {
        "heuristic": HeuristicPolicy,
        "gru": GRUPolicy,
    }
    if policy_name not in policy_map:
        raise click.BadParameter(
            f"Unknown policy: {policy_name!r}. Choose from: {', '.join(policy_map)}"
        )
    policy = policy_map[policy_name]()
    if checkpoint and Path(checkpoint).exists():
        policy.load(checkpoint)
        logger.info("%s loaded from %s", policy.__class__.__name__, checkpoint)
    return policy


def _sidecar_path(checkpoint: str) -> Path:
    """Return the sidecar path for a given checkpoint file."""
    return Path(checkpoint).with_suffix(Path(checkpoint).suffix + ".session")


def _write_sidecar(
    checkpoint: str, session_id: str, shots_taken: int, max_shots: int
) -> None:
    """Write session resume state alongside the checkpoint after every shot."""
    path = _sidecar_path(checkpoint)
    path.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "shots_taken": shots_taken,
                "max_shots": max_shots,
            }
        )
    )


def _read_sidecar(checkpoint: str) -> dict | None:
    """
    Read session resume state from the sidecar file.

    Returns the parsed dict, or None if no sidecar exists.
    Fields: session_id (str), shots_taken (int), max_shots (int).
    """
    path = _sidecar_path(checkpoint)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _delete_sidecar(checkpoint: str) -> None:
    """Delete the sidecar file once a session completes cleanly."""
    path = _sidecar_path(checkpoint)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _save_shot_video(
    frames: list,
    shot_num: int,
    result,
    video_dir: Path,
    fps: int = 60,
) -> Optional[Path]:
    """Save frames as an MP4."""
    import cv2

    if not frames:
        return None
    video_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = result.raw_response.replace(":", "-").replace("/", "-")
    path = video_dir / f"{timestamp}_shot{shot_num:02d}_{label}.mp4"
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()
    logger.info("Shot video saved: %s (%d frames)", path, len(frames))
    return path


def _log_shot(
    shot_num: int,
    x: int,
    y: int,
    result,
    log_path: Path,
    session_id: str,
    policy_name: str,
    cup_num: int = 0,
    outer_session_id: Optional[str] = None,
    cup_x: Optional[int] = None,
    cup_y: Optional[int] = None,
) -> None:
    """Append a shot record to the JSONL log file."""
    record = {
        "type": "shot",
        "session_id": session_id,
        "outer_session_id": outer_session_id,
        "cup_num": cup_num,
        "cup_x": cup_x,
        "cup_y": cup_y,
        "policy": policy_name,
        "shot": shot_num,
        "timestamp": datetime.now().isoformat(),
        "x_steps": x,
        "y_steps": y,
        "hit": result.hit,
        "direction": result.direction,
        "confidence": result.confidence,
        "raw_response": result.raw_response,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def _log_session_end(
    session_id: str,
    policy_name: str,
    outcome: str,
    total_shots: int,
    start_x: int,
    start_y: int,
    winning_x: Optional[int],
    winning_y: Optional[int],
    log_path: Path,
    cup_num: int = 0,
    outer_session_id: Optional[str] = None,
    cup_x: Optional[int] = None,
    cup_y: Optional[int] = None,
    sigma: Optional[float] = None,
) -> None:
    """Append a session_end record to the JSONL log file."""
    record = {
        "type": "session_end",
        "session_id": session_id,
        "outer_session_id": outer_session_id,
        "cup_num": cup_num,
        "cup_x": cup_x,
        "cup_y": cup_y,
        "policy": policy_name,
        "outcome": outcome,
        "total_shots": total_shots,
        "start_x": start_x,
        "start_y": start_y,
        "winning_x": winning_x,
        "winning_y": winning_y,
        "sigma": sigma,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    logger.info(
        "Session %s ended: outcome=%s shots=%d start=(%+d,%+d)",
        session_id,
        outcome,
        total_shots,
        start_x,
        start_y,
    )


def _save_cv_sample(
    frames: list,
    x: int,
    y: int,
    hit: bool,
    image_dir: Path,
    output: Path,
    session_id: str,
    data_dir: Path,
) -> Optional[Path]:
    """
    Save the first frame as a CV training sample.

    Image path stored in the record is relative to data_dir so the dataset
    is portable if the data directory is moved.

    Schema matches pong-collect:
        {image, x_steps, y_steps, scored, session_id}
    """
    import cv2

    if not frames:
        return None
    image_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_path = image_dir / f"{timestamp}.jpg"
    cv2.imwrite(str(image_path), frames[0])
    # Store path relative to data_dir for portability
    try:
        rel_path = str(image_path.relative_to(data_dir))
    except ValueError:
        rel_path = str(image_path)  # fallback: absolute if not under data_dir
    record = {
        "image": rel_path,
        "x_steps": x,
        "y_steps": y,
        "scored": hit,
        "session_id": session_id,
    }
    with open(output, "a") as f:
        f.write(json.dumps(record) + "\n")
    logger.info("CV sample saved: %s (scored=%s)", rel_path, hit)
    return image_path


def run_inner_session(
    *,
    inner_policy,
    start_x: int,
    start_y: int,
    cup_num: int,
    outer_session_id: Optional[str],
    session_id: str,
    policy_name: str,
    max_shots: int,
    pi_host: str,
    server_host: Optional[str],
    stream_port: int,
    launch_offset: float,
    capture_secs: float,
    fps: int,
    resolution: str,
    pi_port: str,
    pi_baud: int,
    log_path: Path,
    dry_run: bool,
    vlm: bool,
    classifier,
    save_cv: bool,
    cv_output_path: Path,
    cv_image_dir_path: Path,
    cv_data_dir: Path,
    video_log_path: Optional[Path],
    checkpoint: Optional[str] = None,
    resuming: bool = False,
    shots_already_taken: int = 0,
    session_id_override: Optional[str] = None,
    verbose: bool = False,
    cup_x: Optional[int] = None,
    cup_y: Optional[int] = None,
    sigma: Optional[float] = None,
    defer_logging: bool = False,
) -> dict:
    """
    Run one inner session (search for one cup) and return the result dict.

    Called by both pong-tune (standalone) and pong-tune-outer (outer loop).
    When defer_logging=True, shot records are returned in "deferred_shots"
    for the caller to label before writing.
    """
    # Resolve session id - use the override (from sidecar) when resuming
    if session_id_override:
        session_id = session_id_override

    if resuming:
        # Checkpoint already loaded - policy has the saved position/state.
        # select_action() returns the saved position without resetting anything.
        x, y = inner_policy.select_action(start_x, start_y)
    else:
        x, y = inner_policy.begin_episode(cup_num, start_x, start_y)

    hits = 0
    shot_num = shots_already_taken
    shot_positions: list[tuple[int, int, bool]] = []  # (x, y, hit)
    deferred_shots: list[dict] = []  # populated when defer_logging=True

    while shot_num < max_shots:
        shot_num += 1
        new_x, new_y = inner_policy.select_action(x, y)
        new_x = max(AXIS_MIN_STEPS["X"], min(AXIS_MAX_STEPS["X"], new_x))
        new_y = max(AXIS_MIN_STEPS["Y"], min(AXIS_MAX_STEPS["Y"], new_y))
        prev_pos = (x, y)
        x, y = new_x, new_y

        click.echo(f"  [Shot {shot_num}/{max_shots}] X={x:+d} Y={y:+d}")

        # Pre-shot prompt: fire for real or simulate?
        # Only shown in human-label mode (not VLM, not dry-run).
        if not vlm and not dry_run:
            pre = _pre_shot_prompt(shot_num, x, y)
        else:
            pre = "go"

        synthetic = pre != "go"

        # Execute shot (real fire, or capture-only still for synthetic)
        if dry_run:
            import time

            time.sleep(0.5)
            frames: list = []
        else:
            from rl.episode import run_shot

            frames = run_shot(
                x_steps=x,
                y_steps=y,
                pi_host=pi_host,
                server_host=server_host,
                stream_port=stream_port,
                launch_offset=launch_offset,
                capture_secs=capture_secs,
                fps=fps,
                resolution=resolution,
                pi_port=pi_port,
                pi_baud=pi_baud,
                dry_run=False,
                capture_only=synthetic,
            )
            click.echo(f"    Received {len(frames)} frames")

        # Classify
        if dry_run:
            result = _random_result()
            click.echo(f"    [dry-run] {result.raw_response}")
        elif synthetic:
            result = pre  # already have the result from the pre-shot prompt
            click.echo(f"    Simulated: {result.raw_response}")
        elif vlm:
            click.echo("    Classifying with VLM...")
            assert classifier is not None
            result = classifier.classify(frames)
            click.echo(f"    VLM: {result.raw_response} (conf={result.confidence:.2f})")
        else:
            result = _human_label(shot_num, x, y)
            if result is None:
                shot_num -= 1  # didn't count - retry same shot number
                x, y = prev_pos  # roll back position - retry same target
                click.echo("    Skipped - retrying same shot.")
                continue
            click.echo(f"    Label: {result.raw_response}")

        # Video log
        if video_log_path and not dry_run and frames:
            saved = _save_shot_video(frames, shot_num, result, video_log_path, fps=fps)
            if saved:
                click.echo(f"    Video: {saved.name}")

        # CV collection - save every shot (hits and misses) for Option C weighting
        # Suppressed in deferred mode: cup identity is not resolved yet.
        if save_cv and not dry_run and frames and not defer_logging:
            saved_cv = _save_cv_sample(
                frames,
                x,
                y,
                result.hit,
                cv_image_dir_path,
                cv_output_path,
                session_id,
                cv_data_dir,
            )
            if saved_cv:
                click.echo(
                    f"    CV sample ({('hit' if result.hit else 'miss')}): {saved_cv.name}"
                )

        # Update policy
        inner_policy.update(result, prev_pos, (x, y))

        # Log shot — suppressed in deferred mode; caller writes after resolution
        if not defer_logging:
            _log_shot(
                shot_num,
                x,
                y,
                result,
                log_path,
                session_id,
                policy_name,
                cup_num,
                outer_session_id,
                cup_x=cup_x,
                cup_y=cup_y,
            )
        else:
            deferred_shots.append({
                "shot":         shot_num,
                "x_steps":      x,
                "y_steps":      y,
                "hit":          result.hit,
                "direction":    result.direction,
                "confidence":   result.confidence,
                "raw_response": result.raw_response,
            })

        # Record shot position for outer GRU mean-miss computation
        shot_positions.append((x, y, result.hit))

        # Save checkpoint + sidecar after every shot so a cancelled session
        # can be resumed from the last completed shot with the correct
        # session_id and remaining shot budget.
        if checkpoint:
            inner_policy.save(checkpoint)
            _write_sidecar(checkpoint, session_id, shot_num, max_shots)

        if result.hit:
            hits += 1
            click.echo(f"\n    *** HIT at X={x:+d} Y={y:+d} (shot {shot_num}) ***\n")
            break

    # Session end
    outcome = "hit" if hits > 0 else "timeout"
    winning_x = x if hits > 0 else None
    winning_y = y if hits > 0 else None

    # Log session end — suppressed in deferred mode; caller writes after resolution
    if not defer_logging:
        _log_session_end(
            session_id,
            policy_name,
            outcome,
            shot_num,
            start_x,
            start_y,
            winning_x,
            winning_y,
            log_path,
            cup_num,
            outer_session_id,
            cup_x=cup_x,
            cup_y=cup_y,
            sigma=sigma,
        )
    inner_policy.end_episode()

    # Delete the sidecar - session completed cleanly, nothing to resume.
    if checkpoint:
        _delete_sidecar(checkpoint)

    result_dict: dict = {
        "outcome": outcome,
        "total_shots": shot_num,
        "start_x": start_x,
        "start_y": start_y,
        "winning_x": winning_x,
        "winning_y": winning_y,
        "shot_positions": shot_positions,  # list of (x, y, hit) tuples
    }
    if defer_logging:
        result_dict["deferred_shots"] = deferred_shots
        result_dict["session_id"] = session_id
    return result_dict


# ---------------------------------------------------------------------------
# pong-tune CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--policy",
    default="heuristic",
    type=click.Choice(["heuristic", "gru"], case_sensitive=False),
    show_default=True,
    help=(
        "Policy type:\n"
        "  heuristic - rule-based binary search (strong baseline, no learning)\n"
        "  gru       - GRU sequential policy (falls back to heuristic until\n"
        "              trained; train with: pong-train-rl bc / pong-train-rl rl)"
    ),
)
@click.option(
    "--max-shots",
    default=20,
    show_default=True,
    type=int,
    help="Maximum shots before stopping.",
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
    help="Cup grid position as X,Y (e.g. --cup 3,2). "
    "Optional - used for episode pairing in graphs.",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Load checkpoint and continue from where it left off.",
)
@click.option(
    "--sigma",
    default=None,
    type=float,
    help="Override GRU exploration noise sigma. "
    "If not set, uses value from checkpoint (default: 0.015). "
    "Only applies to GRUPolicy.",
)
@click.option(
    "--pi-host",
    default="pong-pi",
    show_default=True,
    help="Tailscale hostname of the Raspberry Pi.",
)
@click.option(
    "--server-host",
    default=None,
    help="Tailscale hostname of this server. Required unless --dry-run.",
)
@click.option(
    "--stream-port",
    default=5555,
    show_default=True,
    type=int,
    help="TCP port for the frame stream from the Pi.",
)
@click.option(
    "--vlm",
    is_flag=True,
    default=False,
    help="Use VLM to classify outcomes. Default: human prompt.",
)
@click.option(
    "--vlm-model",
    default="Qwen/Qwen3-VL-4B-Instruct",
    show_default=True,
    help="HuggingFace model ID. Only used with --vlm.",
)
@click.option(
    "--device",
    default="auto",
    show_default=True,
    help="Torch device for VLM (auto/cuda/cpu). Only used with --vlm.",
)
@click.option(
    "--save-debug-frames",
    default=None,
    type=click.Path(),
    help="Save VLM input frames as JPEGs. Only used with --vlm.",
)
@click.option(
    "--launch-offset",
    default=24.0,
    show_default=True,
    type=float,
    help="Seconds from Z pullback to ball release.",
)
@click.option(
    "--capture-secs",
    default=3.0,
    show_default=True,
    type=float,
    help="Video capture window duration in seconds.",
)
@click.option(
    "--fps",
    default=60,
    show_default=True,
    type=int,
    help="Camera frame rate passed to pong-shoot.",
)
@click.option(
    "--resolution",
    default="640x480",
    show_default=True,
    help="Camera resolution as WIDTHxHEIGHT.",
)
@click.option(
    "--pi-port",
    default="/dev/ttyUSB0",
    show_default=True,
    help="Serial port on the Pi for the Arduino.",
)
@click.option(
    "--pi-baud", default=115200, show_default=True, type=int, help="Serial baud rate."
)
@click.option(
    "--save-cv/--no-save-cv",
    default=True,
    help="Save every shot frame as CV training data (default: on). "
    "Pass --no-save-cv to disable.",
)
@click.option(
    "--video-log-dir",
    default=None,
    type=click.Path(),
    help="Save per-shot MP4s for review.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Skip hardware - uses random results.",
)
@click.option(
    "--verbose", "-v", is_flag=True, default=False, help="Enable debug logging."
)
def cli(
    policy: str,
    max_shots: int,
    elastics: int,
    cup: str | None,
    resume: bool,
    sigma: float | None,
    pi_host: str,
    server_host: str | None,
    stream_port: int,
    vlm: bool,
    vlm_model: str,
    device: str,
    save_debug_frames: str | None,
    launch_offset: float,
    capture_secs: float,
    fps: int,
    resolution: str,
    pi_port: str,
    pi_baud: int,
    save_cv: bool,
    video_log_dir: str | None,
    dry_run: bool,
    verbose: bool,
) -> None:
    """
    Run one inner RL session: fire shots at a single cup until hit or timeout.

    The robot always starts from (0, 6500). For outer sessions with learned
    warm-starts across multiple cups, use pong-tune-outer instead.

    Requires pong-motor set-home to have been called on the Pi first.
    """
    _configure_logging(verbose)

    if not dry_run and server_host is None:
        raise click.UsageError("--server-host is required unless --dry-run is set.")

    # Parse cup grid position
    from utils.data_dir import checkpoint_path as _ckpt_path
    from utils.data_dir import elastics_data_dir, parse_cup_arg

    try:
        cup_x, cup_y = parse_cup_arg(cup)
    except ValueError as e:
        raise click.BadParameter(str(e))

    # Derive all paths from --elastics
    data_dir = elastics_data_dir(elastics)
    data_dir.mkdir(parents=True, exist_ok=True)
    ckpt = str(_ckpt_path(data_dir, policy))
    log_path = data_dir / "rl_shots.jsonl"
    cv_output = str(data_dir / "shots.jsonl")
    cv_image_dir = str(data_dir / "images")
    cv_data_dir = str(data_dir)

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    inner_policy = _load_policy(policy, ckpt, resume)

    # Override GRU sigma if explicitly provided.
    # Also set sigma_min to the override value so the decay floor matches -
    # without this, --sigma 0.0 only gives zero noise on shot 1 then decays
    # to sigma_min=0.005 for all subsequent shots in the session.
    if sigma is not None and hasattr(inner_policy, "sigma"):
        inner_policy.sigma = sigma
        inner_policy.sigma_init = sigma
        inner_policy.sigma_min = sigma
        click.echo(f"GRU sigma overridden to {sigma}")

    # Capture effective sigma for logging (after any override)
    effective_sigma = getattr(inner_policy, "sigma", None)

    # Read sidecar when resuming so we continue the same session with the
    # correct shot count and max_shots budget.
    shots_already_taken = 0
    session_id_override = None
    if resume:
        sidecar = _read_sidecar(ckpt)
        if sidecar:
            session_id_override = sidecar["session_id"]
            shots_already_taken = sidecar["shots_taken"]
            max_shots = sidecar["max_shots"]
            click.echo(
                f"Resuming session {session_id_override}: "
                f"{shots_already_taken} shots already taken, "
                f"{max_shots - shots_already_taken} remaining."
            )
        else:
            click.echo(
                "WARNING: --resume set but no sidecar found - "
                "starting shot counter from 1 (policy state still restored)."
            )

    classifier = None
    if not dry_run and vlm:
        from rl.vlm import ShotClassifier

        click.echo(f"Loading VLM {vlm_model}...")
        classifier = ShotClassifier(
            model_id=vlm_model,
            device=device,
            debug_frames_dir=save_debug_frames,
        )
        click.echo("VLM ready.")

    label_mode = "vlm" if vlm else "human"
    display_session = session_id_override or session_id
    click.echo(
        f"\npong-tune: policy={policy}  max_shots={max_shots}  labels={label_mode}\n"
        f"           session={display_session}  start=({DEFAULT_START_X:+d}, {DEFAULT_START_Y:+d})\n"
        f"           pi={pi_host}  server={server_host or 'dry-run'}\n"
        f"           log → {log_path}"
    )
    if save_cv:
        click.echo(f"           cv  → {cv_output}  images → {cv_image_dir}")
    if video_log_dir:
        click.echo(f"           video → {video_log_dir}")
    click.echo("")

    result = run_inner_session(
        inner_policy=inner_policy,
        start_x=DEFAULT_START_X,
        start_y=DEFAULT_START_Y,
        cup_num=-1,
        outer_session_id=None,
        session_id=session_id,
        policy_name=policy,
        max_shots=max_shots,
        pi_host=pi_host,
        server_host=server_host,
        stream_port=stream_port,
        launch_offset=launch_offset,
        capture_secs=capture_secs,
        fps=fps,
        resolution=resolution,
        pi_port=pi_port,
        pi_baud=pi_baud,
        log_path=log_path,
        dry_run=dry_run,
        vlm=vlm,
        classifier=classifier,
        save_cv=save_cv,
        cv_output_path=Path(cv_output),
        cv_image_dir_path=Path(cv_image_dir),
        cv_data_dir=Path(cv_data_dir),
        video_log_path=Path(video_log_dir) if video_log_dir else None,
        checkpoint=ckpt,
        resuming=resume,
        shots_already_taken=shots_already_taken,
        session_id_override=session_id_override,
        verbose=verbose,
        cup_x=cup_x,
        cup_y=cup_y,
        sigma=effective_sigma,
    )

    # Save once more after end_episode() - commits the GRU trajectory buffer.
    # Per-shot saves above preserve state for cancel/resume; this captures
    # the episode trajectory committed at session end.
    inner_policy.save(ckpt)

    click.echo(f"\n{'=' * 50}")
    click.echo(
        f"pong-tune complete: {result['total_shots']} shots, "
        f"outcome={result['outcome']}"
    )
    if result["outcome"] == "hit":
        click.echo(
            f"Winning position: X={result['winning_x']:+d} Y={result['winning_y']:+d}"
        )
    click.echo(f"Shot log: {log_path}")


if __name__ == "__main__":
    cli()
