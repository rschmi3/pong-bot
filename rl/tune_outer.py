"""
rl/tune_outer.py - pong-tune-outer: outer RL session loop.

Orchestrates one outer session: cups placed on the table, found in any order.
The OuterGRUPolicy recommends a warm start position for each inner search.
Shot records are written only after the operator resolves which cup was hit.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import click

from rl.policy import DEFAULT_START_X, DEFAULT_START_Y
from rl.tune import (
    _configure_logging,
    _load_policy,
    _log_session_end,
    _log_shot,
    run_inner_session,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def _log_outer_attempt_end(
    outer_session_id: str,
    outer_attempt_num: int,
    start_x: int,
    start_y: int,
    outcome: str,
    shots: int,
    resolved_cup_x: Optional[int],
    resolved_cup_y: Optional[int],
    log_path: Path,
) -> None:
    """Append one outer_attempt_end record per search attempt."""
    record: dict = {
        "type":              "outer_attempt_end",
        "outer_session_id":  outer_session_id,
        "outer_attempt_num": outer_attempt_num,
        "start_x":           start_x,
        "start_y":           start_y,
        "outcome":           outcome,
        "shots":             shots,
    }
    if outcome == "hit":
        record["resolved_cup_x"] = resolved_cup_x
        record["resolved_cup_y"] = resolved_cup_y
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def _log_outer_session_end(
    outer_session_id: str,
    inner_policy_name: str,
    outer_policy: str,
    cup_results: list[dict],
    log_path: Path,
) -> None:
    """Append an outer_session_end record with found cups in discovery order."""
    total_shots = sum(r["shots"] for r in cup_results)
    cups_hit = sum(1 for r in cup_results if r["outcome"] == "hit")
    num_cups = len(cup_results)
    record = {
        "type":             "outer_session_end",
        "outer_session_id": outer_session_id,
        "inner_policy":     inner_policy_name,
        "outer_policy":     outer_policy,
        "total_shots":      total_shots,
        "cups_hit":         cups_hit,
        "cup_results":      cup_results,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    logger.info(
        "Outer session %s complete: %d/%d cups in %d total shots",
        outer_session_id, cups_hit, num_cups, total_shots,
    )


def _write_deferred_logs(
    result: dict,
    session_id: str,
    policy_name: str,
    outer_session_id: str,
    outer_attempt_num: int,
    resolved_cup_x: Optional[int],
    resolved_cup_y: Optional[int],
    log_path: Path,
    sigma: Optional[float],
) -> None:
    """
    Write per-shot and session_end records after cup identity has been resolved.

    Called by the outer loop after the operator confirms which cup was hit
    (or after a timeout where cup labels are left as None).
    """
    outcome    = result["outcome"]
    start_x    = result["start_x"]
    start_y    = result["start_y"]
    winning_x  = result["winning_x"]
    winning_y  = result["winning_y"]
    total_shots = result["total_shots"]

    deferred = result.get("deferred_shots", [])
    for s in deferred:
        _log_shot(
            s["shot"],
            s["x_steps"],
            s["y_steps"],
            # reconstruct a minimal result-like namespace
            type("R", (), {
                "hit":          s["hit"],
                "direction":    s["direction"],
                "confidence":   s["confidence"],
                "raw_response": s["raw_response"],
            })(),
            log_path,
            session_id,
            policy_name,
            cup_num=0,                  # cup_num is meaningless in unordered mode
            outer_session_id=outer_session_id,
            cup_x=resolved_cup_x,       # resolved after hit, None on timeout
            cup_y=resolved_cup_y,
        )

    _log_session_end(
        session_id=session_id,
        policy_name=policy_name,
        outcome=outcome,
        total_shots=total_shots,
        start_x=start_x,
        start_y=start_y,
        winning_x=winning_x,
        winning_y=winning_y,
        log_path=log_path,
        cup_num=0,
        outer_session_id=outer_session_id,
        cup_x=resolved_cup_x,
        cup_y=resolved_cup_y,
        sigma=sigma,
    )


# ---------------------------------------------------------------------------
# Robot home helper
# ---------------------------------------------------------------------------


def _home_robot(pi_host: str, pi_port: str, pi_baud: int, dry_run: bool) -> None:
    """SSH to Pi and run pong-motor home to return robot to home position."""
    if dry_run:
        click.echo("  [dry-run] Skipping robot home.")
        return
    click.echo("  Homing robot...")
    ssh_opts = ["-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=15",
                "-o", "BatchMode=yes"]
    cmd = ["ssh", *ssh_opts, f"rschmi3@{pi_host}",
           f"pong-motor --port {pi_port} --baud {pi_baud} home"]
    try:
        subprocess.run(cmd, check=True, timeout=120)
        click.echo("  Robot homed.")
    except subprocess.CalledProcessError as e:
        click.echo(f"  WARNING: pong-motor home failed: {e}", err=True)
    except subprocess.TimeoutExpired:
        click.echo("  WARNING: pong-motor home timed out.", err=True)


# ---------------------------------------------------------------------------
# Hit-resolution prompt
# ---------------------------------------------------------------------------


def _prompt_which_cup(
    remaining_cups: list[tuple[int, int]],
    dry_run: bool,
) -> tuple[int, int]:
    """
    Ask the operator which remaining cup was just hit.

    In dry-run mode, automatically selects the first remaining cup.
    Returns the (cup_x, cup_y) of the resolved cup.
    """
    if dry_run:
        resolved = remaining_cups[0]
        click.echo(f"  [dry-run] Auto-resolving hit to cup {resolved}")
        return resolved

    click.echo("\n  Which cup was hit?")
    for i, (cx, cy) in enumerate(remaining_cups, start=1):
        click.echo(f"    {i}) Cup ({cx},{cy})")

    while True:
        raw = click.prompt(
            f"  Enter number (1-{len(remaining_cups)})",
            default="1",
        )
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(remaining_cups):
                return remaining_cups[idx]
        except ValueError:
            pass
        click.echo(f"  Invalid choice — enter a number between 1 and {len(remaining_cups)}.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--inner-policy",
    default="heuristic",
    type=click.Choice(["heuristic", "gru"], case_sensitive=False),
    show_default=True,
    help="Inner policy for each cup search.",
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
    "cups",
    required=True,
    multiple=True,
    type=str,
    help="Cup grid positions as X,Y (e.g. --cup 3,2 --cup 1,4). "
    "Cups may be hit in any order. "
    "Number of --cup arguments defines the session length.",
)
@click.option(
    "--inner-resume",
    is_flag=True,
    default=False,
    help="Load and resume inner policy from checkpoint.",
)
@click.option(
    "--inner-sigma",
    default=None,
    type=float,
    help="Override exploration noise sigma for the inner GRU policy. "
    "If not set, uses value from checkpoint (default: 0.015). "
    "Only applies to GRUPolicy. Use 0.0 for deterministic inner search.",
)
@click.option(
    "--outer-policy",
    default="fixed",
    type=click.Choice(["fixed", "gru"], case_sensitive=False),
    show_default=True,
    help="Outer policy for recommending starting positions between cups. "
    "'fixed' always starts from the default position. "
    "'gru' loads OuterGRUPolicy from the checkpoint (outer_gru.pt must exist).",
)
@click.option(
    "--max-shots-per-cup",
    default=20,
    show_default=True,
    type=int,
    help="Maximum shots per inner session before a timeout.",
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
    help="Use VLM for shot classification. Default: human prompt.",
)
@click.option(
    "--vlm-model",
    default="Qwen/Qwen3-VL-4B-Instruct",
    show_default=True,
    help="VLM model ID. Only used with --vlm.",
)
@click.option(
    "--device",
    default="auto",
    show_default=True,
    help="Torch device for VLM. Only used with --vlm.",
)
@click.option("--launch-offset", default=24.0, show_default=True, type=float)
@click.option("--capture-secs", default=3.0, show_default=True, type=float)
@click.option("--fps", default=60, show_default=True, type=int)
@click.option("--resolution", default="640x480", show_default=True)
@click.option("--pi-port", default="/dev/ttyUSB0", show_default=True)
@click.option("--pi-baud", default=115200, show_default=True, type=int)
@click.option("--video-log-dir", default=None, type=click.Path())
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Skip hardware — uses random results. Hit-resolution auto-selects "
    "the first remaining cup.",
)
@click.option("--verbose", "-v", is_flag=True, default=False)
def cli(
    inner_policy: str,
    elastics: int,
    cups: tuple[str, ...],
    inner_resume: bool,
    inner_sigma: float | None,
    outer_policy: str,
    max_shots_per_cup: int,
    pi_host: str,
    server_host: str | None,
    stream_port: int,
    vlm: bool,
    vlm_model: str,
    device: str,
    launch_offset: float,
    capture_secs: float,
    fps: int,
    resolution: str,
    pi_port: str,
    pi_baud: int,
    video_log_dir: str | None,
    dry_run: bool,
    verbose: bool,
) -> None:
    """
    Run one outer session: search for cups in any order.

    Cups are specified with --cup X,Y. The session continues until all cups
    are found. After each hit the operator is prompted to identify which
    remaining cup was struck before the session advances.

    On timeout, the same remaining cup set is retried with a new outer-policy
    start recommendation.

    All shot data is logged to rl_shots.jsonl with outer_session_id.
    Shot records are written only after cup identity is resolved so labels
    are never wrong.

    CV image logging is intentionally disabled for outer sessions: multiple
    cups are on the table at once, so images cannot be reliably used as
    single-cup aim-head training data. Use pong-tune for CV data collection.

    Requires pong-motor set-home to have been called on the Pi first.
    """
    _configure_logging(verbose)

    if not dry_run and server_host is None:
        raise click.UsageError("--server-host is required unless --dry-run is set.")

    from utils.data_dir import checkpoint_path as _ckpt_path
    from utils.data_dir import elastics_data_dir, parse_cup_list

    # Parse cup grid positions — defines the initial remaining set
    try:
        initial_cups = parse_cup_list(cups)
    except ValueError as e:
        raise click.BadParameter(str(e))
    num_cups = len(initial_cups)

    data_dir = elastics_data_dir(elastics)
    data_dir.mkdir(parents=True, exist_ok=True)
    inner_ckpt = str(_ckpt_path(data_dir, inner_policy))
    outer_ckpt = str(_ckpt_path(data_dir, "outer_gru")) if outer_policy == "gru" else None
    log_path   = data_dir / "rl_shots.jsonl"

    outer_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load inner policy
    pol = _load_policy(inner_policy, inner_ckpt, inner_resume)

    # Override inner GRU sigma if explicitly provided
    if inner_sigma is not None and hasattr(pol, "sigma"):
        pol.sigma = inner_sigma
        pol.sigma_init = inner_sigma
        click.echo(f"Inner GRU sigma overridden to {inner_sigma}")

    # Capture effective sigma for logging
    effective_sigma = getattr(pol, "sigma", None)

    # Load outer policy
    outer_pol = None
    if outer_policy == "gru":
        from rl.policy import OuterGRUPolicy

        if not outer_ckpt or not Path(outer_ckpt).exists():
            raise click.UsageError(
                f"--outer-policy gru requires a trained checkpoint at {outer_ckpt}.\n"
                "Run the BC warm-start first:\n"
                f"  pong-gen-synthetic-outer --elastics {elastics}\n"
                f"  pong-train-rl outer --elastics {elastics} --mode bc "
                f"--input data/{elastics}_elastics/synthetic_outer.jsonl"
            )
        outer_pol = OuterGRUPolicy(max_shots_per_cup=max_shots_per_cup)
        outer_pol.load(outer_ckpt)
        click.echo(f"OuterGRUPolicy loaded from {outer_ckpt}")

    # Load VLM if needed
    classifier = None
    if not dry_run and vlm:
        from rl.vlm import ShotClassifier

        click.echo(f"Loading VLM {vlm_model}...")
        classifier = ShotClassifier(model_id=vlm_model, device=device)
        click.echo("VLM ready.")

    click.echo(
        f"\npong-tune-outer: inner={inner_policy}  outer={outer_policy}  "
        f"cups={num_cups}  max_shots_per_cup={max_shots_per_cup}\n"
        f"                 outer_session={outer_session_id}\n"
        f"                 pi={pi_host}  server={server_host or 'dry-run'}\n"
        f"                 log → {log_path}"
    )

    cup_labels = ", ".join(f"({cx},{cy})" for cx, cy in initial_cups)
    click.echo(f"\nPlace {num_cups} cups on the table: {cup_labels}")
    click.pause("Press any key to begin the outer session...")
    click.echo("")

    # -----------------------------------------------------------------------
    # Main outer loop — find all cups in any order
    # -----------------------------------------------------------------------

    remaining_cups: list[tuple[int, int]] = list(initial_cups)
    found_index     = 0    # increments only on resolved hits
    outer_attempt_num = 0  # increments on every search attempt
    cup_results: list[dict] = []

    while remaining_cups:
        outer_attempt_num += 1
        remaining_label = ", ".join(f"({cx},{cy})" for cx, cy in remaining_cups)
        click.echo(f"\n{'━' * 60}")
        click.echo(
            f"ATTEMPT {outer_attempt_num}  |  "
            f"Found {found_index}/{num_cups}  |  "
            f"Remaining: {remaining_label}"
        )
        click.echo(f"{'━' * 60}")

        # Get start recommendation
        if outer_pol is not None and found_index > 0:
            start_x, start_y = outer_pol.select_start(found_index + 1)
            click.echo(f"OuterGRU recommends start: X={start_x:+d} Y={start_y:+d}")
        else:
            start_x, start_y = DEFAULT_START_X, DEFAULT_START_Y
            click.echo(f"Default start: X={start_x:+d} Y={start_y:+d}")

        session_id = (
            datetime.now().strftime("%Y%m%d_%H%M%S")
            + f"_attempt{outer_attempt_num}"
        )

        result = run_inner_session(
            inner_policy=pol,
            start_x=start_x,
            start_y=start_y,
            cup_num=0,               # unused in unordered mode; set in deferred logs
            outer_session_id=outer_session_id,
            session_id=session_id,
            policy_name=inner_policy,
            max_shots=max_shots_per_cup,
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
            save_cv=False,
            cv_output_path=Path(data_dir / "shots.jsonl"),
            cv_image_dir_path=Path(data_dir / "images"),
            cv_data_dir=data_dir,
            video_log_path=Path(video_log_dir) if video_log_dir else None,
            checkpoint=inner_ckpt,
            verbose=verbose,
            sigma=effective_sigma,
            defer_logging=True,      # write logs only after cup resolution
        )

        # Use the session_id that may have been resolved inside run_inner_session
        resolved_session_id = result.get("session_id", session_id)

        # Compute mean miss position
        shot_positions = result.get("shot_positions", [])
        miss_positions = [(x, y) for x, y, hit in shot_positions if not hit]
        if miss_positions:
            mean_miss_x = int(sum(x for x, y in miss_positions) / len(miss_positions))
            mean_miss_y = int(sum(y for x, y in miss_positions) / len(miss_positions))
        else:
            mean_miss_x, mean_miss_y = start_x, start_y

        if result["outcome"] == "timeout":
            # ---------------------------------------------------------------
            # Timeout — retry same remaining set, do not advance outer state
            # ---------------------------------------------------------------
            click.echo(f"\n  Timeout after {result['total_shots']} shots.")
            click.echo("  Remaining cups unchanged — retrying.")

            # Write deferred logs with no resolved cup
            _write_deferred_logs(
                result=result,
                session_id=resolved_session_id,
                policy_name=inner_policy,
                outer_session_id=outer_session_id,
                outer_attempt_num=outer_attempt_num,
                resolved_cup_x=None,
                resolved_cup_y=None,
                log_path=log_path,
                sigma=effective_sigma,
            )

            # Log attempt-level record
            _log_outer_attempt_end(
                outer_session_id=outer_session_id,
                outer_attempt_num=outer_attempt_num,
                start_x=start_x,
                start_y=start_y,
                outcome="timeout",
                shots=result["total_shots"],
                resolved_cup_x=None,
                resolved_cup_y=None,
                log_path=log_path,
            )

            # Home robot and prompt to retry
            _home_robot(pi_host, pi_port, pi_baud, dry_run)
            click.pause("Press any key to retry...")

        else:
            # ---------------------------------------------------------------
            # Hit — resolve which cup was found before logging anything
            # ---------------------------------------------------------------
            click.echo(
                f"\n    *** HIT at X={result['winning_x']:+d} "
                f"Y={result['winning_y']:+d} "
                f"({result['total_shots']} shots) ***\n"
            )

            resolved_cx, resolved_cy = _prompt_which_cup(remaining_cups, dry_run)
            found_index += 1

            click.echo(
                f"  Cup ({resolved_cx},{resolved_cy}) confirmed as "
                f"found #{found_index}."
            )

            winning_x = result["winning_x"]
            winning_y = result["winning_y"]

            # Write deferred logs with resolved cup labels
            _write_deferred_logs(
                result=result,
                session_id=resolved_session_id,
                policy_name=inner_policy,
                outer_session_id=outer_session_id,
                outer_attempt_num=outer_attempt_num,
                resolved_cup_x=resolved_cx,
                resolved_cup_y=resolved_cy,
                log_path=log_path,
                sigma=effective_sigma,
            )

            # Log attempt-level record
            _log_outer_attempt_end(
                outer_session_id=outer_session_id,
                outer_attempt_num=outer_attempt_num,
                start_x=start_x,
                start_y=start_y,
                outcome="hit",
                shots=result["total_shots"],
                resolved_cup_x=resolved_cx,
                resolved_cup_y=resolved_cy,
                log_path=log_path,
            )

            # Build cup result in new schema
            cup_results.append({
                "found_index":      found_index,
                "outer_attempt_num": outer_attempt_num,
                "session_id":        resolved_session_id,
                "resolved_cup_x":    resolved_cx,
                "resolved_cup_y":    resolved_cy,
                "start_x":           start_x,
                "start_y":           start_y,
                "winning_x":         winning_x,
                "winning_y":         winning_y,
                "mean_miss_x":       mean_miss_x,
                "mean_miss_y":       mean_miss_y,
                "shots":             result["total_shots"],
                "outcome":           "hit",
            })

            # Update outer GRU only on resolved hits
            if outer_pol is not None:
                outer_pol.update(
                    winning_x=winning_x,
                    winning_y=winning_y,
                    shots_taken=result["total_shots"],
                    start_x=start_x,
                    start_y=start_y,
                    mean_miss_x=mean_miss_x,
                    mean_miss_y=mean_miss_y,
                )

            # Remove the resolved cup from remaining
            remaining_cups.remove((resolved_cx, resolved_cy))

            # Save inner checkpoint
            pol.save(inner_ckpt)

            if remaining_cups:
                remaining_label = ", ".join(
                    f"({cx},{cy})" for cx, cy in remaining_cups
                )
                _home_robot(pi_host, pi_port, pi_baud, dry_run)
                click.echo(f"\n  Remove cup ({resolved_cx},{resolved_cy}).")
                click.echo(f"  Remaining cups: {remaining_label}")
                click.pause("Press any key when ready for next search...")

    # -----------------------------------------------------------------------
    # End outer episode
    # -----------------------------------------------------------------------
    if outer_pol is not None:
        outer_pol.end_outer_episode()
        if outer_ckpt:
            outer_pol.save(outer_ckpt)
            click.echo(f"\nOuterGRUPolicy saved to {outer_ckpt}")

    _log_outer_session_end(
        outer_session_id,
        inner_policy,
        outer_policy,
        cup_results,
        log_path,
    )

    # Summary
    total_shots = sum(r["shots"] for r in cup_results)
    click.echo(f"\n{'=' * 60}")
    click.echo("OUTER SESSION COMPLETE")
    click.echo(f"  Cups found:    {len(cup_results)}/{num_cups}")
    click.echo(f"  Total shots:   {total_shots}")
    click.echo(f"  Attempts made: {outer_attempt_num}")
    click.echo("  Cup results (discovery order):")
    for r in cup_results:
        click.echo(
            f"    #{r['found_index']} Cup ({r['resolved_cup_x']},{r['resolved_cup_y']}): "
            f"{r['shots']:2d} shots  "
            f"started ({r['start_x']:+d},{r['start_y']:+d})  "
            f"won at X={r['winning_x']:+d} Y={r['winning_y']:+d}"
        )
    click.echo(f"  Log: {log_path}")


if __name__ == "__main__":
    cli()
