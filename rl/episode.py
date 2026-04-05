"""
rl/episode.py - Server-side shot cycle coordination.

Orchestrates one complete RL shot cycle:
  1. Start StreamReceiver listening for frames
  2. SSH to Pi: pong-shoot (homes all axes, moves to target, fires, streams)
  3. StreamReceiver collects frames while pong-shoot runs
  4. Returns frame buffer to caller (for VLM classification or CV collection)

The Pi-side work is done entirely via pong-shoot.
pong-shoot handles homing, movement, firing, and streaming
in a single SSH call. The server side receives the stream and runs any
downstream processing.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time

import numpy as np

from rl.stream import StreamReceiver

logger = logging.getLogger(__name__)

# SSH connection options
SSH_OPTS = [
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "ConnectTimeout=15",
    "-o",
    "BatchMode=yes",
]


def _ssh(host: str, cmd: str, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run a command on the Pi over SSH. Raises on non-zero exit."""
    full_cmd = ["ssh"] + SSH_OPTS + [f"rschmi3@{host}", cmd]
    logger.debug("SSH → %s: %s", host, cmd)
    result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            f"SSH command failed (rc={result.returncode}):\n"
            f"  cmd: {cmd}\n"
            f"  stderr: {result.stderr.strip()}"
        )
    return result


def run_shot(
    x_steps: int,
    y_steps: int,
    pi_host: str,
    server_host: str | None = None,
    stream_port: int = 5555,
    launch_offset: float = 22.0,
    capture_secs: float = 4.0,
    fps: int = 60,
    resolution: str = "1920x1080",
    pi_port: str = "/dev/ttyUSB0",
    pi_baud: int = 115200,
    dry_run: bool = False,
    capture_only: bool = False,
) -> list[np.ndarray]:
    """
    Execute one complete shot cycle and return the captured frames.

    Starts the StreamReceiver before SSH so it is ready when pong-shoot
    initiates streaming. Returns BGR frames from the Pi camera.
    Returns [] on dry_run; capture_only=True returns frames without firing.
    """
    if dry_run:
        logger.info("episode.run_shot: dry_run - skipping hardware")
        time.sleep(1)
        return []

    assert server_host is not None, "server_host is required when dry_run=False"

    # Step 1: Start receiver before triggering the shot so it is ready
    # to accept the Pi's connection when pong-shoot initiates streaming.
    receiver = StreamReceiver(port=stream_port)
    receiver.start_listening()

    # Step 2: Start pong-shoot on Pi in a separate thread so we can
    # receive the stream concurrently.
    # Assumes pong-shoot is on $PATH on the Pi (nix profile install).
    # Capture-only mode has no launch_offset wait so a shorter timeout suffices.
    total_timeout = 30.0 if capture_only else (launch_offset + capture_secs + 60)

    shoot_cmd = (
        f"pong-shoot "
        f"--x-steps {x_steps} "
        f"--y-steps {y_steps} "
        f"--server-host {server_host} "
        f"--stream-port {stream_port} "
        f"--launch-offset {launch_offset} "
        f"--capture-secs {capture_secs} "
        f"--fps {fps} "
        f"--resolution {resolution} "
        f"--port {pi_port} "
        f"--baud {pi_baud}" + (" --capture-only" if capture_only else "")
    )

    shoot_error: list[Exception] = []

    def _run_shoot():
        try:
            _ssh(pi_host, shoot_cmd, timeout=int(total_timeout) + 10)
        except Exception as e:
            shoot_error.append(e)

    shoot_thread = threading.Thread(target=_run_shoot, daemon=True)
    shoot_thread.start()
    logger.info(
        "episode: pong-shoot started on %s (X=%+d Y=%+d)", pi_host, x_steps, y_steps
    )

    # Step 3: Receive frames while shoot runs
    frames = receiver.receive(timeout=total_timeout)

    # Wait for shoot thread to finish
    shoot_thread.join(timeout=10)
    if shoot_error:
        logger.error("episode: pong-shoot error: %s", shoot_error[0])

    logger.info("episode: shot complete - %d frames received", len(frames))
    return frames
