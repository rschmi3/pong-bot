"""
rl/shoot.py - pong-shoot CLI entry point (runs on Pi).

Executes a full RL shot cycle in a single command:
  1. Return to home position
  2. Move X and Y axes to the target step positions
  3. Fire - StreamSender.trigger() is called inside robot.fire() between
     the engage and pullback moves, connecting to the server and starting
     the launch_offset countdown at that exact moment
  4. Wait for the capture window to complete
  5. Exit

This is called by pong-tune on the server via SSH. The entire shot
sequence - including the frame stream back to the server - is handled
in one SSH call with no python -m invocations.

Usage
-----
    pong-shoot --x-steps 1200 --y-steps 28500 \\
               --server-host my-server --stream-port 5555 \\
               --launch-offset 24.0 --capture-secs 3.0
"""

from __future__ import annotations

import logging
import sys
import time

import click
import numpy as np

from motor_control.grbl import GrblInterface
from motor_control.robot import Robot
from rl.stream import StreamSender


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        format="%(levelname)-8s %(name)s: %(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
        stream=sys.stderr,
    )


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
    "--server-host",
    required=True,
    help="Tailscale hostname of the server running StreamReceiver.",
)
@click.option(
    "--stream-port",
    default=5555,
    show_default=True,
    type=int,
    help="TCP port to stream frames to on the server.",
)
@click.option(
    "--launch-offset",
    default=24.0,
    show_default=True,
    type=float,
    help="Seconds from when the Z pullback command is sent to when the ball "
         "physically leaves the launcher. Capture begins after this delay.",
)
@click.option(
    "--capture-secs",
    default=3.0,
    show_default=True,
    type=float,
    help="Duration of the video capture window in seconds.",
)
@click.option(
    "--fps",
    default=60,
    show_default=True,
    type=int,
    help="Camera capture frame rate.",
)
@click.option(
    "--resolution",
    default="640x480",
    show_default=True,
    help="Camera capture resolution as WIDTHxHEIGHT (e.g. 640x480).",
)
@click.option(
    "--port",
    default="/dev/ttyUSB0",
    show_default=True,
    help="Serial port for the Arduino.",
)
@click.option(
    "--baud", default=115200, show_default=True, type=int, help="Serial baud rate."
)
@click.option(
    "--feed-x", default=None, type=float, help="Override X axis feed rate mm/min."
)
@click.option(
    "--feed-y", default=None, type=float, help="Override Y axis feed rate mm/min."
)
@click.option(
    "--feed-z", default=None, type=float, help="Override Z axis feed rate mm/min."
)
@click.option(
    "--home-y/--no-home-y",
    default=True,
    show_default=True,
    help="Home Y on the physical limit switch before the shot. "
    "Use --no-home-y to opt out.",
)
@click.option(
    "--capture-only",
    is_flag=True,
    default=False,
    help="Move to position and capture one still frame without firing. "
         "Used for synthetic shot data collection: the robot homes, moves "
         "to the target, captures a still via picamera2, streams it to the "
         "server, then homes again. No robot.fire() is called.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Skip hardware and streaming - safe without hardware connected.",
)
@click.option(
    "--verbose", "-v", is_flag=True, default=False, help="Enable debug logging."
)
def cli(
    x_steps: int,
    y_steps: int,
    server_host: str,
    stream_port: int,
    launch_offset: float,
    capture_secs: float,
    fps: int,
    resolution: str,
    port: str,
    baud: int,
    feed_x: float | None,
    feed_y: float | None,
    feed_z: float | None,
    home_y: bool,
    capture_only: bool,
    dry_run: bool,
    verbose: bool,
) -> None:
    """
    Full RL shot cycle: home → move → fire → stream frames to server.

    With --capture-only: home → move → capture still → stream single frame
    → home. No firing. Used for synthetic shot data collection so the CV
    dataset gets a real image at the motor position without wasting a ball.

    StreamSender.trigger() is called inside robot.fire() at the exact
    moment the Z pullback command is sent. It connects to the server and
    waits launch_offset seconds before capturing, so capture starts right
    as the ball launches.

    Assumes set-home has been called before the session. Called by
    pong-tune on the server via SSH - do not call this manually unless
    StreamReceiver is already listening on the server.
    """
    _configure_logging(verbose)

    axis_feed = {
        k: v
        for k, v in {"X": feed_x, "Y": feed_y, "Z": feed_z}.items()
        if v is not None
    }

    click.echo(
        f"pong-shoot: X={x_steps:+d} Y={y_steps:+d} → {server_host}:{stream_port}"
    )
    if dry_run:
        click.echo("[dry-run] motor commands will be logged, streaming skipped")

    with GrblInterface(port=port, baud=baud, dry_run=dry_run) as iface:
        robot = Robot(iface, axis_feed=axis_feed)

        # 1. Home all axes atomically - Y on physical switch (if enabled),
        #    then X and Z to soft home.
        click.echo("Homing all axes...")
        robot.home_all_axes(home_y=home_y)

        # 2. Move to target position
        click.echo(f"Moving to X={x_steps:+d} Y={y_steps:+d}...")
        robot.move_steps("X", x_steps)
        robot.move_steps("Y", y_steps)

        try:
            w, h = (int(v) for v in resolution.split("x"))
        except ValueError:
            raise click.BadParameter(
                f"Resolution must be WIDTHxHEIGHT, got {resolution!r}",
                param_hint="--resolution",
            )

        if capture_only:
            # 3a. Capture still frame without firing.
            click.echo("Capture-only: capturing still frame...")
            try:
                frame: np.ndarray | None = None
                if not dry_run:
                    try:
                        from picamera2 import Picamera2
                        cam = Picamera2()
                        cam.configure(
                            cam.create_still_configuration(
                                main={"size": (w, h), "format": "RGB888"}
                            )
                        )
                        cam.start()
                        time.sleep(1.0)  # allow auto-exposure to settle
                        frame = cam.capture_array()
                        cam.stop()
                        cam.close()
                        click.echo(f"Still captured ({w}x{h})")
                    except ImportError:
                        click.echo("picamera2 not available - sending placeholder frame")
                        frame = np.full((h, w, 3), 128, dtype=np.uint8)

                # 4a. Send single frame to server.
                if not dry_run and frame is not None:
                    sender = StreamSender(
                        host=server_host,
                        port=stream_port,
                        launch_offset=launch_offset,
                        capture_secs=capture_secs,
                        fps=fps,
                        width=w,
                        height=h,
                    )
                    sender.send_frame(frame)
                    click.echo(f"Still streamed → {server_host}:{stream_port}")
            finally:
                click.echo("Returning to home...")
                robot.home()

        else:
            # 3b. Build StreamSender - trigger() is called inside robot.fire()
            #     between the engage and pullback moves.
            sender = None
            if not dry_run:
                sender = StreamSender(
                    host=server_host,
                    port=stream_port,
                    launch_offset=launch_offset,
                    capture_secs=capture_secs,
                    fps=fps,
                    width=w,
                    height=h,
                )
                click.echo(
                    f"StreamSender ready (launch_offset={launch_offset:.1f}s "
                    f"capture={capture_secs:.1f}s {resolution}@{fps}fps "
                    f"→ {server_host}:{stream_port})"
                )

            # 4b. Fire - trigger() called inside fire() at the pullback moment.
            #     Always return to home even if fire raises.
            click.echo("Firing...")
            try:
                robot.fire(stream_sender=sender)
            finally:
                click.echo("Returning to home...")
                robot.home()

            # 5. Wait for the full capture window to elapse before exiting.
            if not dry_run and sender is not None:
                click.echo("Waiting for capture window...")
                sender.wait()

    click.echo("pong-shoot: complete")


if __name__ == "__main__":
    cli()
