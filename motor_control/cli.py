"""
cli.py - Click-based command-line interface for pong-motor.

Entry point: pong-motor (defined in pyproject.toml)

Global options (pass before the subcommand):
  --port          Serial port device       [default: /dev/ttyUSB0]
  --baud          Baud rate                [default: 115200]
  --dry-run       Print GCode, don't send
  --feed-x        Override X axis feed rate mm/min
  --feed-y        Override Y axis feed rate mm/min
  --feed-z        Override Z axis feed rate mm/min

Subcommands:
  fire          Execute the Z-axis fire sequence
  home          Return all axes to their soft-home positions
  home-y        Home Y on the physical switch and set backed-off zero
  info          Query GRBL firmware info and settings (read-only)
  limit-status  Inspect GRBL status and active switch pins
  recover       Recover from a hard-limit alarm (unlock and jog clear)
  reset         Return one axis to its soft-home position
  set-home      Zero the coordinate system at the current position
  steps         Move an axis by a number of steps
"""

from __future__ import annotations

import logging
import sys
import time

import click

from .grbl import GrblInterface
from .robot import AXIS_FEED, Robot

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(levelname)-8s %(name)s: %(message)s",
        level=level,
        stream=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Shared context object
# ---------------------------------------------------------------------------


class _Context:
    """Holds shared CLI options, passed through to every subcommand."""

    def __init__(
        self,
        port: str,
        baud: int,
        dry_run: bool,
        feed_x: float,
        feed_y: float,
        feed_z: float,
        verbose: bool,
    ) -> None:
        self.port = port
        self.baud = baud
        self.dry_run = dry_run
        self.verbose = verbose
        self.axis_feed = {"X": feed_x, "Y": feed_y, "Z": feed_z}

    def make_robot(self, iface: GrblInterface) -> Robot:
        return Robot(iface, axis_feed=self.axis_feed)

    def make_query_interface(self, port_override: str | None = None) -> GrblInterface:
        return GrblInterface(
            port=port_override or self.port,
            baud=self.baud,
            dry_run=self.dry_run,
            motion_init=False,
        )


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@click.group(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--port",
    default="/dev/ttyUSB0",
    show_default=True,
    help="Serial port connected to the Arduino.",
)
@click.option("--baud", default=115200, show_default=True, help="Serial baud rate.")
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print GCode to stdout instead of sending over serial.",
)
@click.option(
    "--feed-x",
    default=AXIS_FEED["X"],
    show_default=True,
    type=float,
    help="Feed rate for the X axis in mm/min.",
)
@click.option(
    "--feed-y",
    default=AXIS_FEED["Y"],
    show_default=True,
    type=float,
    help="Feed rate for the Y axis in mm/min.",
)
@click.option(
    "--feed-z",
    default=AXIS_FEED["Z"],
    show_default=True,
    type=float,
    help="Feed rate for the Z axis in mm/min.",
)
@click.option(
    "--verbose", "-v", is_flag=True, default=False, help="Enable debug-level logging."
)
@click.pass_context
def cli(
    ctx: click.Context,
    port: str,
    baud: int,
    dry_run: bool,
    feed_x: float,
    feed_y: float,
    feed_z: float,
    verbose: bool,
) -> None:
    """
    pong-motor - Motor control CLI for Pong-Bot.

    Uses GRBL firmware over USB serial. Pass --dry-run to test commands
    without hardware connected.

    X and Y are aiming axes. Z is the firing axis.
    Feed rates are set per-axis and can be overridden with --feed-x / --feed-y / --feed-z.
    """
    _configure_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj = _Context(
        port=port,
        baud=baud,
        dry_run=dry_run,
        feed_x=feed_x,
        feed_y=feed_y,
        feed_z=feed_z,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--axis",
    "-a",
    type=click.Choice(["X", "Y", "Z"], case_sensitive=False),
    required=True,
    help="Axis to move.",
)
@click.option(
    "--steps",
    "-s",
    type=int,
    required=True,
    help="Number of steps (positive or negative).",
)
@click.pass_obj
def steps(ctx: _Context, axis: str, steps: int) -> None:
    """Move an axis by a given number of steps."""
    with GrblInterface(port=ctx.port, baud=ctx.baud, dry_run=ctx.dry_run) as iface:
        ctx.make_robot(iface).move_steps(axis.upper(), steps)


@cli.command()
@click.pass_obj
def fire(ctx: _Context) -> None:
    """Execute the Z-axis fire sequence (engage → pull back → reset)."""
    with GrblInterface(port=ctx.port, baud=ctx.baud, dry_run=ctx.dry_run) as iface:
        ctx.make_robot(iface).fire()


@cli.command("set-home")
@click.pass_obj
def set_home(ctx: _Context) -> None:
    """
    Zero the GRBL coordinate system at the current physical position.

    No motors move. Run this once at startup after manually jogging
    the robot to its resting position with the manual controller.
    """
    with GrblInterface(port=ctx.port, baud=ctx.baud, dry_run=ctx.dry_run) as iface:
        ctx.make_robot(iface).set_home()
        click.echo("Home set at current position.")


@cli.command()
@click.option(
    "--axis",
    "-a",
    type=click.Choice(["X", "Y", "Z"], case_sensitive=False),
    required=True,
    help="Axis to reset to its soft-home position.",
)
@click.pass_obj
def reset(ctx: _Context, axis: str) -> None:
    """Return a single axis to its soft-home position (coordinate 0)."""
    with GrblInterface(port=ctx.port, baud=ctx.baud, dry_run=ctx.dry_run) as iface:
        ctx.make_robot(iface).reset(axis.upper())


@cli.command()
@click.pass_obj
def home(ctx: _Context) -> None:
    """Return all axes (X, Y, Z) to their soft-home positions."""
    with GrblInterface(port=ctx.port, baud=ctx.baud, dry_run=ctx.dry_run) as iface:
        ctx.make_robot(iface).home()


@cli.command()
@click.option(
    "--port",
    "port_override",
    default=None,
    help="Override the serial port for this query only.",
)
@click.pass_obj
def info(ctx: _Context, port_override: str | None) -> None:
    """
    Query GRBL firmware version and all settings. Read-only - safe without --dry-run.

    Sends $I (build info) and $$ (all settings) to GRBL and prints the responses.
    """
    with ctx.make_query_interface(port_override) as iface:
        click.echo("=== GRBL build info ($I) ===")
        click.echo(iface.query("$I"))
        click.echo("\n=== GRBL settings ($$) ===")
        click.echo(iface.query("$$"))


@cli.command("limit-status")
@click.option(
    "--watch",
    is_flag=True,
    default=False,
    help="Continuously poll status until interrupted.",
)
@click.option(
    "--count",
    default=10,
    show_default=True,
    type=int,
    help="Number of polls when using --watch.",
)
@click.option(
    "--interval",
    default=0.5,
    show_default=True,
    type=float,
    help="Seconds between polls when using --watch.",
)
@click.pass_obj
def limit_status(ctx: _Context, watch: bool, count: int, interval: float) -> None:
    """Inspect GRBL status and active limit/switch pins without motion init."""
    polls = count if watch else 1
    with ctx.make_query_interface() as iface:
        for i in range(polls):
            status = iface.query_status()
            parsed = iface.parse_status(status)
            pins = parsed.get("Pn", "-") or "-"
            click.echo(f"state={parsed.get('state', '?')}  pins={pins}  raw={status}")
            if watch and i + 1 < polls:
                time.sleep(interval)


@cli.command("home-y")
@click.pass_obj
def home_y(ctx: _Context) -> None:
    """Home the Y axis on its physical switch and set a backed-off zero."""
    with GrblInterface(port=ctx.port, baud=ctx.baud, dry_run=ctx.dry_run) as iface:
        robot = ctx.make_robot(iface)
        robot.home_y_on_switch()
        click.echo("Y homed on physical switch and zeroed.")


@cli.command("recover")
@click.option(
    "--axis",
    "-a",
    default="Y",
    show_default=True,
    type=click.Choice(["X", "Y", "Z"], case_sensitive=False),
    help="Axis to jog away from the limit switch after unlock.",
)
@click.option(
    "--jog-mm",
    default=5,
    show_default=True,
    type=float,
    help="Distance in mm to jog away from the switch.",
)
@click.option(
    "--feed",
    default=100.0,
    show_default=True,
    type=float,
    help="Feed rate in mm/min for the recovery jog.",
)
@click.pass_obj
def recover(ctx: _Context, axis: str, jog_mm: float, feed: float) -> None:
    """
    Recover from a hard-limit alarm.

    When the Y axis hits the hard limit switch, GRBL locks out
    all commands. This subcommand:

      1. Disables hard limits ($21=0)
      2. Sends $X to clear the alarm
      3. Restores relative mode (G91)
      4. Jogs the axis away from the switch
      5. Re-enables hard limits ($21=1)

    Uses motion_init=False so GRBL's alarm state does not cause a hang.
    """
    with GrblInterface(
        port=ctx.port,
        baud=ctx.baud,
        dry_run=ctx.dry_run,
        motion_init=False,
    ) as iface:
        iface.recover_from_alarm(axis=axis.upper(), jog_mm=jog_mm, feed=feed)
        click.echo(
            f"Recovered: jogged {axis.upper()}+{jog_mm}mm away from limit switch."
        )


if __name__ == "__main__":
    cli()
