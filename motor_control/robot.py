"""
motor_control/robot.py - High-level motor control API.

All moves use G91 (relative positioning). set_home() sets G92 origin;
reset() and home() switch briefly to G90 to return to origin.
"""

from __future__ import annotations

import logging
import time
from typing import Protocol

from .grbl import GrblInterface

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants - tune these to match your hardware
# ---------------------------------------------------------------------------

# Per-axis feed rates in mm/min.
# Firmware max rates: $110=$111=1000 (X/Y), $112=600 (Z).
AXIS_FEED: dict[str, float] = {
    "X": 600.0,
    "Y": 600.0,
    "Z": 13000.0,
}

# Z-axis fire stroke - absolute number of steps for both engage and pull back.
# fire() moves Z- by this amount to lock onto the launch mechanism, then Z+
# by the same amount to pull back and release the ball.
FIRE_STEPS: int = 205000

# Y-axis limit-switch homing configuration.
# Hard limits are disabled ($21=0) during homing - switch detection is done
# entirely via polling Pn: in the status report.

Y_HOME_DIRECTION: int = -1  # -1: home is toward Y-, +1: toward Y+

# Jog feed rate for the approach move (toward the switch).
Y_HOME_APPROACH_FEED: float = 50.0  # mm/min

# Jog feed rate for backoff and zero-offset moves (away from the switch).
Y_HOME_BACKOFF_FEED: float = 70.0  # mm/min

# Maximum steps to travel before aborting if the switch is not found.
Y_HOME_MAX_TRAVEL: int = 25000

# Maximum steps to back off before raising an error.
Y_HOME_MAX_BACKOFF: int = 1400

# Final offset: steps to move away from the released switch before G92 Y0.
Y_HOME_ZERO_OFFSET_STEPS: int = 600

# Settle delay after the final zero-offset move.
Y_HOME_SETTLE_SECS: float = 0.2

# Post-fire delay - lets the ball clear the launcher before homing.
FIRE_SETTLE_SECS: float = 1.0

# Soft travel limits in steps from the G92 home origin.
# move_steps() raises ValueError if the projected absolute position would
# exceed these bounds. Limits match the physical range of each axis:
#   X - symmetric aiming axis, ±10000 steps from centre
#   Y - elevation axis, homes to 0 on limit switch, only travels positive
#   Z - fire axis, home is 0, fire sequence travels negative to -FIRE_STEPS
AXIS_MIN_STEPS: dict[str, int] = {"X": -10000, "Y": 0, "Z": -FIRE_STEPS}
AXIS_MAX_STEPS: dict[str, int] = {"X": 10000, "Y": 20000, "Z": 0}

# Valid axis identifiers
VALID_AXES = ("X", "Y", "Z")


class _Triggerable(Protocol):
    """Structural type for objects with a trigger() method (e.g. StreamSender)."""
    def trigger(self) -> None: ...


class Robot:
    """
    High-level motor control API for Pong-Bot.

    All moves are relative (G91) - each call moves by the given amount
    from the current position. Each axis uses its own feed rate from
    AXIS_FEED, included inline in every G01 command.

    Parameters
    ----------
    interface : GrblInterface
        An open (or dry-run) GrblInterface instance.
    axis_feed : dict[str, float] | None
        Per-axis feed rate overrides in mm/min. Any axis not specified
        falls back to AXIS_FEED. Useful for the --feed CLI flag.
    """

    def __init__(
        self,
        interface: GrblInterface,
        axis_feed: dict[str, float] | None = None,
    ) -> None:
        self._grbl = interface
        # Merge caller overrides with module defaults
        self._feed = {**AXIS_FEED, **(axis_feed or {})}
        self._steps_per_mm = self._grbl.query_steps_per_mm()
        # Tracked absolute position in steps from the G92 origin.
        # Kept in sync by move_steps(), reset(), home(), and set_home().
        self._pos: dict[str, int] = {"X": 0, "Y": 0, "Z": 0}
        logger.info(
            (
                "Robot ready - feed rates: X=%.0f Y=%.0f Z=%.0f mm/min; "
                "steps/mm: X=%.0f Y=%.0f Z=%.0f"
            ),
            self._feed["X"],
            self._feed["Y"],
            self._feed["Z"],
            self._steps_per_mm["X"],
            self._steps_per_mm["Y"],
            self._steps_per_mm["Z"],
        )

    def _feed_for(self, axis: str) -> float:
        return self._feed.get(axis, AXIS_FEED.get(axis, 200.0))

    def y_switch_triggered(self) -> bool:
        """
        Return True if the GRBL status report shows the Y limit/switch active.

        Raises RuntimeError if the controller does not expose limit pin state.
        """
        pins = self._grbl.limit_pins()
        return "Y" in pins

    # ------------------------------------------------------------------
    # General movement primitives
    # ------------------------------------------------------------------

    def move_steps(self, axis: str, steps: int) -> None:
        """
        Move an axis by a given number of motor steps relative to its
        current position.

        Converts steps → mm using the GRBL steps/mm setting, then sends
        a G01 relative move with the axis-specific feed rate inline.

        Parameters
        ----------
        axis : str
            Axis to move: 'X', 'Y', or 'Z' (case-insensitive).
        steps : int
            Number of motor steps to move. Positive or negative.
        """
        axis = axis.upper()
        if axis not in VALID_AXES:
            raise ValueError(f"Invalid axis '{axis}'. Must be one of {VALID_AXES}.")

        new_pos = self._pos[axis] + steps
        lo = AXIS_MIN_STEPS[axis]
        hi = AXIS_MAX_STEPS[axis]
        if not (lo <= new_pos <= hi):
            raise ValueError(
                f"move_steps({axis}, {steps:+d}): projected position {new_pos:+d} "
                f"is out of bounds [{lo}, {hi}]"
            )

        mm = steps / self._steps_per_mm[axis]
        feed = self._feed_for(axis)
        logger.info(
            "move_steps(%s, %+d) → %.6f mm @ %.0f mm/min  (pos %+d → %+d)",
            axis,
            steps,
            mm,
            feed,
            self._pos[axis],
            new_pos,
        )
        self._grbl.send_code(f"G01 {axis}{mm:.6f} F{feed:.0f}", wait=0.0)
        self._pos[axis] = new_pos

    # ------------------------------------------------------------------
    # Fire sequence
    # ------------------------------------------------------------------

    def fire(self, stream_sender: _Triggerable | None = None) -> None:
        """Execute the full Z-axis fire sequence: engage, trigger stream, release, reset Z."""
        logger.info("fire(): engaging launch mechanism (Z -%d steps)", FIRE_STEPS)
        self.move_steps("Z", -FIRE_STEPS)

        logger.info("fire(): waiting for engage to complete")
        self._grbl.wait_for_idle(timeout=60.0)

        if stream_sender is not None:
            logger.info("fire(): triggering stream sender")
            stream_sender.trigger()

        logger.info("fire(): pulling back to release (Z +%d steps)", FIRE_STEPS)
        self.move_steps("Z", FIRE_STEPS)

        logger.info("fire(): resetting Z to home")
        self.reset("Z")

        if not self._grbl.dry_run:
            logger.info("fire(): settling %.1fs for ball to clear", FIRE_SETTLE_SECS)
            time.sleep(FIRE_SETTLE_SECS)

    def home_y_on_switch(self) -> None:
        """Home the Y axis onto its physical limit switch and set G92 Y0 at the backed-off position."""
        logger.info("home_y_on_switch(): starting")

        if self._grbl.dry_run:
            logger.info("home_y_on_switch(): dry run - simulating Y zero")
            self._grbl.send_code("G92 Y0", wait=0.0)
            self._pos["Y"] = 0
            return

        # Hard limits are disabled inside _home_y_sequence for the duration
        # of homing, and re-enabled at the end of a successful sequence.
        # On exception: ensure hard limits are re-enabled before re-raising.
        try:
            self._home_y_sequence()
        except Exception:
            logger.warning(
                "home_y_on_switch(): exception during sequence - re-enabling hard limits"
            )
            try:
                self._grbl.set_hard_limits(True)
            except Exception:
                pass
            raise

    def _home_y_sequence(self) -> None:
        """Inner Y homing sequence with hard limits disabled. Approach, backoff, zero offset, set G92 Y0."""
        approach_mm = Y_HOME_MAX_TRAVEL / self._steps_per_mm["Y"]
        backoff_mm = Y_HOME_MAX_BACKOFF / self._steps_per_mm["Y"]
        offset_mm = Y_HOME_ZERO_OFFSET_STEPS / self._steps_per_mm["Y"]
        d = Y_HOME_DIRECTION  # -1 toward switch

        # Disable hard limits for the entire homing sequence.
        # Switch detection is done via polling Pn:Y - no alarm needed.
        self._grbl.set_hard_limits(False)

        # Ensure relative mode.
        self._grbl.send_code("G91", wait=0.0)

        # --- Pre-check: already on switch? Back off first. ---
        if self.y_switch_triggered():
            logger.info("Switch already active - backing off before approach")
            self._grbl.jog("Y", -d * backoff_mm, Y_HOME_BACKOFF_FEED)
            self._grbl.poll_until_pin_clear("Y")
            logger.info("Pre-check backoff complete")

        # --- Phase 1: Approach ---
        logger.info(
            "Approach: jogging Y %.2fmm @ %.0f mm/min",
            approach_mm,
            Y_HOME_APPROACH_FEED,
        )
        hit = self._grbl.jog_until_pin_active(
            axis="Y",
            distance_mm=d * approach_mm,
            feed=Y_HOME_APPROACH_FEED,
            pin="Y",
        )
        if not hit:
            raise RuntimeError(
                "Y home switch not found within max travel - "
                "check switch wiring and direction."
            )
        logger.info("Approach: switch hit")

        # --- Phase 2: Backoff ---
        logger.info(
            "Backoff: jogging Y %.2fmm @ %.0f mm/min",
            backoff_mm,
            Y_HOME_BACKOFF_FEED,
        )
        self._grbl.jog("Y", -d * backoff_mm, Y_HOME_BACKOFF_FEED)
        self._grbl.poll_until_pin_clear("Y")
        logger.info("Backoff: switch released")

        # --- Phase 3: Zero offset ---
        logger.info("Zero offset: moving Y %.2fmm away from switch", offset_mm)
        self._grbl.jog("Y", -d * offset_mm, Y_HOME_BACKOFF_FEED)
        self._grbl.wait_for_idle()
        time.sleep(Y_HOME_SETTLE_SECS)

        # --- Phase 4: Re-enable hard limits and set home ---
        self._grbl.set_hard_limits(True)
        logger.info("Setting G92 Y0 at backed-off home position")
        self._grbl.send_code("G92 Y0", wait=0.0)
        self._pos["Y"] = 0

    def home_all_axes(self, home_y: bool = True) -> None:
        """
        Return the robot to a fully known home state.

        If home_y is True, homes Y on the physical limit switch first
        (sets repeatable G92 Y0), then soft-homes X and Z.
        If home_y is False, soft-homes all three axes.

        This is the recommended entry point for all shot sequences - it
        ensures a consistent starting state regardless of where the axes
        currently are.
        """
        if home_y:
            logger.info("home_all_axes(): homing Y on physical switch")
            self.home_y_on_switch()
        else:
            self.reset("Y")
        logger.info("home_all_axes(): resetting X and Z to soft home")
        self.reset("X")
        self.reset("Z")

    # ------------------------------------------------------------------
    # Homing (soft - no limit switches)
    # ------------------------------------------------------------------

    def set_home(self) -> None:
        """Send G92 X0 Y0 Z0 to set the current position as the coordinate origin. No motors move."""
        logger.info("set_home(): zeroing coordinate system at current position (G92)")
        self._grbl.send_code("G92 X0 Y0 Z0", wait=0.0)
        self._pos = {"X": 0, "Y": 0, "Z": 0}

    def reset(self, axis: str) -> None:
        """Return a single axis to its soft-home position (coordinate 0) using G90 then back to G91."""
        axis = axis.upper()
        if axis not in VALID_AXES:
            raise ValueError(f"Invalid axis '{axis}'. Must be one of {VALID_AXES}.")
        feed = self._feed_for(axis)
        logger.info("reset(%s): returning to soft home @ %.0f mm/min", axis, feed)
        self._grbl.send_code("G90", wait=0.0)
        self._grbl.send_code(f"G01 {axis}0 F{feed:.0f}", wait=0.0)
        self._grbl.send_code("G91", wait=0.0)
        self._pos[axis] = 0

    def home(self) -> None:
        """Return all axes to their soft-home positions using G90 then back to G91."""
        logger.info("home(): returning all axes to soft home")
        self._grbl.send_code("G90", wait=0.0)
        for axis in VALID_AXES:
            feed = self._feed_for(axis)
            self._grbl.send_code(f"G01 {axis}0 F{feed:.0f}", wait=0.0)
        self._grbl.send_code("G91", wait=0.0)
        self._pos = {"X": 0, "Y": 0, "Z": 0}
