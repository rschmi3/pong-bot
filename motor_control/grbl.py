"""
grbl.py - Low-level GRBL serial interface.

Handles opening the serial port, sending GCode commands, and waiting for
the GRBL 'ok' response. Supports a dry_run mode that logs GCode to stdout
instead of sending over serial - useful for development without hardware.
"""

from __future__ import annotations

import logging
import time

import serial

logger = logging.getLogger(__name__)


class GrblInterface:
    """
    Low-level interface to a GRBL-firmware Arduino over serial.

    Parameters
    ----------
    port : str
        Serial port device, e.g. '/dev/ttyUSB0'.
    baud : int
        Baud rate - must match GRBL firmware setting (default 115200).
    dry_run : bool
        If True, GCode is logged to stdout instead of sent over serial.
        Safe to use without hardware connected.
    """

    GRBL_BOOT_DELAY = 2.0  # seconds to wait for GRBL to boot after open
    OK_RESPONSE = b"ok"  # GRBL success response token

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baud: int = 115200,
        dry_run: bool = False,
        motion_init: bool = True,
    ) -> None:
        self.port = port
        self.baud = baud
        self.dry_run = dry_run
        self.motion_init = motion_init
        self._serial: serial.Serial | None = None

        if not dry_run:
            logger.info("Opening serial port %s at %d baud", port, baud)
            self._serial = serial.Serial(
                port,
                baud,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=10,
            )
            logger.info("Waiting %.1fs for GRBL to boot...", self.GRBL_BOOT_DELAY)
            time.sleep(self.GRBL_BOOT_DELAY)

            # Flush any startup message GRBL sends (e.g. "Grbl 1.1f ['$' for help]")
            self._serial.reset_input_buffer()
            logger.info("Serial ready")

            # Enable pin state reporting in '?' status responses ($10 bit 4).
            # Without this, Pn:Y never appears and limit-switch polling is blind.
            # $10=17 = MPos (bit 0) + Pin state (bit 4).
            self._serial.write(b"$10=17\n")
            self._serial.flush()
            time.sleep(0.1)

            self._serial.reset_input_buffer()
            logger.info("Pin state reporting enabled ($10=17)")
        else:
            logger.info("DRY RUN mode - GCode will be printed, not sent")

        if self.motion_init:
            self._init_grbl()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> GrblInterface:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the serial port if open."""
        if self._serial is not None and self._serial.is_open:
            self._serial.close()
            logger.info("Serial port closed")

    # ------------------------------------------------------------------
    # Core send / query
    # ------------------------------------------------------------------

    def send_code(self, gcode: str, wait: float = 1.0) -> None:
        """
        Send a single GCode command and block until GRBL responds with 'ok'.

        Parameters
        ----------
        gcode : str
            GCode string to send, e.g. 'G91' or 'G01 X1.25'.
            A trailing newline is added automatically if not present.
        wait : float
            Extra seconds to sleep after receiving 'ok'. Use this to give
            the motion time to complete before sending the next command.
        """
        gcode = gcode.rstrip("\n") + "\n"

        if self.dry_run:
            logger.info("DRY RUN >> %s", gcode.rstrip())
            return

        if self._serial is None:
            raise RuntimeError("Serial port is not open")

        logger.debug("Sending: %s", gcode.rstrip())
        self._serial.write(gcode.encode())
        self._serial.flush()

        # Block until we receive a line containing 'ok' (or 'error'/'ALARM').
        # Use startswith for ALARM to avoid false matches on comments/messages.
        deadline = time.monotonic() + 15.0
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Timed out waiting for GRBL response to '{gcode.rstrip()}'"
                )
            line = self._serial.readline().strip()
            if not line:
                continue

            logger.debug("GRBL << %s", line.decode(errors="replace"))

            if self.OK_RESPONSE in line:
                break

            if line.startswith(b"ALARM"):
                raise RuntimeError(
                    f"GRBL alarm response to '{gcode.rstrip()}': "
                    f"{line.decode(errors='replace')}"
                )

            if b"error" in line:
                raise RuntimeError(
                    f"GRBL error response to '{gcode.rstrip()}': "
                    f"{line.decode(errors='replace')}"
                )

        if wait > 0:
            time.sleep(wait)

    def query_status(self) -> str:
        """
        Poll GRBL status using the real-time '?' command.

        Returns a single status line such as:
            <Idle|MPos:0.000,0.000,0.000|Pn:Y>

        Unlike '$' queries, real-time status reports do not end with 'ok'.
        """
        if self.dry_run:
            logger.info("DRY RUN query >> ?")
            return "<Idle|MPos:0.000,0.000,0.000>"

        if self._serial is None:
            raise RuntimeError("Serial port is not open")

        # Send real-time '?' - do NOT flush the input buffer first.
        # Flushing discards ok/alarm responses from prior commands and races
        # with GRBL's recovery after alarm. Instead, read and discard any
        # non-status lines until we see a <...> status response.
        self._serial.write(b"?")
        self._serial.flush()

        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline:
            line = self._serial.readline().strip()
            if not line:
                continue
            decoded = line.decode(errors="replace")
            logger.debug("GRBL << %s", decoded)
            if decoded.startswith("<") and decoded.endswith(">"):
                return decoded
            # Silently discard ok/error/alarm lines - they are stale responses
            # from prior commands and will not match the status format.

        raise TimeoutError("Timed out waiting for GRBL status response to '?'")

    @staticmethod
    def parse_status(status: str) -> dict[str, str]:
        """Parse a GRBL status line into a dict of fields."""
        status = status.strip()
        if not (status.startswith("<") and status.endswith(">")):
            raise ValueError(f"Not a GRBL status line: {status!r}")

        parts = status[1:-1].split("|")
        parsed = {"state": parts[0]}
        for part in parts[1:]:
            if ":" in part:
                key, value = part.split(":", 1)
                parsed[key] = value
            else:
                parsed[part] = ""
        return parsed

    def limit_pins(self) -> set[str]:
        """
        Return the set of active pin letters from the GRBL status report.

        Most controllers encode limit pins in the Pn field, e.g. Pn:Y.
        """
        status = self.query_status()
        parsed = self.parse_status(status)
        return set(parsed.get("Pn", ""))

    def query_steps_per_mm(self) -> dict[str, float]:
        """
        Query GRBL $100/$101/$102 settings and return per-axis steps/mm.

        Returns
        -------
        dict[str, float]
            Mapping like {"X": 800.0, "Y": 800.0, "Z": 800.0}.
        """
        if self.dry_run:
            logger.info("DRY RUN query >> $$ (steps/mm)")
            return {"X": 800.0, "Y": 800.0, "Z": 800.0}

        response = self.query("$$")
        mapping = {"$100": "X", "$101": "Y", "$102": "Z"}
        result: dict[str, float] = {}

        for line in response.splitlines():
            for setting, axis in mapping.items():
                prefix = setting + "="
                if line.startswith(prefix):
                    result[axis] = float(line.split("=", 1)[1].strip())

        if len(result) != 3:
            raise RuntimeError(
                f"Could not parse GRBL steps/mm settings from $$ response: {result}"
            )

        return result

    def wait_for_idle(self, timeout: float = 30.0, poll_interval: float = 0.05) -> str:
        """
        Poll GRBL status until the controller reports Idle.

        Returns the final status line. This is useful after incremental moves
        where an 'ok' response alone is not enough to guarantee the mechanism
        has finished moving.
        """
        if self.dry_run:
            logger.info("DRY RUN wait_for_idle()")
            return "<Idle|MPos:0.000,0.000,0.000>"

        deadline = time.monotonic() + timeout
        last_status = ""
        while time.monotonic() < deadline:
            last_status = self.query_status()
            parsed = self.parse_status(last_status)
            state = parsed.get("state", "")
            if state == "Idle":
                return last_status
            if state == "Alarm":
                raise RuntimeError(
                    f"GRBL entered Alarm state while waiting for idle: {last_status}"
                )
            time.sleep(poll_interval)

        raise TimeoutError(
            f"Timed out waiting for GRBL to become Idle. Last status: {last_status!r}"
        )

    def query(self, cmd: str, timeout: float = 10.0) -> str:
        """
        Send a read-only GRBL '$' query command and return the full response.

        Safe to call without dry_run - does not move any motors.

        Parameters
        ----------
        cmd : str
            GRBL query command, e.g. '$I' or '$$'.
        timeout : float
            Seconds to wait for the response before raising TimeoutError.
        """
        cmd = cmd.rstrip("\n") + "\n"

        if self.dry_run:
            logger.info("DRY RUN query >> %s", cmd.rstrip())
            return "(dry run - no response)"

        if self._serial is None:
            raise RuntimeError("Serial port is not open")

        logger.debug("Query: %s", cmd.rstrip())
        self._serial.reset_input_buffer()
        self._serial.write(cmd.encode())
        self._serial.flush()

        lines = []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self._serial.readline().strip()
            if not line:
                continue
            decoded = line.decode(errors="replace")
            logger.debug("GRBL << %s", decoded)
            lines.append(decoded)
            if self.OK_RESPONSE in line or b"error" in line:
                return "\n".join(lines)

        raise TimeoutError(
            f"Timed out waiting for GRBL query response to '{cmd.rstrip()}'"
        )

    # ------------------------------------------------------------------
    # Jog / homing helpers
    # ------------------------------------------------------------------

    def jog(self, axis: str, distance_mm: float, feed: float) -> None:
        """
        Send a single GRBL jog command and wait for the 'ok' acknowledgment.

        'ok' means the jog was accepted into the planner - NOT that motion
        has completed. Call wait_for_idle() or jog_until_pin_active() after.
        """
        cmd = f"$J=G91 {axis}{distance_mm:.6f} F{feed:.0f}"
        logger.info("JOG %s", cmd)
        self.send_code(cmd)

    def jog_cancel(self) -> None:
        """
        Send the real-time jog-cancel byte (0x85) and wait for Idle.

        Cancels an in-progress jog immediately and drains the planner.
        """
        if self.dry_run:
            logger.info("DRY RUN jog_cancel()")
            return
        if self._serial is None:
            raise RuntimeError("Serial port is not open")

        logger.info("Sending jog cancel (0x85)")
        self._serial.write(b"\x85")
        self._serial.flush()
        self.wait_for_idle()
        # Allow physical deceleration to complete
        # GRBL reports idle before the motor has fully stopped
        time.sleep(0.3)

    def jog_until_pin_active(
        self,
        axis: str,
        distance_mm: float,
        feed: float,
        pin: str,
        timeout: float = 60.0,
    ) -> bool:
        """
        Start a jog move and poll '?' until the given pin becomes active.

        When the pin is seen, immediately sends jog-cancel (0x85) and waits
        for Idle. Returns True if the pin was triggered, False if the jog
        completed without the pin ever activating (switch not found).

        Hard limits must be disabled ($21=0) before calling this - the
        approach relies entirely on polling Pn: in the status report.
        Requires $10 to include pin state (bit 4), set in __init__().

        No sleep between polls query_status() blocks on serial readline
        """
        if self.dry_run:
            logger.info(
                "DRY RUN jog_until_pin_active(%s, %.2fmm, %.0f, pin=%s)",
                axis,
                distance_mm,
                feed,
                pin,
            )
            return True

        self.jog(axis, distance_mm, feed)

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.query_status()
            parsed = self.parse_status(status)
            state = parsed.get("state", "")
            pins = set(parsed.get("Pn", ""))

            logger.debug("jog_until_pin_active: state=%s pins=%s", state, pins)

            if pin in pins:
                logger.info("Pin %s active - cancelling jog", pin)
                self.jog_cancel()
                return True

            if state == "Idle":
                # Jog finished without the pin activating.
                return False

        # Timeout - cancel whatever is running and raise.
        self.jog_cancel()
        raise TimeoutError(
            f"jog_until_pin_active() timed out after {timeout:.0f}s "
            f"waiting for pin '{pin}'"
        )

    def poll_until_pin_clear(
        self,
        pin: str,
        timeout: float = 30.0,
    ) -> None:
        """
        Poll '?' until the given pin (e.g. 'Y') is no longer active.

        Used during backoff to detect when the switch releases.
        Sends jog-cancel and waits for Idle once the pin clears.
        """
        if self.dry_run:
            logger.info("DRY RUN poll_until_pin_clear(%s)", pin)
            return

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.query_status()
            parsed = self.parse_status(status)
            pins = set(parsed.get("Pn", ""))
            logger.debug(
                "poll_until_pin_clear: pins=%s state=%s", pins, parsed.get("state")
            )
            if pin not in pins:
                logger.info("Pin %s cleared - cancelling jog", pin)
                self.jog_cancel()
                return

        raise TimeoutError(
            f"Timed out waiting for pin '{pin}' to clear after {timeout:.0f}s"
        )

    def set_hard_limits(self, enabled: bool) -> None:
        """Enable ($21=1) or disable ($21=0) GRBL hard limits."""
        val = 1 if enabled else 0
        logger.info("Setting hard limits $21=%d", val)
        self.send_code(f"$21={val}")

    def recover_from_alarm(
        self,
        axis: str = "Y",
        jog_mm: float = 2.5,
        feed: float = 100.0,
    ) -> None:
        """
        Recover GRBL from a hard-limit alarm state.

        After a hard limit trips, GRBL locks out all commands. This method:
          1. Disables hard limits ($21=0) so $X can actually clear the alarm.
          2. Sends $X raw (no 'ok' wait - GRBL ignores the ok in alarm state).
          3. Restores relative mode (G91).
          4. Jogs the given axis away from the switch by jog_mm at feed mm/min.
          5. Re-enables hard limits ($21=1).

        Must be called with motion_init=False (the normal GrblInterface init
        would hang waiting for 'ok' in alarm state).

        Parameters
        ----------
        axis : str
            Axis to jog away from the limit switch (default 'Y').
        jog_mm : float
            Distance in mm to jog away from the switch (default 2.5mm).
        feed : float
            Feed rate in mm/min for the recovery jog (default 100).
        """
        if self.dry_run:
            logger.info(
                "DRY RUN recover_from_alarm(axis=%s, jog_mm=%.2f, feed=%.0f)",
                axis,
                jog_mm,
                feed,
            )
            return

        if self._serial is None:
            raise RuntimeError("Serial port is not open")

        move_duration = (jog_mm / feed) * 60.0  # seconds

        logger.info("Recovery: disabling hard limits ($21=0)")
        self._serial.write(b"$21=0\n")
        self._serial.flush()
        time.sleep(0.3)

        logger.info("Recovery: clearing alarm ($X)")
        self._serial.write(b"$X\n")
        self._serial.flush()
        time.sleep(0.3)

        status = self.query_status()
        logger.info("Recovery: status after $X: %s", status)

        logger.info("Recovery: restoring relative mode (G91)")
        self._serial.write(b"G91\n")
        self._serial.flush()
        time.sleep(0.1)

        logger.info(
            "Recovery: jogging %s+%.2fmm at F%.0f (est. %.1fs)",
            axis,
            jog_mm,
            feed,
            move_duration,
        )
        cmd = f"G01 {axis}{jog_mm:.3f} F{feed:.0f}\n"
        self._serial.write(cmd.encode())
        self._serial.flush()
        time.sleep(move_duration + 0.5)

        logger.info("Recovery: re-enabling hard limits ($21=1)")
        self._serial.write(b"$21=1\n")
        self._serial.flush()
        time.sleep(0.1)

        status = self.query_status()
        logger.info("Recovery complete. Final status: %s", status)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _init_grbl(self) -> None:
        """Send safe initialisation commands - no motion, no spindle."""
        logger.info("Initialising GRBL (relative mode)")
        # G91: relative positioning - every move is an increment from the
        # current position rather than an absolute coordinate. This means
        # repeated calls to move_steps(X, 100) each move 100 steps forward.
        self.send_code("G91", wait=0.0)
