"""
vision/stream.py - Frame streaming between Pi and server.

Two classes:

  StreamSender   (runs on Pi)
    Captures frames from the Pi camera and sends them over a TCP socket
    to the server. trigger() is called from inside robot.fire() at the
    moment the Z pullback command is sent - StreamSender then connects to
    the server, waits launch_offset seconds for the ball to physically
    release, and captures for capture_secs.

  StreamReceiver (runs on server)
    Listens on a TCP port, receives length-prefixed JPEG frames, decodes
    them into numpy arrays, and returns the full frame buffer once the
    sender closes the connection.

Wire protocol
-------------
Each frame is sent as:
  [4 bytes big-endian uint32: payload length] [N bytes: JPEG data]

The receiver reads until the socket closes (sender disconnects after
capture_secs have elapsed).
"""

from __future__ import annotations

import logging
import socket
import struct
import threading
import time
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Default stream settings
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5555
DEFAULT_LAUNCH_OFFSET = 24.0  # seconds from Z pullback command to ball release
DEFAULT_CAPTURE_SEC = 3.0     # seconds of video to capture after ball launches
DEFAULT_FPS = 60              # frames per second - 60fps gives ~6 ball frames vs ~3 at 30fps
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480


# ---------------------------------------------------------------------------
# StreamSender - runs on Pi
# ---------------------------------------------------------------------------


class StreamSender:
    """
    Captures frames from the Pi camera and streams them to the server.

    Designed to be triggered from inside robot.fire() at the exact moment
    the Z pullback command is sent. trigger() connects to the server and
    starts a background thread that waits launch_offset seconds (for the
    ball to physically release) then captures for capture_secs.

    Typical usage in robot.fire():
        self.move_steps("Z", -FIRE_STEPS)   # engage
        if stream_sender:
            stream_sender.trigger()          # connect + start countdown
        self.move_steps("Z", FIRE_STEPS)    # pull back

    Parameters
    ----------
    host : str
        Tailscale hostname or IP of the server running StreamReceiver.
    port : int
        TCP port to connect to.
    launch_offset : float
        Seconds from when trigger() is called (Z pullback command sent)
        to when the ball physically leaves the launcher. Capture begins
        after this delay.
    capture_secs : float
        How many seconds of video to capture after launch_offset elapses.
    fps : int
        Camera frame rate. Capture runs in a tight loop - the camera
        hardware throttles to this rate via the FrameRate control.
    width : int
        Capture width in pixels.
    height : int
        Capture height in pixels.
    """

    def __init__(
        self,
        host: str,
        port: int = DEFAULT_PORT,
        launch_offset: float = DEFAULT_LAUNCH_OFFSET,
        capture_secs: float = DEFAULT_CAPTURE_SEC,
        fps: int = DEFAULT_FPS,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
    ) -> None:
        self.host = host
        self.port = port
        self.launch_offset = launch_offset
        self.capture_secs = capture_secs
        self.fps = fps
        self.width = width
        self.height = height
        self._thread: Optional[threading.Thread] = None

    def trigger(self) -> None:
        """
        Connect to the server and start the capture countdown.

        Call this from robot.fire() immediately before the Z pullback
        command is sent. Returns immediately - capture runs in a background
        thread.
        """
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.connect((self.host, self.port))
            logger.info("StreamSender: connected to %s:%d", self.host, self.port)
        except OSError as e:
            logger.error(
                "StreamSender: could not connect to %s:%d - %s", self.host, self.port, e
            )
            self._sock = None
            return

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(
            "StreamSender: triggered - launch_offset=%.1fs capture=%.1fs",
            self.launch_offset,
            self.capture_secs,
        )

    def send_frame(self, frame: np.ndarray) -> None:
        """
        Send a single pre-captured frame to the server and close.

        Used for capture-only (synthetic shot) mode: the robot has moved to
        position and captured a still without firing. Connects to the server,
        sends one JPEG frame using the same length-prefixed wire protocol as
        _capture_and_send(), then closes. No launch_offset wait, no camera
        acquisition - the caller provides the frame directly.

        The StreamReceiver handles this transparently since it reads until
        the connection closes - one frame is a valid stream.
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.host, self.port))
            logger.info("StreamSender.send_frame: connected to %s:%d", self.host, self.port)
        except OSError as e:
            logger.error(
                "StreamSender.send_frame: could not connect to %s:%d - %s",
                self.host, self.port, e,
            )
            return

        try:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                logger.error("StreamSender.send_frame: JPEG encode failed")
                return
            data = buf.tobytes()
            sock.sendall(struct.pack(">I", len(data)) + data)
            logger.info("StreamSender.send_frame: sent 1 frame (%d bytes)", len(data))
        finally:
            sock.close()

    def wait(self) -> None:
        """Block until the stream has finished."""
        if self._thread:
            self._thread.join()

    def _run(self) -> None:
        """Background thread: wait for ball launch, capture, stream, close."""
        if not self._sock:
            return

        logger.debug(
            "StreamSender: waiting %.1fs for ball to launch", self.launch_offset
        )
        time.sleep(self.launch_offset)

        # Capture and send frames
        try:
            self._capture_and_send(self._sock)
        finally:
            self._sock.close()
            logger.info("StreamSender: stream complete")

    def _capture_and_send(self, sock: socket.socket) -> None:
        """Capture frames from picamera2 and send over socket."""
        try:
            from picamera2 import Picamera2
        except ImportError:
            logger.error("StreamSender: picamera2 not available")
            return

        cam = Picamera2()
        config = cam.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            controls={"FrameRate": self.fps},
        )
        cam.configure(config)
        cam.start()

        end_time = time.monotonic() + self.capture_secs
        sent = 0

        try:
            while time.monotonic() < end_time:
                # picamera2 "RGB888" format is BGR in memory layout (matches
                # OpenCV convention) despite the misleading libcamera name.
                # No colour conversion needed - capture_array() is already BGR.
                bgr = cam.capture_array()

                # Encode as JPEG
                ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    continue

                data = buf.tobytes()

                # Send: 4-byte length prefix + JPEG payload
                sock.sendall(struct.pack(">I", len(data)) + data)
                sent += 1

        finally:
            cam.stop()
            cam.close()
            logger.info("StreamSender: sent %d frames", sent)


# ---------------------------------------------------------------------------
# StreamReceiver - runs on server
# ---------------------------------------------------------------------------


class StreamReceiver:
    """
    Receives JPEG frames from a StreamSender over TCP and returns them
    as a list of numpy arrays.

    Parameters
    ----------
    port : int
        TCP port to listen on.
    max_frames : int
        Maximum number of frames to buffer (prevents runaway memory use).
    """

    def __init__(self, port: int = DEFAULT_PORT, max_frames: int = 300) -> None:
        self.port = port
        self.max_frames = max_frames
        self._frames: list[np.ndarray] = []
        self._server_sock: Optional[socket.socket] = None

    def start_listening(self) -> None:
        """Bind and listen. Call before triggering the shot on the Pi."""
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((DEFAULT_HOST, self.port))
        self._server_sock.listen(1)
        logger.info("StreamReceiver: listening on port %d", self.port)

    def receive(self, timeout: float = 30.0) -> list[np.ndarray]:
        """
        Accept one connection and receive all frames until sender disconnects.

        Parameters
        ----------
        timeout : float
            How long to wait for a connection before giving up (seconds).

        Returns
        -------
        list of np.ndarray
            Decoded BGR frames in capture order.
        """
        if self._server_sock is None:
            raise RuntimeError("Call start_listening() before receive()")

        self._server_sock.settimeout(timeout)
        try:
            conn, addr = self._server_sock.accept()
        except socket.timeout:
            logger.warning("StreamReceiver: timed out waiting for connection")
            return []

        logger.info("StreamReceiver: connection from %s", addr)
        self._frames = []

        try:
            conn.settimeout(35.0)  # per-read timeout - must cover launch_offset wait
            while len(self._frames) < self.max_frames:
                # Read 4-byte length prefix
                header = self._recvall(conn, 4)
                if not header:
                    break  # sender closed connection
                length = struct.unpack(">I", header)[0]

                # Read JPEG payload
                data = self._recvall(conn, length)
                if not data:
                    break

                # Decode JPEG → numpy BGR
                arr = np.frombuffer(data, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    self._frames.append(frame)

        except (socket.timeout, ConnectionResetError) as e:
            logger.warning("StreamReceiver: connection ended: %s", e)
        finally:
            conn.close()
            self._server_sock.close()
            self._server_sock = None

        logger.info("StreamReceiver: received %d frames", len(self._frames))
        return self._frames

    @staticmethod
    def _recvall(sock: socket.socket, n: int) -> bytes:
        """Read exactly n bytes from socket, or return b'' on close."""
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                return b""
            buf += chunk
        return buf
