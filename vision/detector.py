"""
vision/detector.py - ONNX-based end-to-end aim predictor for Pi inference.

Loads the exported aim_model.onnx (full model: backbone + aim head) via
cv2.dnn and provides both detection and aim prediction in a single
inference call. No torch dependency - runs entirely on CPU via OpenCV.

The ONNX model has named outputs:
    "features" - [B, 512] dual-path features (detected cell + global pool)
    "boxes"    - [B, 4]   (cx, cy, w, h) normalised bounding box
    "scores"   - [B]      detection confidence
    "steps"    - [B, 2]   normalised aim prediction in [-1, 1]
    "raw_grid" - [B, 5, H, W] raw grid output (not used at inference)

Step values are denormalised by multiplying by NORMALISE_X / NORMALISE_Y
(from vision.models.pong_model) before returning to the caller.

Usage
-----
    from vision.detector import AimPredictor

    predictor = AimPredictor("models/aim_model.onnx")
    frame     = ...  # numpy BGR image from picamera2

    x_steps, y_steps = predictor.predict(frame)
    bbox = predictor.detect(frame)  # for debugging / visualisation
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from vision.transforms import preprocess

logger = logging.getLogger(__name__)


class AimPredictor:
    """
    End-to-end ONNX model: image → detection + aim prediction.

    Wraps a cv2.dnn network loaded from the exported aim_model.onnx file.
    Designed to run on the Pi without any torch dependency.

    Parameters
    ----------
    model_path : str
        Path to the aim_model.onnx file exported by train_head.py.
    score_threshold : float
        Minimum detection confidence to accept a prediction. Below this
        threshold, predict() returns None. Default 0.5.
    """

    def __init__(self, model_path: str, score_threshold: float = 0.5) -> None:
        self.score_threshold = score_threshold
        logger.info("Loading aim model from %s", model_path)
        self._net = cv2.dnn.readNetFromONNX(model_path)
        logger.info("Aim model ready")

    def _forward(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        """
        Run a single forward pass and return all outputs.

        Returns dict with keys "boxes", "scores", "steps" as numpy arrays.
        """
        chw = preprocess(frame)  # [3, 640, 640] float32
        blob = chw[np.newaxis, ...]  # [1, 3, 640, 640]
        self._net.setInput(blob)

        output_names = self._net.getUnconnectedOutLayersNames()
        outputs = self._net.forward(output_names)
        return dict(zip(output_names, outputs))

    def predict_and_detect(
        self,
        frame: np.ndarray,
    ) -> tuple[
        tuple[int, int] | None,
        tuple[float, float, float, float, float] | None,
        np.ndarray | None,
    ]:
        """
        Run a single forward pass and return aim prediction, bbox, and raw grid.

        Prefer this over calling predict() and detect() separately - it runs
        the ONNX model exactly once.

        Parameters
        ----------
        frame : np.ndarray
            Raw BGR image from the camera (any resolution).

        Returns
        -------
        steps    : tuple (x_steps, y_steps) as integers, or None if no cup
                   detected above the score threshold.
        bbox     : tuple (cx, cy, w, h, score) normalised [0, 1], or None if
                   below score threshold.
        raw_grid : np.ndarray [1, 5, 80, 80] raw detection head output, or
                   None if not present in the ONNX outputs. Channel 4 is the
                   raw score logit (apply sigmoid to get confidence per cell).
                   Useful for score heatmap visualisation without torch.
        """
        from utils.data_dir import NORMALISE_X, NORMALISE_Y

        out = self._forward(frame)

        scores = out.get("scores")
        steps = out.get("steps")
        boxes = out.get("boxes")
        raw_grid = out.get("raw_grid")  # [1, 5, 80, 80] or None

        if scores is None:
            logger.warning("Model output missing 'scores'")
            return None, None, raw_grid

        score = float(scores.flatten()[0])
        if score < self.score_threshold:
            logger.debug(
                "Detection score %.3f below threshold %.3f",
                score,
                self.score_threshold,
            )
            return None, None, raw_grid

        # Aim prediction
        aim: tuple[int, int] | None = None
        if steps is not None:
            raw_x, raw_y = steps.flatten()[:2]
            x_steps = raw_x * NORMALISE_X
            y_steps = raw_y * NORMALISE_Y
            aim = (int(round(x_steps)), int(round(y_steps)))
            logger.debug(
                "Aim prediction: x_steps=%+d y_steps=%+d (score=%.3f, raw=%.3f,%.3f)",
                aim[0],
                aim[1],
                score,
                raw_x,
                raw_y,
            )
        else:
            logger.warning("Model output missing 'steps'")

        # Bounding box
        bbox: tuple[float, float, float, float, float] | None = None
        if boxes is not None:
            cx, cy, w, h = boxes.flatten()[:4]
            bbox = (float(cx), float(cy), float(w), float(h), score)

        return aim, bbox, raw_grid

    def predict(self, frame: np.ndarray) -> tuple[int, int] | None:
        """
        Predict aim step values from a camera frame.

        Prefer predict_and_detect() when you also need the bounding box - it
        avoids running two separate forward passes.

        Parameters
        ----------
        frame : np.ndarray
            Raw BGR image from the camera (any resolution).

        Returns
        -------
        tuple (x_steps, y_steps) as integers, or None if no cup detected
        above the score threshold.
        """
        aim, _, _ = self.predict_and_detect(frame)
        return aim

    def detect(
        self,
        frame: np.ndarray,
    ) -> tuple[float, float, float, float, float] | None:
        """
        Run detection and return the bounding box (for debugging/visualisation).

        Prefer predict_and_detect() when you also need aim steps - it avoids
        running two separate forward passes.

        Parameters
        ----------
        frame : np.ndarray
            Raw BGR image from the camera (any resolution).

        Returns
        -------
        tuple (cx, cy, w, h, score) or None if below score threshold.
        All bbox values are normalised to [0, 1].
        """
        _, bbox, _ = self.predict_and_detect(frame)
        return bbox
