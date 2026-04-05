"""
vision/models - Model definitions for Pong-Bot.

Exports
-------
DetectorModel    Abstract base class for object detection backbones.
AimModel         Abstract base class for full end-to-end aim models.
PongDetector     From-scratch CNN detector with SE attention + grid head.
PongAimModelV1   Aim model v1: 516-dim combined head (baseline).
PongAimModelV2   Aim model v2: 518-dim combined head (+ derived bbox features).
PongAimModelV3   Aim model v3: split X/Y heads (513/515-dim).
"""

from .base import AimModel, DetectorModel
from .pong_model import PongAimModelV1, PongAimModelV2, PongAimModelV3, PongDetector

__all__ = [
    "DetectorModel",
    "AimModel",
    "PongDetector",
    "PongAimModelV1",
    "PongAimModelV2",
    "PongAimModelV3",
]
