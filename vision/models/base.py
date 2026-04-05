"""
vision/models/base.py - Abstract base classes for the two-phase training pipeline.

Two base classes:

  DetectorModel   - Phase 1: trained on labelled bounding box images.
                    Subclass this to define your own detector.

  AimModel        - Phase 2: full end-to-end image → step values model.
                    Composed of a DetectorModel backbone (frozen) + a dense
                    prediction head. Subclass this to define your own head.

Training flow
-------------
Phase 1:
    detector = MyDetector()
    # train with train_detector.py
    # save to models/cup_detector.pt

Phase 2:
    model = MyAimModel(backbone=MyDetector())
    model.load_backbone("models/cup_detector.pt")
    model.freeze_backbone()
    # train head only with train_head.py
    # export full end-to-end model to models/aim_model.onnx
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class DetectorModel(nn.Module, ABC):
    """Abstract base class for the object detection backbone. forward() must return boxes [B,4] and scores [B]."""

    @abstractmethod
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Accept Tensor[B, 3, H, W], return dict with 'boxes' [B,4] and 'scores' [B]."""
        ...

    @abstractmethod
    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        """Run the encoder trunk. Returns spatial feature map [B, C, H, W]."""
        ...


class AimModel(nn.Module, ABC):
    """Abstract base class for the full end-to-end image → step values model (backbone + head)."""

    def __init__(self, backbone: DetectorModel) -> None:
        super().__init__()
        self.backbone = backbone

    @abstractmethod
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        End-to-end forward pass: image → detection + aim prediction.

        Parameters
        ----------
        images : Tensor[B, 3, H, W]
            Batch of preprocessed images.

        Returns
        -------
        dict with keys:
            "features"  : Tensor[B, D]       - features for aim head
            "boxes"     : Tensor[B, 4]       - best (cx, cy, w, h) normalised 0–1
            "scores"    : Tensor[B]          - best detection confidence 0–1
            "steps"     : Tensor[B, 2]       - predicted step values (normalised)
            "raw_grid"  : Tensor[B, 5, H, W] - raw grid output (training only)
        """
        ...

    # ------------------------------------------------------------------
    # Backbone weight management
    # ------------------------------------------------------------------

    def load_backbone(self, path: str) -> None:
        """Load phase 1 detector weights into self.backbone."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        # Support both raw state_dict and checkpoint dicts
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        # strict=False: newly added register_buffer entries (e.g. grid_col/grid_row)
        # are correctly initialised by __init__ and won't be in older checkpoints.
        self.backbone.load_state_dict(state, strict=False)

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (requires_grad=False) for phase 2 head-only training."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters for end-to-end fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.backbone.train()

    def head_parameters(self) -> list[nn.Parameter]:
        """Return only the head parameters - useful for building an optimiser
        that only updates the head during phase 2."""
        backbone_ids = {id(p) for p in self.backbone.parameters()}
        return [p for p in self.parameters() if id(p) not in backbone_ids]
