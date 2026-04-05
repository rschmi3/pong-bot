"""
vision/models/pong_model.py - PongDetector and PongAimModel V1/V2/V3 implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.data_dir import NORMALISE_X, NORMALISE_Y  # noqa: F401 - re-exported

from .base import AimModel, DetectorModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Encoder output channels (last residual block).
ENCODER_DIM = 256

# Aim head input dimension: detected cell features + global pool features.
AIM_INPUT_DIM = ENCODER_DIM * 2  # 512

# Full aim head input: backbone features + bbox (cx, cy, w, h) + derived features
# (aspect ratio h/w, area w*h). cx is the strongest predictor of x_steps (r2=0.98);
# bbox h and the derived aspect/area features encode cup distance/depth for y_steps
# more discriminatively than cy alone (cy varies only 1.5% across Y range at
# typical camera angles, while h varies 40%).
AIM_HEAD_V1_INPUT_DIM = AIM_INPUT_DIM + 4  # 516 = 512 + 4 bbox (v1 baseline)
AIM_HEAD_INPUT_DIM = AIM_INPUT_DIM + 6  # 518 = 512 + 4 bbox + 2 derived (v2)

# Split aim head input dimensions (PongAimModelV3)
# X head: features + bbox_cx only (r2=0.98 with x_steps)
# Y head: features + bbox_h + aspect(h/w) + area(w*h)
#   bbox_h varies 40% across Y range vs 1.5% for bbox_cy
SPLIT_X_INPUT_DIM = AIM_INPUT_DIM + 1  # 513: features(512) + bbox_cx(1)
SPLIT_Y_INPUT_DIM = (
    AIM_INPUT_DIM + 3
)  # 515: features(512) + bbox_h(1) + aspect(1) + area(1)

# Default anchor priors for box size decoding (w, h as fraction of image).
# These are overridden by auto-anchor computation during training.
DEFAULT_ANCHOR_W = 0.055
DEFAULT_ANCHOR_H = 0.116


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention (Hu et al., 2018).

    Learns per-channel importance weights via global pooling → bottleneck
    MLP → sigmoid gating. Applied as a multiplicative mask on the input
    feature map.
    """

    def __init__(self, channels: int, ratio: int = 16) -> None:
        super().__init__()
        mid = max(channels // ratio, 4)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        w = self.squeeze(x).view(B, C)
        w = self.excite(w).view(B, C, 1, 1)
        return x * w


class ResConvBlock(nn.Module):
    """
    Two-conv residual block with SE attention.

    Conv(in→out, 3x3, stride) → BN → ReLU →
    Conv(out→out, 3x3, stride=1) → BN → SE →
    Add(residual) → ReLU

    A 1×1 projection shortcut is used when in_ch != out_ch or stride != 1.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch)

        if in_ch != out_ch or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + residual, inplace=True)


# ---------------------------------------------------------------------------
# PongDetector
# ---------------------------------------------------------------------------


class PongDetector(DetectorModel):
    """
    Single-cup CNN detector with SE attention, residual connections,
    and a fully convolutional grid detection head.

    Architecture
    ------------
    Input:  Tensor[B, 3, 640, 640]

    Encoder (SE + residual blocks):
      ResConvBlock(3,   32, stride=2)  ->  [B,  32, 320, 320]
      ResConvBlock(32,  64, stride=2)  ->  [B,  64, 160, 160]
      ResConvBlock(64, 128, stride=2)  ->  [B, 128,  80,  80]
      ResConvBlock(128,256, stride=1)  ->  [B, 256,  80,  80]

    Detection head (fully convolutional):
      Conv2d(256, 128, 3, pad=1) -> BN -> ReLU
      Conv2d(128, 5, 1) -> [B, 5, H, W]
      Channels: (cx_offset, cy_offset, w, h, score)

    Grid decoding (YOLO-style):
      cx = (sigmoid(cx_off) + col) / W
      cy = (sigmoid(cy_off) + row) / H
      w  = anchor_w * exp(clamp(w_raw, -5, 5))
      h  = anchor_h * exp(clamp(h_raw, -5, 5))
      score = sigmoid(score_raw)

    Best detection = cell with highest score.

    Dual-path features for the aim head:
      cell_features:   256-dim from the detected cell's encoder features
      global_features: 256-dim from global average pool of the full feature map
      aim_input = cat(cell, global) -> 512-dim

    Loss (in train_detector.py):
      CIoU loss on the positive cell's box vs ground truth
      Focal loss on all cells' scores (positive cell neighbourhood=1, rest=0)
    """

    # Grid size fixed for INPUT_SIZE=(640,640) with the current encoder stride.
    GRID_H: int = 80
    GRID_W: int = 80

    def __init__(
        self,
        anchor_w: float = DEFAULT_ANCHOR_W,
        anchor_h: float = DEFAULT_ANCHOR_H,
    ) -> None:
        super().__init__()

        # Anchor priors for box size decoding.
        # Set by auto-anchor during training; baked into ONNX at export.
        self.register_buffer("anchor_w", torch.tensor(anchor_w, dtype=torch.float32))
        self.register_buffer("anchor_h", torch.tensor(anchor_h, dtype=torch.float32))

        # Pre-computed float32 grid offsets - baked as constants into ONNX so
        # no dynamic int64 Shape/Mul/Add ops are generated during export.
        # grid_col[0, 0, j] = j  (raw column indices as float32)
        # grid_row[0, i, 0] = i  (raw row indices as float32)
        # Division by GRID_W/GRID_H happens in _decode_grid using the static
        # Python int class attributes, which the ONNX tracer folds to constants.
        col = torch.arange(self.GRID_W, dtype=torch.float32).view(1, 1, self.GRID_W)
        row = torch.arange(self.GRID_H, dtype=torch.float32).view(1, self.GRID_H, 1)
        self.register_buffer("grid_col", col)  # [1, 1, W]
        self.register_buffer("grid_row", row)  # [1, H, 1]

        # Encoder - SE + residual blocks
        self.block1: nn.Module = ResConvBlock(3, 32, stride=2)  # 640 -> 320
        self.block2: nn.Module = ResConvBlock(32, 64, stride=2)  # 320 -> 160
        self.block3: nn.Module = ResConvBlock(64, 128, stride=2)  # 160 -> 80
        self.block4: nn.Module = ResConvBlock(128, 256, stride=1)  # 80  -> 80

        # Fully convolutional detection head
        self.detect_head: nn.Sequential = nn.Sequential(
            nn.Conv2d(ENCODER_DIM, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 5, 1),  # 5 channels: cx_off, cy_off, w, h, score
        )

        # Global average pool for the aim head's global context path
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        # Small init for the detection head's final 1x1 conv so raw outputs
        # start near 0: sigmoid(0)=0.5 for offsets/score, exp(0)=1.0 for
        # anchor-scaled w/h. Prevents sigmoid saturation at initialization.
        final_conv = self.detect_head[3]
        nn.init.normal_(final_conv.weight, std=0.01)
        nn.init.zeros_(final_conv.bias)

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        """Run the encoder and return the spatial feature map [B, 256, H, W]."""
        x = self.block1(images)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x

    def _decode_grid(self, raw: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Decode raw detection head output into (cx, cy, w, h, score) per cell.

        cx, cy: sigmoid offset + cell position, normalised to [0, 1].
        w, h:   anchor * exp(raw), YOLO-style size decoding.
        score:  sigmoid confidence.
        Returns dict with "boxes" [B, H*W, 4] and "scores" [B, H*W].
        """
        B, _, H, W = raw.shape

        # Use pre-computed float32 grid buffers (registered in __init__) so
        # that no dynamic int64 Shape/Mul/Add ops appear in the ONNX graph.
        # grid_col stores raw column indices [0..GRID_W-1] as float32;
        # dividing by the static Python int GRID_W is folded to a constant
        # scalar multiply by the ONNX tracer - no Shape ops generated.
        cx = (torch.sigmoid(raw[:, 0]) + self.grid_col) / self.GRID_W  # [B, H, W]
        cy = (torch.sigmoid(raw[:, 1]) + self.grid_row) / self.GRID_H  # [B, H, W]

        # Decode size via anchor * exp(raw) - YOLO-style, clamped to prevent overflow
        w = self.anchor_w * torch.exp(torch.clamp(raw[:, 2], -5.0, 5.0))  # [B, H, W]
        h = self.anchor_h * torch.exp(torch.clamp(raw[:, 3], -5.0, 5.0))  # [B, H, W]

        score = torch.sigmoid(raw[:, 4])  # [B, H, W]

        # Flatten spatial dims using static constants to avoid dynamic
        # int64 reshape ops in ONNX (cv2.dnn cannot handle int64 arithmetic)
        N = self.GRID_H * self.GRID_W
        boxes = torch.stack([cx, cy, w, h], dim=-1).reshape(B, N, 4)  # [B, H*W, 4]
        scores = score.reshape(B, N)  # [B, H*W]

        return {"boxes": boxes, "scores": scores}

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        images : Tensor[B, 3, 640, 640]

        Returns
        -------
        dict:
            "features"  : Tensor[B, 512] - dual-path features for aim head
                          (256 from detected cell + 256 from global pool)
            "boxes"     : Tensor[B, 4]   - best detection (cx, cy, w, h) 0-1
            "scores"    : Tensor[B]      - best detection confidence
            "raw_grid"  : Tensor[B, 5, H, W] - raw detection head output
                          (used by training loss, score heatmap visualisation)
        """
        feat = self._encode(images)  # [B, 256, H, W]
        raw = self.detect_head(feat)  # [B, 5, H, W]

        decoded = self._decode_grid(raw)
        all_boxes = decoded["boxes"]  # [B, H*W, 4]
        all_scores = decoded["scores"]  # [B, H*W]

        # Best detection per image (highest score cell).
        # Use torch.gather with .repeat() (not .expand()) - expand generates
        # dynamic ConstantOfShape/Mul/Add int64 ops in ONNX that cv2.dnn
        # cannot handle. repeat() emits a static Tile op instead.
        best_idx = all_scores.argmax(dim=1, keepdim=True)  # [B, 1]
        scores = all_scores.gather(1, best_idx).squeeze(1)  # [B]

        box_idx = best_idx.unsqueeze(-1).repeat(1, 1, 4)  # [B, 1, 4]
        boxes = all_boxes.gather(1, box_idx).squeeze(1)  # [B, 4]

        # Clamp w/h to [0, 1] for downstream
        boxes = torch.cat(
            [
                boxes[:, :2],
                boxes[:, 2:].clamp(0.0, 1.0),
            ],
            dim=1,
        )

        # Dual-path features for the aim head:
        #   Path A: 256-dim from the detected cell's encoder features
        #   Path B: 256-dim from global average pool
        # Use static GRID constants to avoid dynamic reshape int64 ops.
        N = self.GRID_H * self.GRID_W  # 6400
        feat_flat = feat.reshape(-1, ENCODER_DIM, N).permute(0, 2, 1)  # [B, H*W, C]
        cell_idx = best_idx.unsqueeze(-1).repeat(1, 1, ENCODER_DIM)  # [B, 1, C]
        cell_feat = feat_flat.gather(1, cell_idx).squeeze(1)  # [B, C]
        global_feat = self.global_pool(feat)  # [B, 256]
        aim_features = torch.cat([cell_feat, global_feat], dim=1)  # [B, 512]

        return {
            "features": aim_features,
            "boxes": boxes,
            "scores": scores,
            "raw_grid": raw,
        }


# ---------------------------------------------------------------------------
# PongAimModelV1 - baseline (516-dim, no derived features)
# ---------------------------------------------------------------------------


class PongAimModelV1(AimModel):
    """
    Aim model v1 (baseline): 516-dim combined head.

    Input: [features(512), bbox(cx, cy, w, h)] -> 516-dim.
    No derived bbox features (aspect ratio, area). This is the original
    architecture before the discovery that bbox h varies 40% across the
    Y range while bbox cy varies only 1.5%.

    Architecture
    ------------
    backbone : PongDetector
        Produces detection boxes/scores and 512-dim dual-path features.

    head:
      Input: cat(features[512], cx, cy, w, h) = 516-dim
      Linear(516, 256) -> ReLU -> Dropout(0.2)
      Linear(256, 128) -> ReLU -> Dropout(0.2)
      Linear(128, 2)   -> Tanh
      Output: [B, 2] normalised (x_steps, y_steps) in [-1, 1]

    Training
    --------
    pong-train-head --model v1 --backbone <ckpt>
    """

    def __init__(self, backbone: PongDetector | None = None) -> None:
        if backbone is None:
            backbone = PongDetector()
        super().__init__(backbone)

        self.head = nn.Sequential(
            nn.Linear(AIM_HEAD_V1_INPUT_DIM, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
            nn.Tanh(),
        )

        self._init_head()

    def _init_head(self) -> None:
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """End-to-end forward pass. Returns features, boxes, scores, steps, raw_grid."""
        out = self.backbone(images)
        feat = out["features"]  # [B, 512]
        bbox = out["boxes"]  # [B, 4] - (cx, cy, w, h)

        head_input = torch.cat([feat, bbox], dim=1)  # [B, 516]
        steps = self.head(head_input)  # [B, 2]

        return {
            "features": feat,
            "boxes": bbox,
            "scores": out["scores"],
            "steps": steps,
            "raw_grid": out["raw_grid"],
        }


# ---------------------------------------------------------------------------
# PongAimModelV2 - combined head with derived features (518-dim)
# ---------------------------------------------------------------------------


class PongAimModelV2(AimModel):
    """
    Aim model v2: 518-dim combined head with derived bbox features.

    Extends v1 by adding two derived features (h/w aspect ratio, w*h area)
    to the head input, giving a 518-dim input. These features encode cup
    distance/depth more discriminatively than bbox cy alone: bbox h varies
    40% across the Y range while bbox cy varies only 1.5%.

    Architecture
    ------------
    backbone : PongDetector
        Produces detection boxes/scores and 512-dim dual-path features.
        Frozen during phase 2 training (unless --unfreeze-backbone is set).

    head:
      Input: cat(features[512], cx, cy, w, h, h/w, w*h) = 518-dim
      Linear(518, 256) -> ReLU -> Dropout(0.2)
      Linear(256, 128) -> ReLU -> Dropout(0.2)
      Linear(128, 2)   -> Tanh
      Output: [B, 2] normalised (x_steps, y_steps) in [-1, 1]

    Training
    --------
    pong-train-head --model v2 --backbone <ckpt>
    """

    def __init__(self, backbone: PongDetector | None = None) -> None:
        if backbone is None:
            backbone = PongDetector()
        super().__init__(backbone)

        self.head = nn.Sequential(
            nn.Linear(AIM_HEAD_INPUT_DIM, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
            nn.Tanh(),
        )

        self._init_head()

    def _init_head(self) -> None:
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """End-to-end forward pass. Returns features, boxes, scores, steps, raw_grid."""
        out = self.backbone(images)
        feat = out["features"]  # [B, 512]
        bbox = out["boxes"]  # [B, 4] - (cx, cy, w, h)

        # Derived bbox features: aspect ratio (h/w) and area (w*h)
        aspect = bbox[:, 3:4] / (bbox[:, 2:3] + 1e-6)  # h/w [B, 1]
        area = bbox[:, 2:3] * bbox[:, 3:4]  # w*h [B, 1]

        head_input = torch.cat([feat, bbox, aspect, area], dim=1)  # [B, 518]
        steps = self.head(head_input)  # [B, 2]

        return {
            "features": feat,
            "boxes": bbox,
            "scores": out["scores"],
            "steps": steps,
            "raw_grid": out["raw_grid"],
        }


# ---------------------------------------------------------------------------
# PongAimModelV3 - split X/Y heads
# ---------------------------------------------------------------------------


class PongAimModelV3(AimModel):
    """
    Aim model v3: independent X and Y prediction heads.

    Decouples X and Y prediction so that Y-oriented bbox features (aspect
    ratio, area) cannot interfere with X accuracy and vice versa.
    PongAimModelV1, V2, and V3 are all concrete implementations of the
    AimModel ABC - alternatives, not parent/child.

    X head input: [features(512), bbox_cx(1)] -> 513-dim
        bbox_cx has r2=0.98 with x_steps; no other features add signal.

    Y head input: [features(512), bbox_h(1), aspect(h/w)(1), area(w*h)(1)] -> 515-dim
        bbox_h varies 40% across Y range (vs 1.5% for bbox_cy which is excluded).
        Aspect ratio and area provide additional depth/distance signal.

    head:
        X head: Linear(513, 128) -> ReLU -> Dropout(0.2) -> Linear(128, 1) -> Tanh
        Y head: Linear(515, 128) -> ReLU -> Dropout(0.2) -> Linear(128, 1) -> Tanh
        Output: cat([x_steps, y_steps], dim=1) = Tensor[B, 2]

    Training
    --------
    pong-train-head --model v3 --backbone <ckpt>
    head_parameters() in AimModel base class returns all non-backbone params,
    so x_head and y_head are both trained correctly without any override.

    ONNX export
    -----------
    Output shape [B, 2] is identical to V1/V2 - Pi-side inference
    in detector.py requires no changes.
    """

    def __init__(self, backbone: PongDetector | None = None) -> None:
        if backbone is None:
            backbone = PongDetector()
        super().__init__(backbone)  # AimModel.__init__ - sets self.backbone only

        self.x_head = nn.Sequential(
            nn.Linear(SPLIT_X_INPUT_DIM, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Tanh(),
        )
        self.y_head = nn.Sequential(
            nn.Linear(SPLIT_Y_INPUT_DIM, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Tanh(),
        )
        self._init_heads()

    def _init_heads(self) -> None:
        for head in (self.x_head, self.y_head):
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """End-to-end forward pass. Returns features, boxes, scores, steps, raw_grid."""
        out = self.backbone(images)
        feat = out["features"]  # [B, 512]
        bbox = out["boxes"]  # [B, 4]: (cx, cy, w, h)

        aspect = bbox[:, 3:4] / (bbox[:, 2:3] + 1e-6)  # h/w  [B, 1]
        area = bbox[:, 2:3] * bbox[:, 3:4]  # w*h  [B, 1]

        x_input = torch.cat([feat, bbox[:, 0:1]], dim=1)  # [B, 513]
        y_input = torch.cat([feat, bbox[:, 3:4], aspect, area], dim=1)  # [B, 515]

        x_steps = self.x_head(x_input)  # [B, 1]
        y_steps = self.y_head(y_input)  # [B, 1]

        steps = torch.cat([x_steps, y_steps], dim=1)  # [B, 2]

        return {
            "features": feat,
            "boxes": bbox,
            "scores": out["scores"],
            "steps": steps,
            "raw_grid": out["raw_grid"],
        }
