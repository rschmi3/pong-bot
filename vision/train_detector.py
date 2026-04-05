"""
training/train_detector.py - Phase 1: train the object detection backbone.

Trains a DetectorModel subclass on labelled bounding box images and exports
the best checkpoint as both .pt (for phase 2) and .onnx (for Pi inference).

Expected data layout (managed by pong-roboflow pull)
----------------------------------------------------
    data/
      train/
        images/   *.jpg   - training images
        labels/   *.txt   - YOLO-format labels: one file per image, same stem
      valid/
        images/   *.jpg   - validation images
        labels/   *.txt
      test/                - optional; used for final evaluation only
        images/
        labels/

    Each label line: <class_id> <cx> <cy> <w> <h>  (normalised 0–1)
    Only one class expected (the cup, class 0).

Usage
-----
    python training/train_detector.py \\
        --data data/ \\
        --epochs 50 \\
        --output models/

    # Use a custom model class (dotted import path):
    python training/train_detector.py \\
        --model my_module.MyDetector \\
        --data data/ --epochs 50 --output models/
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    raise ImportError(
        "torch is required for training. "
        "Install the training package: nix profile install .#training"
    ) from None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class CupDetectionDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """
    Loads image/label pairs from split directories under data_dir.

    Expects:
        data_dir/{split}/images/*.jpg
        data_dir/{split}/labels/{stem}.txt

    Labels are YOLO-format .txt files: one line per image with
    "<class_id> <cx> <cy> <w> <h>" normalised to [0, 1].
    Only the first line is used (single-box assumption).
    Images without a corresponding label file are treated as negative samples (no cup present).

    Parameters
    ----------
    data_dir : Path
        Base data directory (e.g. data/).
    split : str or list[str]
        One or more split names to load from (e.g. "train", ["train", "valid"]).
    augment_data : bool
        Whether to apply augmentations.
    """

    def __init__(
        self,
        data_dirs: Path | list[Path],
        split: str | list[str] = "train",
        augment_data: bool = True,
    ) -> None:
        self.augment = augment_data
        # samples: (image_path, box [4], is_positive)
        self.samples: list[tuple[Path, np.ndarray, bool]] = []

        dirs = [data_dirs] if isinstance(data_dirs, Path) else data_dirs
        splits = [split] if isinstance(split, str) else split

        n_positive = 0
        n_negative = 0

        for data_dir in dirs:
            for s in splits:
                image_dir = data_dir / s / "images"
                label_dir = data_dir / s / "labels"

                if not image_dir.exists():
                    logger.warning("Split directory not found: %s", image_dir)
                    continue

                for image_path in sorted(image_dir.glob("*.jpg")):
                    label_path = label_dir / (image_path.stem + ".txt")
                    if not label_path.exists():
                        # No label file = negative sample (no cup in image)
                        box = np.zeros(4, dtype=np.float32)
                        self.samples.append((image_path, box, False))
                        n_negative += 1
                        continue

                    with open(label_path) as f:
                        line = f.readline().strip()
                    if not line:
                        # Empty label file = also negative
                        box = np.zeros(4, dtype=np.float32)
                        self.samples.append((image_path, box, False))
                        n_negative += 1
                        continue

                    parts = line.split()
                    if len(parts) < 5:
                        logger.warning("Bad label in %s - skipping", label_path.name)
                        continue

                    # parts[0] is class_id (ignored - single class)
                    box = np.array([float(p) for p in parts[1:5]], dtype=np.float32)
                    self.samples.append((image_path, box, True))
                    n_positive += 1

        logger.info(
            "Dataset: %d images (%d positive, %d negative) from split(s) %s "
            "across %d config(s)",
            len(self.samples),
            n_positive,
            n_negative,
            splits,
            len(dirs),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        import cv2

        from vision.transforms import augment, preprocess

        image_path, box, is_positive = self.samples[idx]

        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Could not read image: {image_path}")

        chw = preprocess(frame)  # [3, 640, 640] float32

        if self.augment:
            chw, box = augment(chw, box)

        image_tensor = torch.from_numpy(chw)
        box_tensor = torch.from_numpy(box)
        score_tensor = torch.tensor(1.0 if is_positive else 0.0)

        return image_tensor, box_tensor, score_tensor


# ---------------------------------------------------------------------------
# CIoU loss
# ---------------------------------------------------------------------------


def _ciou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute Complete IoU between predicted and target boxes.

    Both inputs: Tensor[..., 4] in (cx, cy, w, h) normalised format.
    Returns: Tensor[...] CIoU values in [-1, 1] (higher is better).
    """
    import math

    # Convert to x1y1x2y2
    px1 = pred[..., 0] - pred[..., 2] / 2
    py1 = pred[..., 1] - pred[..., 3] / 2
    px2 = pred[..., 0] + pred[..., 2] / 2
    py2 = pred[..., 1] + pred[..., 3] / 2

    gx1 = target[..., 0] - target[..., 2] / 2
    gy1 = target[..., 1] - target[..., 3] / 2
    gx2 = target[..., 0] + target[..., 2] / 2
    gy2 = target[..., 1] + target[..., 3] / 2

    # Intersection
    ix1 = torch.max(px1, gx1)
    iy1 = torch.max(py1, gy1)
    ix2 = torch.min(px2, gx2)
    iy2 = torch.min(py2, gy2)
    inter = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)

    # Union
    pred_area = (px2 - px1) * (py2 - py1)
    gt_area = (gx2 - gx1) * (gy2 - gy1)
    union = pred_area + gt_area - inter + 1e-7

    iou = inter / union

    # Enclosing box diagonal
    ex1 = torch.min(px1, gx1)
    ey1 = torch.min(py1, gy1)
    ex2 = torch.max(px2, gx2)
    ey2 = torch.max(py2, gy2)
    diag_sq = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2 + 1e-7

    # Center distance
    center_dist_sq = (pred[..., 0] - target[..., 0]) ** 2 + (
        pred[..., 1] - target[..., 1]
    ) ** 2

    # Aspect ratio consistency
    v = (4 / (math.pi**2)) * (
        torch.atan(target[..., 2] / (target[..., 3] + 1e-7))
        - torch.atan(pred[..., 2] / (pred[..., 3] + 1e-7))
    ) ** 2
    alpha = v / (1 - iou + v + 1e-7)

    return iou - center_dist_sq / diag_sq - alpha.detach() * v


def _focal_bce(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    pos_weight: float = 100.0,
) -> torch.Tensor:
    """
    Focal loss with positive cell weighting.

    Down-weights easy negatives (the vast majority of 80×80 grid cells)
    and up-weights the rare positive cell to counteract the ~6400:1 imbalance.
    """
    import torch.nn.functional as Fn

    bce = Fn.binary_cross_entropy(pred, target, reduction="none")
    pt = torch.where(target > 0.5, pred, 1 - pred)
    focal_weight = alpha * (1 - pt) ** gamma
    cell_weight = torch.where(target > 0.5, pos_weight, 1.0)
    return (focal_weight * cell_weight * bce).mean()


def _compute_loss(
    model,
    raw_grid: torch.Tensor,
    target_boxes: torch.Tensor,
    target_scores: torch.Tensor,
    bbox_weight: float,
) -> torch.Tensor:
    """
    Compute per-cell detection loss: CIoU on positive cell + focal BCE on all.

    Parameters
    ----------
    model : PongDetector - needed for _decode_grid (anchor values)
    raw_grid : Tensor[B, 5, H, W] - raw detection head output
    target_boxes : Tensor[B, 4] - (cx, cy, w, h) normalised ground truth
    target_scores : Tensor[B] - 1.0 for positive, 0.0 for negative images
    bbox_weight : float - weight for CIoU loss relative to score loss

    Returns
    -------
    Scalar loss tensor.
    """
    B, _, H, W = raw_grid.shape

    # Build score target grid: all zeros except the positive cell
    score_target = torch.zeros(B, H * W, device=raw_grid.device)

    # Decode all grid predictions using the model's anchors
    decoded = model._decode_grid(raw_grid)
    all_boxes = decoded["boxes"]  # [B, H*W, 4]
    all_scores = decoded["scores"]  # [B, H*W]

    # Multi-cell positive assignment: assign a neighborhood of cells around
    # the GT center as positive. This prevents the model from anchoring to
    # the cup's bottom edge (most visually distinctive) instead of the center.
    # Using a 3x3 neighborhood (9 cells) for each positive image.
    RADIUS = 1  # 1 = 3x3, 2 = 5x5

    ciou_loss = torch.tensor(0.0, device=raw_grid.device)
    n_pos = 0

    for b in range(B):
        if target_scores[b] < 0.5:
            # Negative image - no box loss, all scores should be 0
            continue

        # Find the center grid cell for this ground truth box
        gt_cx, gt_cy = target_boxes[b, 0].item(), target_boxes[b, 1].item()
        center_col = min(int(gt_cx * W), W - 1)
        center_row = min(int(gt_cy * H), H - 1)
        gt_box = target_boxes[b]  # [4]

        # Assign a neighborhood of cells as positive
        for dr in range(-RADIUS, RADIUS + 1):
            for dc in range(-RADIUS, RADIUS + 1):
                r = center_row + dr
                c = center_col + dc
                if r < 0 or r >= H or c < 0 or c >= W:
                    continue
                cell_idx = r * W + c

                # Set this cell's score target to 1
                score_target[b, cell_idx] = 1.0

                # CIoU loss for this cell's prediction vs ground truth
                pred_box = all_boxes[b, cell_idx]  # [4]
                ciou_val = _ciou(pred_box.unsqueeze(0), gt_box.unsqueeze(0))
                ciou_loss = ciou_loss + (1.0 - ciou_val.squeeze())
                n_pos += 1

    if n_pos > 0:
        ciou_loss = ciou_loss / n_pos

    # Focal BCE score loss with positive cell weighting
    score_loss = _focal_bce(all_scores, score_target)

    return bbox_weight * ciou_loss + score_loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _train(
    model,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    device,
    output_dir: Path,
    bbox_weight: float,
    ckpt_filename: str = "cup_detector.pt",
) -> Path:
    """Run training loop with CIoU + BCE loss. Returns path to best checkpoint."""
    model = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser,
            patience=5,
            factor=0.5,
        )
        if val_loader is not None
        else None
    )

    best_val_loss = float("inf")
    best_ckpt = output_dir / ckpt_filename

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for images, boxes, scores in train_loader:
            images = images.to(device)
            boxes = boxes.to(device)
            scores = scores.to(device)

            optimiser.zero_grad()
            out = model(images)

            loss = _compute_loss(model, out["raw_grid"], boxes, scores, bbox_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- Validate ---
        val_loss = float("nan")
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, boxes, scores in val_loader:
                    images = images.to(device)
                    boxes = boxes.to(device)
                    scores = scores.to(device)
                    out = model(images)
                    loss = _compute_loss(
                        model, out["raw_grid"], boxes, scores, bbox_weight
                    )
                    val_loss += loss.item()
            val_loss /= len(val_loader)

            if scheduler is not None:
                scheduler.step(val_loss)

        logger.info(
            "Epoch %3d/%d  train=%.4f  val=%.4f",
            epoch,
            epochs,
            train_loss,
            val_loss,
        )

        # Save best checkpoint based on val loss (or train loss if no val)
        check_loss = val_loss if val_loader is not None else train_loss
        if check_loss < best_val_loss:
            best_val_loss = check_loss
            torch.save(model.state_dict(), best_ckpt)
            logger.info("  → New best checkpoint saved (loss=%.4f)", best_val_loss)

    return best_ckpt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--elastics",
    required=True,
    type=int,
    multiple=True,
    help="Number of elastics to train on (e.g. --elastics 2 --elastics 4). "
    "Combines data from all specified elastic configs.",
)
@click.option("--epochs", default=50, show_default=True, type=int)
@click.option(
    "--seed",
    default=0,
    show_default=True,
    type=int,
    help="Random seed for reproducible training.",
)
@click.option("--batch-size", default=16, show_default=True, type=int)
@click.option(
    "--lr", default=5e-3, show_default=True, type=float, help="Initial learning rate."
)
@click.option(
    "--bbox-weight",
    default=5.0,
    show_default=True,
    type=float,
    help="Weight applied to the bbox regression loss term.",
)
@click.option(
    "--output",
    default="models",
    show_default=True,
    type=click.Path(),
    help="Directory to save checkpoints and ONNX export.",
)
@click.option("--device", default="cuda", show_default=True, help="Training device.")
@click.option("--verbose", "-v", is_flag=True, default=False)
@click.option(
    "--output-name",
    default="cup_detector",
    show_default=True,
    help="Base name for the checkpoint file. Saves <name>.pt to the "
    "--output directory. Defaults to 'cup_detector'. Use a unique "
    "name when running parallel training jobs to avoid conflicts.",
)
def cli(
    elastics: tuple[int, ...],
    epochs: int,
    seed: int,
    batch_size: int,
    lr: float,
    bbox_weight: float,
    output: str,
    device: str,
    verbose: bool,
    output_name: str,
) -> None:
    """
    Phase 1: Train the object detection backbone.

    Trains on labelled bounding box images and exports the best weights
    as cup_detector.pt and cup_detector.onnx.
    """
    logging.basicConfig(
        format="%(levelname)-8s %(name)s: %(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
        stream=sys.stdout,
    )

    from torch.utils.data import DataLoader

    from utils.data_dir import elastics_data_dir

    # Reproducible training
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Combine data from all specified elastic configs
    data_dirs = [elastics_data_dir(n) for n in elastics]
    click.echo(f"Loading data from: {[str(d) for d in data_dirs]}")

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build datasets from split directories across all elastic configs
    train_ds = CupDetectionDataset(data_dirs, split="train", augment_data=True)

    # Auto-compute anchor priors from positive training labels
    pos_widths = [s[1][2] for s in train_ds.samples if s[2]]  # box[2] = w
    pos_heights = [s[1][3] for s in train_ds.samples if s[2]]  # box[3] = h
    if pos_widths:
        anchor_w = float(np.mean(pos_widths))
        anchor_h = float(np.mean(pos_heights))
        click.echo(
            f"Auto anchor: w={anchor_w:.4f} h={anchor_h:.4f} "
            f"(from {len(pos_widths)} positive labels)"
        )
    else:
        from vision.models.pong_model import DEFAULT_ANCHOR_H, DEFAULT_ANCHOR_W

        anchor_w, anchor_h = DEFAULT_ANCHOR_W, DEFAULT_ANCHOR_H
        click.echo(
            f"No positive labels - using default anchors: w={anchor_w} h={anchor_h}"
        )

    # Instantiate PongDetector with computed anchors
    from vision.models.pong_model import PongDetector

    detector = PongDetector(anchor_w=anchor_w, anchor_h=anchor_h)
    n_params = sum(p.numel() for p in detector.parameters())
    logger.info("Model: %s  (%s parameters)", PongDetector.__name__, f"{n_params:,}")
    val_ds = CupDetectionDataset(data_dirs, split="valid", augment_data=False)

    if len(train_ds) == 0:
        click.echo(
            "No labelled training images found. "
            "Run 'pong-roboflow pull --elastics N --version V' to download labels.",
            err=True,
        )
        sys.exit(1)

    if len(val_ds) == 0:
        click.echo(
            "WARNING: No validation images found. "
            "Training will proceed without validation - results may overfit.",
            err=True,
        )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
        if len(val_ds) > 0
        else None
    )

    logger.info(
        "Train: %d  Val: %d  Device: %s",
        len(train_ds),
        len(val_ds),
        device,
    )

    # Train
    best_ckpt = _train(
        detector,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=lr,
        device=torch.device(device),
        output_dir=output_dir,
        bbox_weight=bbox_weight,
        ckpt_filename=f"{output_name}.pt",
    )

    click.echo(f"\nDone. Checkpoint: {best_ckpt}")
    click.echo(
        "Next step: train the aim head with:\n"
        "  python -m training.train_head --elastics N --backbone "
        f"{best_ckpt}"
    )


if __name__ == "__main__":
    cli()
