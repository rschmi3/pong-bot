"""
vision/train_head.py - pong-train-head: train the aim prediction head (phase 2).

Loads phase 1 backbone weights, trains the prediction head on shot data with
Gaussian distance-weighted pseudo-labels. Exports backbone + head as ONNX.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset
except ImportError:
    raise ImportError(
        "torch is required for training. "
        "Install the training package: nix profile install .#training"
    ) from None

from utils.data_dir import NORMALISE_X, NORMALISE_Y

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from vision.models.base import AimModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ShotDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """
    Dataset of (bbox, target_steps, weight) tuples for aim head training.

    Uses Gaussian distance-weighted pseudo-labels: weight = exp(-dist/temperature)
    where dist is the Euclidean distance in step-space from each shot to the
    session's winning position. Loads bounding boxes from YOLO-format label files.
    """

    def __init__(
        self,
        shots_path: Path,
        rl_shots_path: Path,
        data_dir: Path,
        temperature: float = 2000.0,
    ) -> None:
        import math

        # samples: (image_path, normalised_target, weight)
        self.samples: list[tuple[Path, np.ndarray, float]] = []

        # Build session_ends lookup: session_id → session_end record
        session_ends: dict[str, dict] = {}
        if rl_shots_path.exists():
            with open(rl_shots_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("type") == "session_end":
                        session_ends[r["session_id"]] = r
        logger.info(
            "Loaded %d session_end records from %s", len(session_ends), rl_shots_path
        )

        # Compute per-cell median winning_y from all hit sessions that have cup
        # labels.  Using the median across sessions for the same (cup_x, cup_y)
        # cell reduces pseudo-label noise caused by RL convergence variance
        # (sessions that converge from different directions land at slightly
        # different winning positions).  X stays per-session because within-cell
        # X variance is small and meaningful.  Sessions without cup labels are
        # ignored - they cannot contribute to the cell median.
        import statistics
        from collections import defaultdict as _dd

        cell_wy: dict[tuple[int, int], list[int]] = _dd(list)
        for se in session_ends.values():
            if (
                se.get("outcome") == "hit"
                and se.get("cup_x") is not None
                and se.get("cup_y") is not None
                and se.get("winning_y") is not None
            ):
                cell_wy[(se["cup_x"], se["cup_y"])].append(se["winning_y"])

        median_wy_by_cell: dict[tuple[int, int], float] = {
            cell: statistics.median(wys) for cell, wys in cell_wy.items()
        }

        logger.info(
            "Per-cell median Y targets: %d cells  (unlabelled sessions ignored)",
            len(median_wy_by_cell),
        )
        for cell in sorted(median_wy_by_cell):
            wys = cell_wy[cell]
            logger.debug(
                "  cell %s: n=%d  median=%.0f  std=%.0f  values=%s",
                cell,
                len(wys),
                median_wy_by_cell[cell],
                (sum((v - median_wy_by_cell[cell]) ** 2 for v in wys) / len(wys))
                ** 0.5,
                wys,
            )

        # Load shot records
        if not shots_path.exists():
            logger.warning("shots.jsonl not found: %s", shots_path)
            return
        with open(shots_path) as f:
            records = [json.loads(line) for line in f if line.strip()]
        logger.info("%d shot records found in %s", len(records), shots_path)

        skipped_no_image = 0
        skipped_no_truth = 0
        n_hits = 0
        n_near = 0  # weight > 0.5
        n_far = 0  # weight <= 0.5

        for record in records:
            # Resolve image path (stored relative to data_dir)
            rel = record.get("image", "")
            image_path = data_dir / rel
            if not image_path.exists():
                skipped_no_image += 1
                continue

            x = record["x_steps"]
            y = record["y_steps"]
            session_id = record.get("session_id")

            # Determine target and weight
            target: np.ndarray | None = None
            weight: float = 0.0

            if session_id and session_id in session_ends:
                se = session_ends[session_id]
                if se.get("outcome") == "hit" and se.get("winning_x") is not None:
                    wx = se["winning_x"]
                    wy_session = se["winning_y"]

                    # Use per-cell median Y if available (reduces pseudo-label
                    # noise from RL convergence variance across sessions).
                    # Sessions without cup labels are not in the median lookup
                    # so they fall back to the session's own winning_y.
                    # X stays per-session - within-cell X variance is small
                    # and meaningful.
                    cup_x_se = se.get("cup_x")
                    cup_y_se = se.get("cup_y")
                    if cup_x_se is not None and cup_y_se is not None:
                        wy_target = median_wy_by_cell.get(
                            (cup_x_se, cup_y_se),
                            wy_session,
                        )
                    else:
                        wy_target = wy_session

                    # Gaussian weight uses per-session winning_y (the actual
                    # position the robot converged to) so shots far from that
                    # session's hit position are correctly down-weighted.
                    dist = math.sqrt((x - wx) ** 2 + (y - wy_session) ** 2)
                    weight = math.exp(-dist / temperature)
                    # Normalise targets to [-1, 1] range
                    target = np.array(
                        [wx / NORMALISE_X, wy_target / NORMALISE_Y],
                        dtype=np.float32,
                    )
                else:
                    # Session timed out or no cup label - no ground truth
                    skipped_no_truth += 1
                    continue
            elif record.get("scored") is True:
                # pong-collect hit or RL hit with no session_id: exact ground truth
                target = np.array(
                    [x / NORMALISE_X, y / NORMALISE_Y],
                    dtype=np.float32,
                )
                weight = 1.0
            else:
                # No session context and not a confirmed hit - skip
                skipped_no_truth += 1
                continue

            self.samples.append((image_path, target, weight))

            # Stats
            if weight >= 0.99:
                n_hits += 1
            elif weight > 0.5:
                n_near += 1
            else:
                n_far += 1

        logger.info(
            "ShotDataset: %d samples loaded\n"
            "  hits (weight≈1.0):         %4d\n"
            "  near-misses (weight>0.5):  %4d\n"
            "  far-misses (weight≤0.5):   %4d\n"
            "  skipped (no image):        %4d\n"
            "  skipped (no ground truth): %4d",
            len(self.samples),
            n_hits,
            n_near,
            n_far,
            skipped_no_image,
            skipped_no_truth,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        import cv2

        from vision.transforms import preprocess

        image_path, target, weight = self.samples[idx]

        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Could not read image: {image_path}")

        # X-flip augmentation (p=0.5): mirror the image horizontally and negate
        # the x target. Doubles effective dataset size and forces the head to
        # learn that horizontal cup position determines the sign of x_steps.
        # The backbone (frozen) will produce a mirrored bbox cx for the flipped
        # image, so the bbox → steps mapping remains consistent.
        import random

        if random.random() < 0.5:
            frame = cv2.flip(frame, 1)  # horizontal flip
            target = target.copy()
            target[0] = -target[0]  # negate normalised x target

        chw = preprocess(frame)  # [3, 640, 640] float32
        return (
            torch.from_numpy(chw),
            torch.from_numpy(target),
            torch.tensor(weight, dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _train_head(
    model: AimModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    output_dir: Path,
    unfreeze_backbone: bool = False,
    backbone_lr_scale: float = 0.1,
    loss: str = "huber",
    ckpt_filename: str = "aim_model.pt",
) -> Path:
    model = model.to(device)

    # Build loss functions - one for val (scalar), one for weighted train loss
    if loss == "mse":
        val_loss_fn = nn.MSELoss()

        def per_sample_loss(p, t):
            return nn.functional.mse_loss(p, t, reduction="none")
    elif loss == "l1":
        val_loss_fn = nn.L1Loss()

        def per_sample_loss(p, t):
            return nn.functional.l1_loss(p, t, reduction="none")
    elif loss == "huber":
        val_loss_fn = nn.HuberLoss(delta=1.0, reduction="mean")

        def per_sample_loss(p, t):
            return nn.functional.huber_loss(p, t, reduction="none", delta=1.0)
    else:
        raise ValueError(f"Unknown loss: {loss!r}. Choose mse, l1, or huber.")

    logger.info("Loss function: %s", loss)

    if unfreeze_backbone:
        # Differential LR: head at lr, backbone at lr * backbone_lr_scale.
        # Two param groups so the scheduler steps both proportionally.
        backbone_lr = lr * backbone_lr_scale
        head_params = model.head_parameters()
        backbone_params = list(model.backbone.parameters())
        optimiser = torch.optim.Adam(
            [
                {"params": head_params, "lr": lr},
                {"params": backbone_params, "lr": backbone_lr},
            ]
        )
        logger.info(
            "Differential LR - head: %.2e  backbone: %.2e",
            lr,
            backbone_lr,
        )
    else:
        optimiser = torch.optim.Adam(model.head_parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        patience=10,
        factor=0.5,
    )

    best_val_loss = float("inf")
    best_ckpt = output_dir / ckpt_filename

    for epoch in range(1, epochs + 1):
        if unfreeze_backbone:
            # Full model trains end-to-end
            model.train()
        else:
            # Backbone stays in eval mode (frozen), head in train mode
            model.train()
            model.backbone.eval()
        train_loss = 0.0
        for images, targets, weights in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            weights = weights.to(device)

            optimiser.zero_grad()
            out = model(images)
            preds = out["steps"]  # [B, 2] normalised

            # Weighted loss - Option C Gaussian kernel weighting
            per_sample = per_sample_loss(preds, targets)  # [B, 2]
            batch_loss = (weights.unsqueeze(1) * per_sample).mean()
            batch_loss.backward()
            # Clip all params (backbone + head) when fine-tuning end-to-end
            params_to_clip = (
                list(model.parameters())
                if unfreeze_backbone
                else model.head_parameters()
            )
            torch.nn.utils.clip_grad_norm_(params_to_clip, 1.0)
            optimiser.step()
            train_loss += batch_loss.item()
        train_loss /= len(train_loader)

        # Validate (unweighted for fair comparison across epochs)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets, _weights in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                out = model(images)
                preds = out["steps"]
                val_loss += val_loss_fn(preds, targets).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        logger.info(
            "Epoch %3d/%d  train=%.6f  val=%.6f",
            epoch,
            epochs,
            train_loss,
            val_loss,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_ckpt)
            logger.info("  -> New best checkpoint (val=%.6f)", best_val_loss)

    return best_ckpt


# ---------------------------------------------------------------------------
# ONNX export (full end-to-end model)
# ---------------------------------------------------------------------------


def _export_end_to_end_onnx(
    model, ckpt_path: Path, output_dir: Path, onnx_filename: str = "aim_model.onnx"
) -> Path:
    """
    Export the full AimModel (backbone + head) to ONNX for Pi inference.

    The exported model takes a preprocessed image [1, 3, 640, 640] and
    produces named outputs: features [1, 512], boxes [1, 4], scores [1],
    steps [1, 2], raw_grid [1, 5, 80, 80].
    On the Pi, cv2.dnn.readNetFromONNX loads this and runs end-to-end.

    Uses the TorchScript exporter (dynamo=False) for broad compatibility
    with cv2.dnn - no onnxscript dependency at export time.
    """
    model.cpu()
    # strict=False: newly added register_buffer entries (e.g. grid_col/grid_row)
    # may not be present in older checkpoints; they are correctly initialised by
    # __init__ so missing keys are safe to ignore.
    model.load_state_dict(
        torch.load(ckpt_path, map_location="cpu", weights_only=True),
        strict=False,
    )
    model.eval()

    from vision.transforms import INPUT_SIZE

    dummy = torch.zeros(1, 3, INPUT_SIZE[1], INPUT_SIZE[0])
    onnx_path = output_dir / onnx_filename

    torch.onnx.export(
        model,
        (dummy,),
        str(onnx_path),
        input_names=["images"],
        output_names=["features", "boxes", "scores", "steps", "raw_grid"],
        dynamic_axes={"images": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    logger.info("Exported end-to-end ONNX to %s", onnx_path)
    return onnx_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--backbone",
    default=None,
    type=click.Path(exists=True),
    help="Path to phase 1 detector checkpoint (cup_detector.pt). "
    "Loads backbone weights only - head is randomly initialised. "
    "Mutually exclusive with --resume.",
)
@click.option(
    "--resume",
    default=None,
    type=click.Path(exists=True),
    help="Path to a full aim_model.pt to resume training from. "
    "Loads both backbone and head weights. "
    "Mutually exclusive with --backbone.",
)
@click.option(
    "--elastics",
    required=True,
    type=int,
    help="Number of elastics on the launcher (e.g. 2 or 4). "
    "Determines data directory: data/{N}_elastics/",
)
@click.option(
    "--temperature",
    default=2000.0,
    show_default=True,
    type=float,
    help="Gaussian kernel bandwidth in motor steps for distance-weighted "
    "pseudo-labels. weight = exp(-dist / temperature). Lower = only "
    "near-misses contribute; higher = all misses contribute broadly.",
)
@click.option("--epochs", default=100, show_default=True, type=int)
@click.option("--batch-size", default=32, show_default=True, type=int)
@click.option("--lr", default=1e-3, show_default=True, type=float)
@click.option("--val-split", default=0.15, show_default=True, type=float)
@click.option("--device", default="cuda", show_default=True)
@click.option(
    "--loss",
    default="huber",
    show_default=True,
    type=click.Choice(["mse", "l1", "huber"]),
    help="Regression loss function. "
    "mse: minimises to mean, sensitive to outliers. "
    "l1: minimises to median, robust to outlier pseudo-labels. "
    "huber: smooth L1 - MSE near zero, L1 for large errors (recommended).",
)
@click.option(
    "--unfreeze-backbone",
    is_flag=True,
    default=False,
    help="Fine-tune the full model end-to-end with differential learning rates. "
    "Backbone trains at lr * --backbone-lr-scale; head trains at --lr.",
)
@click.option(
    "--backbone-lr-scale",
    default=0.1,
    show_default=True,
    type=float,
    help="Backbone LR multiplier when --unfreeze-backbone is set. "
    "Backbone LR = lr * backbone-lr-scale.",
)
@click.option("--verbose", "-v", is_flag=True, default=False)
@click.option(
    "--output-name",
    default="aim_model",
    show_default=True,
    help="Base name for output files. Saves <name>.pt and <name>.onnx "
    "to data/{N}_elastics/checkpoints/. Defaults to 'aim_model'. "
    "Use a unique name when running parallel training jobs to avoid "
    "checkpoint conflicts.",
)
@click.option(
    "--model",
    default="v2",
    show_default=True,
    type=click.Choice(["v1", "v2", "v3"]),
    help="Aim model version. "
    "v1: 516-dim combined head - baseline, no derived features (PongAimModelV1). "
    "v2: 518-dim combined head - adds aspect ratio + area (PongAimModelV2). "
    "v3: split X/Y heads - 513-dim X head + 515-dim Y head (PongAimModelV3).",
)
def cli(
    backbone: str | None,
    resume: str | None,
    elastics: int,
    temperature: float,
    epochs: int,
    batch_size: int,
    lr: float,
    val_split: float,
    device: str,
    loss: str,
    unfreeze_backbone: bool,
    backbone_lr_scale: float,
    verbose: bool,
    output_name: str,
    model: str,
) -> None:
    """
    Phase 2: Train the aim prediction head (optionally with backbone fine-tuning).

    By default the backbone is frozen and only the head is trained. Use
    --unfreeze-backbone to fine-tune end-to-end with differential LRs: head
    at --lr and backbone at --lr * --backbone-lr-scale (default 0.1x).

    Use --resume to continue training from an existing aim_model.pt (loads
    both backbone and head weights). Use --backbone to start with a fresh
    head on top of a pre-trained detector backbone. The two are mutually
    exclusive.

    Bounding boxes are loaded from YOLO label files in data-dir/labels/
    rather than running the ONNX detector at training time - uses ground-truth
    boxes and is faster. Images without a label file are skipped.

    Uses Option C Gaussian distance-weighted pseudo-labels: every shot in a
    session that converged to a hit contributes to training, weighted by
    exp(-dist / temperature) where dist is the distance in motor step space
    from the shot to the winning position. Direct hits have dist=0, weight=1.0.

    Saves checkpoint and ONNX export to data/{N}_elastics/checkpoints/.
    """
    if resume and backbone:
        raise click.UsageError("--resume and --backbone are mutually exclusive.")
    logging.basicConfig(
        format="%(levelname)-8s %(name)s: %(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
        stream=sys.stdout,
    )

    from torch.utils.data import DataLoader, random_split

    from utils.data_dir import elastics_data_dir

    data_dir_path = elastics_data_dir(elastics)
    shots_path = data_dir_path / "shots.jsonl"
    rl_shots_path = data_dir_path / "rl_shots.jsonl"
    output_dir = data_dir_path / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_filename = f"{output_name}.pt"
    onnx_filename = f"{output_name}.onnx"

    # Build dataset
    dataset = ShotDataset(shots_path, rl_shots_path, data_dir_path, temperature)
    if len(dataset) == 0:
        click.echo(
            "No usable samples found. Check that shots.jsonl exists and "
            "labelled images are in data/<split>/images/ with labels in "
            f"data/<split>/labels/ under {data_dir_path}.",
            err=True,
        )
        sys.exit(1)

    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Instantiate model and load weights
    if model == "v1":
        from vision.models.pong_model import PongAimModelV1

        aim_model = PongAimModelV1()
    elif model == "v3":
        from vision.models.pong_model import PongAimModelV3

        aim_model = PongAimModelV3()
    else:
        from vision.models.pong_model import PongAimModelV2

        aim_model = PongAimModelV2()
    if resume:
        # Resume: load full checkpoint (backbone + head)
        state = torch.load(resume, map_location="cpu", weights_only=True)
        aim_model.load_state_dict(state, strict=False)
        logger.info("Resumed from full checkpoint %s", resume)
    elif backbone:
        # Fresh head on pre-trained backbone
        aim_model.load_backbone(backbone)
        logger.info("Backbone loaded from %s - head randomly initialised", backbone)
    else:
        logger.info("No checkpoint provided - training from scratch")

    if unfreeze_backbone:
        aim_model.unfreeze_backbone()
        n_backbone = sum(p.numel() for p in aim_model.backbone.parameters())
        logger.info(
            "Backbone UNFROZEN for end-to-end fine-tuning (%s params at lr=%.2e)",
            f"{n_backbone:,}",
            lr * backbone_lr_scale,
        )
    else:
        aim_model.freeze_backbone()
        logger.info("Backbone frozen")

    n_head = sum(p.numel() for p in aim_model.head_parameters())
    logger.info(
        "Model: %s  Head params: %s  Device: %s  Temperature: %.0f",
        type(aim_model).__name__,
        f"{n_head:,}",
        device,
        temperature,
    )
    logger.info("Train: %d  Val: %d", n_train, n_val)

    # Train head (and optionally backbone)
    best_ckpt = _train_head(
        aim_model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=lr,
        device=torch.device(device),
        output_dir=output_dir,
        unfreeze_backbone=unfreeze_backbone,
        backbone_lr_scale=backbone_lr_scale,
        loss=loss,
        ckpt_filename=ckpt_filename,
    )

    # Export full end-to-end ONNX model
    try:
        onnx_path = _export_end_to_end_onnx(
            aim_model, best_ckpt, output_dir, onnx_filename=onnx_filename
        )
        click.echo(f"\nDone. Checkpoint: {best_ckpt}")
        click.echo(f"      ONNX:       {onnx_path}")
    except Exception as e:
        click.echo(f"\nDone. Checkpoint: {best_ckpt}")
        click.echo(f"      ONNX export failed: {e}")
        click.echo("      Install onnx/onnxscript or retry export later.")


if __name__ == "__main__":
    cli()
