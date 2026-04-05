"""
vision/transforms.py - Image preprocessing and augmentation for the detector pipeline.
"""

from __future__ import annotations

import cv2
import numpy as np

# Input size expected by the detector (and any model trained with this pipeline).
INPUT_SIZE: tuple[int, int] = (640, 640)


def preprocess(frame: np.ndarray) -> np.ndarray:
    """Resize to INPUT_SIZE, BGR->RGB, normalise to [0,1], transpose to CHW. Returns float32 [3,640,640]."""
    resized = cv2.resize(frame, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normed = rgb.astype(np.float32) / 255.0
    chw = normed.transpose(2, 0, 1)  # HWC -> CHW
    return chw


def augment(
    image: np.ndarray,
    boxes: np.ndarray,
    p_flip: float = 0.5,
    p_jitter: float = 0.5,
    p_scale_crop: float = 0.5,
    p_translate: float = 0.3,
    p_noise: float = 0.3,
    p_blur: float = 0.2,
    p_cutout: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply random augmentations to a preprocessed image and its bounding box.

    Used during detector training only. Preserves (cx, cy, w, h) normalised format.
    Augmentations: horizontal flip, scale+crop, translation, colour jitter,
    Gaussian noise, Gaussian blur, random cutout.
    """
    image = image.copy()
    boxes = boxes.copy()
    _, H, W = image.shape
    is_negative = boxes[2] < 1e-6 and boxes[3] < 1e-6  # no box = negative

    # --- Random horizontal flip ---
    if np.random.random() < p_flip:
        image = image[:, :, ::-1].copy()
        if not is_negative:
            boxes[0] = 1.0 - boxes[0]

    # --- Random scale + crop ---
    if np.random.random() < p_scale_crop and not is_negative:
        scale = np.random.uniform(0.8, 1.2)
        new_H, new_W = int(H * scale), int(W * scale)

        # Resize each channel (image is CHW)
        scaled = np.stack(
            [
                cv2.resize(image[c], (new_W, new_H), interpolation=cv2.INTER_LINEAR)
                for c in range(3)
            ]
        )

        if scale > 1.0:
            # Crop back to original size - random crop position
            y0 = np.random.randint(0, new_H - H + 1)
            x0 = np.random.randint(0, new_W - W + 1)
            image = scaled[:, y0 : y0 + H, x0 : x0 + W]
            # Adjust bbox: shift by crop offset, scale by 1/scale
            boxes[0] = boxes[0] * scale - x0 / W
            boxes[1] = boxes[1] * scale - y0 / H
            boxes[2] = boxes[2] * scale
            boxes[3] = boxes[3] * scale
        else:
            # Pad to original size - centre the scaled image
            padded = np.zeros_like(image)
            y0 = (H - new_H) // 2
            x0 = (W - new_W) // 2
            padded[:, y0 : y0 + new_H, x0 : x0 + new_W] = scaled
            image = padded
            # Adjust bbox
            boxes[0] = boxes[0] * scale + x0 / W
            boxes[1] = boxes[1] * scale + y0 / H
            boxes[2] = boxes[2] * scale
            boxes[3] = boxes[3] * scale

        # Clamp box to valid range
        boxes = np.clip(boxes, 0.0, 1.0)

    # --- Random translation ---
    if np.random.random() < p_translate and not is_negative:
        max_shift = 0.15
        tx = np.random.uniform(-max_shift, max_shift)
        ty = np.random.uniform(-max_shift, max_shift)
        tx_px = int(tx * W)
        ty_px = int(ty * H)

        shifted = np.zeros_like(image)
        # Compute source and destination slices
        src_y0 = max(0, -ty_px)
        src_y1 = min(H, H - ty_px)
        src_x0 = max(0, -tx_px)
        src_x1 = min(W, W - tx_px)
        dst_y0 = max(0, ty_px)
        dst_x0 = max(0, tx_px)
        h_copy = src_y1 - src_y0
        w_copy = src_x1 - src_x0
        shifted[:, dst_y0 : dst_y0 + h_copy, dst_x0 : dst_x0 + w_copy] = image[
            :, src_y0 : src_y0 + h_copy, src_x0 : src_x0 + w_copy
        ]
        image = shifted

        boxes[0] = boxes[0] + tx
        boxes[1] = boxes[1] + ty
        boxes = np.clip(boxes, 0.0, 1.0)

    # --- Colour jitter (brightness, contrast, hue, saturation) ---
    if np.random.random() < p_jitter:
        # Brightness
        brightness = np.random.uniform(-0.2, 0.2)
        image = np.clip(image + brightness, 0.0, 1.0)

        # Contrast
        contrast = np.random.uniform(0.7, 1.3)
        image = np.clip((image - 0.5) * contrast + 0.5, 0.0, 1.0)

        # Hue shift (rotate RGB channels slightly)
        if np.random.random() < 0.5:
            hue_shift = np.random.uniform(-0.05, 0.05)
            # Simple approximation: shift each channel differently
            for c in range(3):
                shift = hue_shift * (1.0 if c == 0 else (-0.5 if c == 1 else -0.5))
                image[c] = np.clip(image[c] + shift, 0.0, 1.0)

        # Saturation
        if np.random.random() < 0.5:
            sat_factor = np.random.uniform(0.7, 1.3)
            gray = image.mean(axis=0, keepdims=True)  # [1, H, W]
            image = np.clip(gray + sat_factor * (image - gray), 0.0, 1.0)

    # --- Gaussian noise ---
    if np.random.random() < p_noise:
        sigma = np.random.uniform(0.01, 0.03)
        noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
        image = np.clip(image + noise, 0.0, 1.0)

    # --- Gaussian blur ---
    if np.random.random() < p_blur:
        ksize = np.random.choice([3, 5])
        sigma = np.random.uniform(0.5, 1.5)
        for c in range(3):
            image[c] = cv2.GaussianBlur(
                image[c],
                (ksize, ksize),
                sigma,
            )

    # --- Random cutout / erasing ---
    if np.random.random() < p_cutout:
        # Erase a random rectangle (10-30% of image area)
        erase_h = int(H * np.random.uniform(0.1, 0.3))
        erase_w = int(W * np.random.uniform(0.1, 0.3))
        y0 = np.random.randint(0, H - erase_h + 1)
        x0 = np.random.randint(0, W - erase_w + 1)
        image[:, y0 : y0 + erase_h, x0 : x0 + erase_w] = 0.0

    return image, boxes

