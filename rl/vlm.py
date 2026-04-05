"""
vision/vlm.py - Qwen3-VL-4B shot classifier.

Uses a locally-running Qwen3-VL-4B-Instruct model to classify the outcome
of a shot from a short video clip (list of frames captured by StreamReceiver).

No training required - the model is used as-is with a structured prompt.
The VLM already understands spatial relationships and motion from its
pretraining; we only need to tell it what to look for and what format
to reply in.

VRAM usage: ~8GB (Qwen3-VL-4B at float16), leaving ~4GB free on a 3080 Ti.

Usage
-----
    classifier = ShotClassifier()  # loads model once
    frames = receiver.receive()    # list of BGR np.ndarray from StreamReceiver
    result = classifier.classify(frames)
    print(result.hit, result.direction)  # True / None or "left"/"right"/"long"/"short"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Maximum frames to pass to the model. 48 is safe at 640×480 on a 12GB GPU
# (3080 Ti) with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
# Frames are selected using motion-based scoring to prioritise frames
# containing the ball rather than uniform subsampling.
NUM_FRAMES = 48

# Structured prompt - explicit format instruction is critical for reliable parsing.
PROMPT = (
    "You are analysing a short video clip from a robotic ping pong ball launcher.\n\n"
    "Scene: the camera is mounted on the robot at one end of a 6-foot folding "
    "table. A single RED solo cup sits at the far end of the table, approximately "
    "5.5 feet away.\n\n"
    "The ball: a YELLOW ping pong ball launched from near the camera. It will "
    "appear as a small yellow blur or streak starting from near the top-centre "
    "of the frame and arcing downward toward the far end of the table. Focus on "
    "where the ball's arc ENDS relative to the cup.\n\n"
    "CRITICAL: do NOT classify based on which side of the frame the cup is on. "
    "The cup may appear on the left, right, or centre of the frame - its "
    "position in the frame is irrelevant. Classify ONLY based on where the "
    "BALL LANDS relative to the CUP.\n\n"
    "Classification tokens:\n"
    "  HIT          - the ball landed inside the cup\n"
    "  MISS:LEFT    - the ball landed to the LEFT of the cup\n"
    "  MISS:RIGHT   - the ball landed to the RIGHT of the cup\n"
    "  MISS:LONG    - the ball flew past the cup or off the far end of the table\n"
    "  MISS:SHORT   - the ball landed on the table surface before reaching "
    "the cup\n\n"
    "Rules:\n"
    "1. MISS:SHORT only applies if the ball visibly lands ON THE TABLE SURFACE "
    "before the cup. If the ball flies off the table or its landing is not "
    "visible on the table, it is NOT short.\n"
    "2. If the ball clearly missed left or right of the cup, always prefer "
    "MISS:LEFT or MISS:RIGHT over MISS:SHORT or MISS:LONG.\n"
    "3. If the ball trajectory is unclear or the ball is not visible, use the "
    "launch direction visible in the first frames as your best guide.\n\n"
    "First, in 2-3 sentences describe what you observe: where the yellow ball "
    "goes and where it lands relative to the red cup. Then on the final line, "
    "write only the classification token and nothing else."
)

Direction = Literal["left", "right", "long", "short"]


@dataclass
class ShotResult:
    """
    Result of VLM shot classification.

    Attributes
    ----------
    hit : bool
        True if the ball went into the cup.
    direction : str or None
        For misses: 'left', 'right', 'long', or 'short'.
        None if it was a hit or the VLM response was ambiguous.
    confidence : float
        Placeholder confidence value. Qwen2-VL does not expose token
        probabilities easily - this is set to 1.0 for unambiguous parses
        and 0.5 for fallback parses. Use it as a rough quality signal.
    raw_response : str
        The raw text response from the model for debugging.
    """

    hit: bool
    direction: Direction | None
    confidence: float
    raw_response: str

    def __str__(self) -> str:
        if self.hit:
            return "HIT"
        return f"MISS:{self.direction.upper()}" if self.direction else "MISS:UNKNOWN"


class ShotClassifier:
    """
    Qwen3-VL-4B shot outcome classifier.

    Loads the model once at construction time and reuses it for all
    subsequent classify() calls.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID. Defaults to Qwen3-VL-4B-Instruct (~8GB VRAM).
    device : str
        Torch device. 'auto' (default) selects CUDA if available, else CPU.
        Pass 'cpu' explicitly to force CPU inference (slow but functional).
        Pass 'cuda' to force GPU.
    debug_frames_dir : str or None
        If set, save the subsampled frames passed to the model as JPEGs in
        this directory for debugging. Files are named frame_000.jpg etc and
        overwritten on each classify() call. Disabled when None (default).
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
        device: str = "auto",
        debug_frames_dir: str | None = None,
    ) -> None:
        self.model_id = model_id
        self._device_arg = device
        self._debug_frames_dir = debug_frames_dir
        self._model = None
        self._processor = None
        self._load_model()

    def _resolve_device(self) -> str:
        """Resolve 'auto' to 'cuda' or 'cpu' based on availability."""
        if self._device_arg != "auto":
            return self._device_arg
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self) -> None:
        """Load Qwen3-VL model and processor."""
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        device = self._resolve_device()
        # float16 is not supported on CPU - use float32 for CPU inference.
        dtype = torch.float16 if device != "cpu" else torch.float32

        logger.info(
            "Loading VLM: %s on %s dtype=%s (this may take a moment...)",
            self.model_id,
            device,
            dtype,
        )

        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self._model.eval()
        self._device = device
        logger.info("VLM loaded on %s", device)

    def classify(self, frames: list[np.ndarray]) -> ShotResult:
        """
        Classify a shot from a list of BGR video frames.

        Parameters
        ----------
        frames : list[np.ndarray]
            BGR frames from StreamReceiver. Any resolution - will be
            resized to 448×448 before passing to the model.

        Returns
        -------
        ShotResult
        """
        if not frames:
            logger.warning(
                "classify() called with empty frame list - returning MISS:UNKNOWN"
            )
            return ShotResult(
                hit=False, direction=None, confidence=0.0, raw_response=""
            )

        # Select NUM_FRAMES using motion-based scoring to prioritise frames
        # where the ball is visible, supplemented by uniform scene-context frames.
        sampled = self._select_frames(frames, NUM_FRAMES)

        # Debug: save the exact frames the model will see as JPEGs.
        if self._debug_frames_dir:
            import pathlib
            dbg = pathlib.Path(self._debug_frames_dir)
            dbg.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(sampled):
                cv2.imwrite(str(dbg / f"frame_{i:03d}.jpg"), f)
            logger.info("Debug frames saved to %s", self._debug_frames_dir)

        # Convert BGR → RGB PIL images
        from PIL import Image as PILImage
        from qwen_vl_utils import process_vision_info

        pil_frames = [
            PILImage.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in sampled
        ]

        # Build the message in Qwen3-VL chat format.
        # fps     - used by qwen_vl_utils for video file inputs
        # sample_fps - used by fetch_video for list-of-frames inputs; sets
        #              metadata.fps which the processor needs for timestamps
        content = [{"type": "video", "video": pil_frames, "fps": 30.0, "sample_fps": 30.0}]
        content.append({"type": "text", "text": PROMPT})

        messages = [{"role": "user", "content": content}]

        # Tokenise
        import torch

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Qwen3-VL requires image_patch_size=16 and return_video_metadata=True.
        # Videos are returned as (frames, metadata) tuples that must be unzipped.
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        # When return_video_metadata=True, video_inputs is a list of
        # (frames_tensor, metadata_dict) tuples - unzip to keep both.
        # The metadata dicts (fps, frames_indices, total_num_frames) are
        # passed to the processor so it can construct VideoMetadata objects
        # with the correct fps for frame timestamp computation.
        if video_inputs is not None:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs = list(video_inputs)
            video_metadatas = list(video_metadatas)
        else:
            video_metadatas = None

        # Drop fps from video_kwargs - it comes back as a list from
        # qwen_vl_utils but is already encoded in video_metadatas.
        video_kwargs.pop("fps", None)

        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadatas,
            return_tensors="pt",
            padding=True,
            **video_kwargs,
        )

        # Expand video_grid_thw: the processor produces [[T, H, W]] (one row
        # per video), but Qwen3-VL's processor inserts per-frame timestamps
        # in the text which creates T separate type-2 token groups.
        # get_rope_index() iterates grid_thw once per group, so we need
        # T rows of [[1, H, W]] instead of one row of [[T, H, W]].
        if "video_grid_thw" in inputs and inputs["video_grid_thw"] is not None:
            import torch

            orig = inputs["video_grid_thw"]
            expanded = []
            for row in orig:
                t, h, w = row.tolist()
                for _ in range(int(t)):
                    expanded.append([1, h, w])
            inputs["video_grid_thw"] = torch.tensor(
                expanded, dtype=orig.dtype, device=orig.device
            )

        inputs = inputs.to(self._device)

        # Generate
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=256,  # allow reasoning text before classification
                do_sample=False,
            )

        # Decode - strip the input tokens from the output
        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[0][input_len:]
        raw_response = self._processor.decode(
            generated, skip_special_tokens=True
        ).strip()

        logger.info("VLM raw response: %r", raw_response)
        return self._parse(raw_response)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _select_frames(frames: list[np.ndarray], n: int) -> list[np.ndarray]:
        """
        Select n frames prioritising high-motion frames (ball in flight).

        Strategy:
        - Compute per-frame motion score = mean absolute diff vs previous frame
        - Take top n//2 highest-motion frames (captures ball trajectory)
        - Fill remaining n//2 slots with evenly-spaced frames (scene context)
        - Deduplicate and sort by original frame index

        This guarantees that the frames containing the fast-moving ball are
        always selected, regardless of where they fall in the clip.
        """
        if len(frames) <= n:
            return frames

        # Motion scores: mean absolute pixel diff vs previous frame
        scores = [0.0]  # first frame has no predecessor
        for i in range(1, len(frames)):
            diff = np.abs(frames[i].astype(np.float32) - frames[i - 1].astype(np.float32))
            scores.append(float(diff.mean()))

        # Top n//2 highest-motion indices
        n_motion = n // 2
        n_uniform = n - n_motion

        motion_indices = sorted(
            range(len(frames)),
            key=lambda i: scores[i],
            reverse=True,
        )[:n_motion]

        # n_uniform evenly-spaced indices for scene context
        uniform_indices = [
            int(i * (len(frames) - 1) / (n_uniform - 1))
            for i in range(n_uniform)
        ]

        # Merge, deduplicate, sort by index to preserve temporal order
        selected = sorted(set(motion_indices) | set(uniform_indices))
        return [frames[i] for i in selected]

    @staticmethod
    def _parse(response: str) -> ShotResult:
        """
        Parse the model's text response into a ShotResult.

        With thinking mode the model produces 2-3 sentences of reasoning
        followed by the classification token on the final line. We scan
        from the end of the response to find the first valid token,
        ignoring any reasoning text before it.
        """
        direction_map = {
            "MISS:LEFT": "left",
            "MISS:RIGHT": "right",
            "MISS:LONG": "long",
            "MISS:SHORT": "short",
        }

        # Scan from the end - find the last line that is a valid token
        lines = response.strip().splitlines()
        for line in reversed(lines):
            token = line.strip().upper()
            if token == "HIT":
                return ShotResult(
                    hit=True, direction=None, confidence=1.0, raw_response=response
                )
            if token in direction_map:
                return ShotResult(
                    hit=False,
                    direction=direction_map[token],
                    confidence=1.0,
                    raw_response=response,
                )

        # Fuzzy fallback - scan full response for keywords
        upper = response.upper()
        for token, direction in [
            ("MISS:LEFT", "left"),
            ("MISS:RIGHT", "right"),
            ("MISS:LONG", "long"),
            ("MISS:SHORT", "short"),
        ]:
            if token in upper:
                logger.warning(
                    "VLM response did not end with token - fuzzy matched %s",
                    direction,
                )
                return ShotResult(
                    hit=False,
                    direction=direction,
                    confidence=0.5,
                    raw_response=response,
                )

        if "HIT" in upper:
            logger.warning("VLM response fuzzy matched HIT")
            return ShotResult(
                hit=True, direction=None, confidence=0.5, raw_response=response
            )

        logger.warning(
            "VLM response %r could not be parsed - returning MISS:UNKNOWN", response
        )
        return ShotResult(
            hit=False, direction=None, confidence=0.0, raw_response=response
        )
