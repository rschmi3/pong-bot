"""
rl/policy.py - Inner and outer GRU policies for the RL shot loop.

Inner policies (HeuristicPolicy, GRUPolicy) run one cup search per episode.
OuterGRUPolicy recommends a warm start position for each inner search
based on cups already found in the current outer session.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class Policy(ABC):
    """Abstract base class for inner (per-cup) shot adjustment policies."""

    def begin_episode(
        self, cup_num: int, start_x: int, start_y: int
    ) -> tuple[int, int]:
        """
        Called before the first shot of an inner session.

        Returns the starting (x, y) position to use for this session.
        Default: return the provided start unchanged.
        Subclasses may override to reset per-episode state.
        """
        return start_x, start_y

    @abstractmethod
    def select_action(self, x_steps: int, y_steps: int) -> tuple[int, int]:
        """Return the next target (x_steps, y_steps) from the current position."""
        ...

    @abstractmethod
    def update(
        self, result, prev_pos: tuple[int, int], new_pos: tuple[int, int]
    ) -> None:
        """Update policy state after receiving the shot result."""
        ...

    def end_episode(self) -> None:
        """
        Called at the end of each inner session (hit or timeout).
        Default is a no-op. Subclasses override to commit trajectory buffers.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Save policy state to disk."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Load policy state from disk."""
        ...


# ---------------------------------------------------------------------------
# Action space bounds - single source of truth, imported from robot.py
# ---------------------------------------------------------------------------

from motor_control.robot import AXIS_MAX_STEPS, AXIS_MIN_STEPS  # noqa: E402

X_MIN: int = AXIS_MIN_STEPS["X"]
X_MAX: int = AXIS_MAX_STEPS["X"]
Y_MIN: int = AXIS_MIN_STEPS["Y"]
Y_MAX: int = AXIS_MAX_STEPS["Y"]

NORMALISE_STEPS: int = 50_000  # shared normalisation scale for all neural policies

# Default starting position used by all non-outer-GRU policies
DEFAULT_START_X: int = 0
DEFAULT_START_Y: int = 6500


# ---------------------------------------------------------------------------
# HeuristicPolicy
# ---------------------------------------------------------------------------


class HeuristicPolicy(Policy):
    """
    Simple rule-based binary search policy. No learning.

    Always starts from (DEFAULT_START_X, DEFAULT_START_Y) = (0, 6500).
    Each result directly adjusts the corresponding axis:
      LEFT  → increase X    RIGHT → decrease X
      SHORT → decrease Y    LONG  → increase Y

    Each axis runs independent binary search: step size halves on direction
    reversal, converging on the optimal position.
    """

    def __init__(self, initial_step: int = 2000, min_step: int = 50) -> None:
        self.initial_step = initial_step
        self.min_step = min_step
        self._reset()

    def _reset(self) -> None:
        self._step: dict[str, float] = {
            "X": float(self.initial_step),
            "Y": float(self.initial_step),
        }
        self._last_dir: dict[str, int] = {"X": 0, "Y": 0}
        self._x: int = DEFAULT_START_X
        self._y: int = DEFAULT_START_Y

    def begin_episode(
        self, cup_num: int, start_x: int, start_y: int
    ) -> tuple[int, int]:
        """Reset binary search state and return the fixed default start."""
        self._reset()
        return DEFAULT_START_X, DEFAULT_START_Y

    def _adjust_axis(self, axis: str, sign: int, new_pos: tuple[int, int]) -> None:
        if self._last_dir[axis] != 0 and self._last_dir[axis] != sign:
            self._step[axis] = max(self.min_step, self._step[axis] / 2)
            logger.info(
                "HeuristicPolicy: direction reversed on %s - step now %.0f",
                axis,
                self._step[axis],
            )
        self._last_dir[axis] = sign
        delta = int(sign * self._step[axis])
        if axis == "X":
            self._x = max(X_MIN, min(X_MAX, new_pos[0] + delta))
            self._y = new_pos[1]
        else:
            self._x = new_pos[0]
            self._y = max(Y_MIN, min(Y_MAX, new_pos[1] + delta))
        logger.info(
            "HeuristicPolicy: next X=%+d Y=%+d (delta %s %+d)",
            self._x,
            self._y,
            axis,
            delta,
        )

    def update(
        self, result, prev_pos: tuple[int, int], new_pos: tuple[int, int]
    ) -> None:
        if result.hit:
            return
        direction = result.direction
        if direction is None:
            return
        axis_map = {
            "left": ("X", +1),
            "right": ("X", -1),
            "short": ("Y", -1),
            "long": ("Y", +1),
        }
        if direction in axis_map:
            axis, sign = axis_map[direction]
            self._adjust_axis(axis, sign, new_pos)
        else:
            self._x = new_pos[0]
            self._y = new_pos[1]

    def select_action(self, x_steps: int, y_steps: int) -> tuple[int, int]:
        return self._x, self._y

    def save(self, path: str) -> None:
        import json

        state = {
            "step": self._step,
            "last_dir": self._last_dir,
            "x": self._x,
            "y": self._y,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(state, indent=2))
        logger.info("HeuristicPolicy saved to %s", path)

    def load(self, path: str) -> None:
        import json

        state = json.loads(Path(path).read_text())
        self._step = state["step"]
        self._last_dir = state["last_dir"]
        self._x = state["x"]
        self._y = state["y"]
        logger.info("HeuristicPolicy loaded from %s", path)


# ---------------------------------------------------------------------------
# GRUPolicy (inner)
# ---------------------------------------------------------------------------


class GRUPolicy(Policy):
    """GRU-based sequential inner policy. Falls back to HeuristicPolicy until weights are loaded."""

    INPUT_SIZE = 8

    def __init__(
        self,
        hidden_size: int = 64,
        sigma: float = 0.010,
        sigma_decay: float = 0.995,
        sigma_min: float = 0.005,
        max_shots: int = 20,
    ) -> None:
        self.hidden_size = hidden_size
        self.sigma_init = sigma
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.sigma_min = sigma_min
        self.max_shots = max_shots

        self._trained = False
        self._fallback = HeuristicPolicy()

        self._history: list[list[float]] = []
        self._actions: list[list[float]] = []
        self._rewards: list[float] = []
        self._shot_num: int = 0
        self._last_dir: Optional[str] = None
        self._last_hit: bool = False
        self._cup_num: int = 0

        self._init_network()
        logger.info(
            "GRUPolicy: hidden=%d sigma=%.3f trained=%s",
            hidden_size,
            sigma,
            self._trained,
        )

    def _init_network(self) -> None:
        try:
            import torch
            import torch.nn as nn

            self._torch = torch
            self.gru = nn.GRU(
                input_size=self.INPUT_SIZE,
                hidden_size=self.hidden_size,
                batch_first=True,
            )
            self.head = nn.Linear(self.hidden_size, 2)
            self._hidden: Optional[object] = None
        except ImportError:
            raise ImportError(
                "torch is required for GRUPolicy. "
                "Run in the training devShell: nix develop .#training"
            )

    def _build_state(self, x: int, y: int) -> list[float]:
        dir_vec = [0.0, 0.0, 0.0, 0.0]
        if self._last_dir == "left":
            dir_vec[0] = 1.0
        elif self._last_dir == "right":
            dir_vec[1] = 1.0
        elif self._last_dir == "short":
            dir_vec[2] = 1.0
        elif self._last_dir == "long":
            dir_vec[3] = 1.0
        return [
            x / NORMALISE_STEPS,
            y / NORMALISE_STEPS,
            dir_vec[0],
            dir_vec[1],
            dir_vec[2],
            dir_vec[3],
            float(self._last_hit),
            self._shot_num / self.max_shots,
        ]

    def begin_episode(
        self, cup_num: int, start_x: int, start_y: int
    ) -> tuple[int, int]:
        """Reset per-session state. Returns start_x, start_y unchanged."""
        self._cup_num = cup_num
        self._history = []
        self._actions = []
        self._rewards = []
        self._shot_num = 0
        self._last_dir = None
        self._last_hit = False
        self._hidden = None
        self.sigma = self.sigma_init
        self._fallback = HeuristicPolicy()
        return start_x, start_y

    def select_action(self, x_steps: int, y_steps: int) -> tuple[int, int]:
        if not self._trained:
            return self._fallback.select_action(x_steps, y_steps)

        torch = self._torch
        state_vec = self._build_state(x_steps, y_steps)
        self._history.append(state_vec)

        inp = torch.tensor([[state_vec]], dtype=torch.float32)
        with torch.no_grad():
            out, self._hidden = self.gru(inp, self._hidden)
            mean_delta = self.head(out[0, 0])

        noise = torch.randn(2) * self.sigma
        delta = (mean_delta + noise).numpy()
        self._actions.append(delta.tolist())

        dx = int(delta[0] * NORMALISE_STEPS)
        dy = int(delta[1] * NORMALISE_STEPS)
        new_x = max(X_MIN, min(X_MAX, x_steps + dx))
        new_y = max(Y_MIN, min(Y_MAX, y_steps + dy))

        self._shot_num += 1
        self.sigma = max(self.sigma_min, self.sigma * self.sigma_decay)

        logger.info(
            "GRUPolicy: delta=(%+d, %+d) sigma=%.4f → (%+d, %+d)",
            dx,
            dy,
            self.sigma,
            new_x,
            new_y,
        )
        return new_x, new_y

    def update(
        self, result, prev_pos: tuple[int, int], new_pos: tuple[int, int]
    ) -> None:
        if not self._trained:
            self._fallback.update(result, prev_pos, new_pos)
        reward = 1.0 if result.hit else -0.05
        self._rewards.append(reward)
        self._last_dir = result.direction
        self._last_hit = result.hit

    def end_episode(self) -> None:
        """Reset per-session state."""
        # State reset happens in begin_episode for the next cup, but also
        # reset here so end_episode is safe to call without a following begin.
        self._history = []
        self._actions = []
        self._rewards = []
        self._shot_num = 0
        self._last_dir = None
        self._last_hit = False
        self._hidden = None
        self.sigma = self.sigma_init

    def save(self, path: str) -> None:
        torch = self._torch
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "gru": self.gru.state_dict(),
                "head": self.head.state_dict(),
                "sigma": self.sigma,
                "trained": self._trained,
                "hidden_size": self.hidden_size,
                "sigma_init": self.sigma_init,
                "sigma_min": self.sigma_min,
                "sigma_decay": self.sigma_decay,
                "max_shots": self.max_shots,
                # Runtime state for mid-session resume
                "rt_hidden": (
                    self._hidden.detach().cpu() if self._hidden is not None else None
                ),
                "rt_last_dir": self._last_dir,
                "rt_last_hit": self._last_hit,
                "rt_shot_num": self._shot_num,
            },
            path,
        )
        logger.info(
            "GRUPolicy saved to %s (trained=%s)",
            path,
            self._trained,
        )

    def load(self, path: str) -> None:
        torch = self._torch
        state = torch.load(path, map_location="cpu", weights_only=False)
        saved_hidden = state.get("hidden_size", self.hidden_size)
        if saved_hidden != self.hidden_size:
            self.hidden_size = saved_hidden
            self._init_network()
        self.gru.load_state_dict(state["gru"])
        self.head.load_state_dict(state["head"])
        self.sigma = state.get("sigma", self.sigma)
        self.sigma_init = state.get("sigma_init", self.sigma_init)
        self.sigma_min = state.get("sigma_min", self.sigma_min)
        self.sigma_decay = state.get("sigma_decay", self.sigma_decay)
        self.max_shots = state.get("max_shots", self.max_shots)
        self._trained = state.get("trained", False)
        # Runtime state for mid-session resume
        self._hidden = state.get("rt_hidden")
        self._last_dir = state.get("rt_last_dir")
        self._last_hit = state.get("rt_last_hit", False)
        self._shot_num = state.get("rt_shot_num", 0)
        logger.info(
            "GRUPolicy loaded from %s (trained=%s, sigma=%.4f)",
            path,
            self._trained,
            self.sigma,
        )


# ---------------------------------------------------------------------------
# OuterGRUPolicy
# ---------------------------------------------------------------------------


class OuterGRUPolicy:
    """
    Outer GRU policy: recommends a warm start position for each inner search
    based on the history of cups already found in the current outer session.

    Input per timestep (7-dim): winning position, shots taken, start position,
    mean miss position. Output: (start_x_norm, start_y_norm) for the next search.
    """

    INPUT_SIZE = 7

    def __init__(
        self,
        hidden_size: int = 32,
        sigma: float = 0.05,
        sigma_decay: float = 0.99,
        sigma_min: float = 0.005,
        max_shots_per_cup: int = 20,
    ) -> None:
        self.hidden_size = hidden_size
        self.sigma_init = sigma
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.sigma_min = sigma_min
        self.max_shots_per_cup = max_shots_per_cup

        self._trained = False

        # Per-outer-session state
        self._history: list[list[float]] = []  # input vecs (one per cup found)
        self._starts: list[list[float]] = []  # recommended starts (actions)
        self._rewards: list[float] = []  # per-cup reward = -shots_taken_norm
        self._hidden: Optional[object] = None  # GRU hidden state
        self._last_out: Optional[object] = None  # last GRU output for select_start

        self._init_network()
        logger.info(
            "OuterGRUPolicy: hidden=%d sigma=%.3f trained=%s",
            hidden_size,
            sigma,
            self._trained,
        )

    def _init_network(self) -> None:
        try:
            import torch
            import torch.nn as nn

            self._torch = torch
            self.gru = nn.GRU(
                input_size=self.INPUT_SIZE,
                hidden_size=self.hidden_size,
                batch_first=True,
            )
            self.head = nn.Linear(self.hidden_size, 2)
            self._hidden = None
        except ImportError:
            raise ImportError(
                "torch is required for OuterGRUPolicy. "
                "Run in the training devShell: nix develop .#training"
            )

    def _default_start(self) -> tuple[int, int]:
        """Fallback start when no recommendation available: centre of table."""
        return DEFAULT_START_X, DEFAULT_START_Y

    def select_start(self, cup_num: int) -> tuple[int, int]:
        """Recommend a starting position for cup_num (1-indexed). Returns default for cup 1 or if untrained."""
        if not self._trained or self._last_out is None:
            # No prior cup found yet or untrained - use default start
            sx, sy = self._default_start()
            logger.info(
                "OuterGRUPolicy: cup %d - no history, using default (%+d, %+d)",
                cup_num,
                sx,
                sy,
            )
            return sx, sy

        torch = self._torch
        # Apply head to the GRU output stored by the last update() call.
        # Do NOT run another GRU step - update() already advanced the hidden state.
        with torch.no_grad():
            mean_start = self.head(self._last_out)

        noise = torch.randn(2) * self.sigma
        start = (mean_start + noise).numpy()
        self._starts.append(start.tolist())

        sx = int(np.clip(start[0] * NORMALISE_STEPS, X_MIN, X_MAX))
        sy = int(np.clip(start[1] * NORMALISE_STEPS, Y_MIN, Y_MAX))

        self.sigma = max(self.sigma_min, self.sigma * self.sigma_decay)
        logger.info(
            "OuterGRUPolicy: cup %d → recommended start (%+d, %+d) sigma=%.4f",
            cup_num,
            sx,
            sy,
            self.sigma,
        )
        return sx, sy

    def update(
        self,
        winning_x: int,
        winning_y: int,
        shots_taken: int,
        start_x: int,
        start_y: int,
        mean_miss_x: int,
        mean_miss_y: int,
    ) -> None:
        """Update GRU hidden state after an inner session completes. For timeouts, pass mean_miss as winning pos."""
        torch = self._torch
        vec = [
            winning_x / NORMALISE_STEPS,
            winning_y / NORMALISE_STEPS,
            shots_taken / self.max_shots_per_cup,
            start_x / NORMALISE_STEPS,
            start_y / NORMALISE_STEPS,
            mean_miss_x / NORMALISE_STEPS,
            mean_miss_y / NORMALISE_STEPS,
        ]
        self._history.append(vec)

        # Reward: negative normalised shots (fewer shots = better recommendation)
        self._rewards.append(-shots_taken / self.max_shots_per_cup)

        # Advance hidden state with this cup's outcome and store the output
        # so select_start() can apply the head without re-stepping the GRU.
        inp = torch.tensor([[vec]], dtype=torch.float32)
        with torch.no_grad():
            out, self._hidden = self.gru(inp, self._hidden)
            self._last_out = out[0, 0]  # shape (hidden_size,)

        logger.info(
            "OuterGRUPolicy: cup found at (%+d, %+d) in %d shots "
            "(started at (%+d, %+d))",
            winning_x,
            winning_y,
            shots_taken,
            start_x,
            start_y,
        )

    def end_outer_episode(self) -> None:
        """Reset per-outer-session state."""
        self._history = []
        self._starts = []
        self._rewards = []
        self._hidden = None
        self._last_out = None
        self.sigma = self.sigma_init

    def save(self, path: str) -> None:
        torch = self._torch
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "gru": self.gru.state_dict(),
                "head": self.head.state_dict(),
                "sigma": self.sigma,
                "sigma_init": self.sigma_init,
                "sigma_min": self.sigma_min,
                "sigma_decay": self.sigma_decay,
                "max_shots_per_cup": self.max_shots_per_cup,
                "hidden_size": self.hidden_size,
                "trained": self._trained,
                # Runtime state for mid-session resume
                "rt_hidden": (
                    self._hidden.detach().cpu() if self._hidden is not None else None
                ),
                "rt_last_out": (
                    self._last_out.detach().cpu()
                    if self._last_out is not None
                    else None
                ),
                "rt_history": self._history,
                "rt_starts": self._starts,
                "rt_rewards": self._rewards,
            },
            path,
        )
        logger.info(
            "OuterGRUPolicy saved to %s (trained=%s)",
            path,
            self._trained,
        )

    def load(self, path: str) -> None:
        torch = self._torch
        state = torch.load(path, map_location="cpu", weights_only=False)
        saved_hidden = state.get("hidden_size", self.hidden_size)
        if saved_hidden != self.hidden_size:
            self.hidden_size = saved_hidden
            self._init_network()
        self.gru.load_state_dict(state["gru"])
        self.head.load_state_dict(state["head"])
        self.sigma = state.get("sigma", self.sigma)
        self.sigma_init = state.get("sigma_init", self.sigma_init)
        self.sigma_min = state.get("sigma_min", self.sigma_min)
        self.sigma_decay = state.get("sigma_decay", self.sigma_decay)
        self.max_shots_per_cup = state.get("max_shots_per_cup", self.max_shots_per_cup)
        self._trained = state.get("trained", False)
        # Runtime state for mid-session resume
        self._hidden = state.get("rt_hidden")
        self._last_out = state.get("rt_last_out")
        self._history = state.get("rt_history", [])
        self._starts = state.get("rt_starts", [])
        self._rewards = state.get("rt_rewards", [])
        logger.info(
            "OuterGRUPolicy loaded from %s (trained=%s)",
            path,
            self._trained,
        )
