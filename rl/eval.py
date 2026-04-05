"""
training/eval_policy.py - pong-eval: offline policy evaluation via simulation.

Loads cup positions from historical session_end records, then simulates full
episodes for one or more policies without firing the robot. At each simulated
shot, the true direction is computed from first principles by comparing the
policy's chosen position to the known cup position.

Direction convention (matches HeuristicPolicy and training data):
  wx > x  → ball missed left  → direction = "left"   → need +x
  wx < x  → ball missed right → direction = "right"  → need -x
  wy > y  → ball went long    → direction = "long"   → need +y (higher arc = shorter)
  wy < y  → ball fell short   → direction = "short"  → need -y (lower arc = longer)

Dominant axis is determined by normalising each axis error by its total range
(20000 steps for both X and Y) before comparing.

Usage
-----
    pong-eval --elastics 5
    pong-eval --elastics 5 --sigma 0.0 --sigma 0.01
    pong-eval --elastics 5 --verbose
    pong-eval --elastics 5 --no-heuristic
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import click

logger = logging.getLogger(__name__)

# Axis ranges - used for normalised dominance comparison
X_RANGE: int = 20_000   # -10000 to +10000
Y_RANGE: int = 20_000   # 0 to 20000

DEFAULT_HIT_TOLERANCE: int = 500   # steps in each axis
DEFAULT_MAX_SHOTS:     int = 20
DEFAULT_SOURCE_POLICY: str = "heuristic"


# ---------------------------------------------------------------------------
# Simulated result object - matches the interface expected by policy.update()
# ---------------------------------------------------------------------------


@dataclass
class _SimResult:
    hit: bool
    direction: str | None
    raw_response: str = ""
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Core simulation helpers
# ---------------------------------------------------------------------------


def _sim_direction(
    x: int,
    y: int,
    wx: int,
    wy: int,
) -> str:
    """
    Compute the dominant miss direction given current position (x, y)
    and cup position (wx, wy).

    Normalises both axes by their total range before comparing dominance
    so that X and Y errors are on equal footing.
    """
    dx = wx - x   # positive = cup is to the right = ball missed left
    dy = wy - y   # positive = cup is farther away = ball went long

    dx_norm = abs(dx) / X_RANGE
    dy_norm = abs(dy) / Y_RANGE

    if dx_norm >= dy_norm:
        return "left" if dx > 0 else "right"
    else:
        return "long" if dy > 0 else "short"


def _is_hit(x: int, y: int, wx: int, wy: int, tol: int) -> bool:
    return abs(x - wx) <= tol and abs(y - wy) <= tol


def _sim_episode(
    policy,
    wx: int,
    wy: int,
    max_shots: int,
    hit_tolerance: int,
    sigma_override: float | None,
    seed_offset: int = 0,
) -> dict:
    """
    Simulate one episode for a cup at (wx, wy).

    Returns a dict:
        hit        : bool
        shots      : int (shots fired, including the hit shot)
        trajectory : list of (x, y, direction, hit) tuples
    """
    import torch
    import numpy as np

    # Per-episode reproducibility - combine global seed with offset
    torch.manual_seed(42 + seed_offset)
    np.random.seed(42 + seed_offset)

    # Apply sigma override before begin_episode (which resets sigma to sigma_init)
    if sigma_override is not None and hasattr(policy, "sigma_init"):
        policy.sigma_init = sigma_override

    from rl.policy import DEFAULT_START_X, DEFAULT_START_Y
    policy.begin_episode(cup_num=-1, start_x=DEFAULT_START_X, start_y=DEFAULT_START_Y)

    x, y = DEFAULT_START_X, DEFAULT_START_Y
    trajectory: list[tuple[int, int, str | None, bool]] = []
    hit = False

    for _ in range(max_shots):
        new_x, new_y = policy.select_action(x, y)

        hit = _is_hit(new_x, new_y, wx, wy, hit_tolerance)
        direction = None if hit else _sim_direction(new_x, new_y, wx, wy)

        trajectory.append((new_x, new_y, direction, hit))

        result = _SimResult(hit=hit, direction=direction)
        policy.update(result, (x, y), (new_x, new_y))

        x, y = new_x, new_y

        if hit:
            break

    policy.end_episode()
    return {
        "hit":        hit,
        "shots":      len(trajectory),
        "trajectory": trajectory,
    }


# ---------------------------------------------------------------------------
# Directional accuracy from simulation trajectories
# ---------------------------------------------------------------------------


def _directional_accuracy(
    results: list[dict],
    cup_positions: list[tuple[int, int]],
) -> dict[str, dict]:
    """
    Compute directional accuracy from simulation trajectories.

    For each shot (except the first in each episode, where last_dir=None),
    check whether the policy moved in the correct direction given the
    direction signal it received on the previous shot.
    """
    stats: dict[str, dict] = {
        d: {"correct": 0, "wrong": 0}
        for d in ("left", "right", "short", "long")
    }

    for ep_result, (wx, wy) in zip(results, cup_positions):
        traj = ep_result["trajectory"]
        for i in range(1, len(traj)):
            prev_x, prev_y, prev_dir, _ = traj[i - 1]
            cur_x,  cur_y,  _,        _ = traj[i]

            if prev_dir is None:
                continue

            dx = cur_x - prev_x
            dy = cur_y - prev_y

            if prev_dir == "left":
                correct = dx > 0
            elif prev_dir == "right":
                correct = dx < 0
            elif prev_dir == "short":
                correct = dy < 0
            elif prev_dir == "long":
                correct = dy > 0
            else:
                continue

            if correct:
                stats[prev_dir]["correct"] += 1
            else:
                stats[prev_dir]["wrong"] += 1

    return stats


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _print_results(
    label: str,
    results: list[dict],
    cup_positions: list[tuple[int, int]],
    verbose: bool,
    show_dir_accuracy: bool = True,
) -> None:
    hits   = [r for r in results if r["hit"]]
    misses = [r for r in results if not r["hit"]]
    n      = len(results)

    hit_rate   = len(hits) / n if n else 0.0
    mean_hits  = sum(r["shots"] for r in hits) / len(hits) if hits else float("nan")
    mean_all   = sum(r["shots"] for r in results) / n if n else float("nan")

    click.echo(f"\n--- {label} ---")
    click.echo(f"  Hit rate:          {hit_rate:.1%}  ({len(hits)}/{n})")
    click.echo(f"  Mean shots (hits): {mean_hits:.1f}")
    click.echo(f"  Mean shots (all):  {mean_all:.1f}")
    if misses:
        click.echo(f"  Timeouts:          {len(misses)}")

    if show_dir_accuracy:
        acc = _directional_accuracy(results, cup_positions)
        click.echo("  Directional accuracy:")
        for d in ("left", "right", "short", "long"):
            s = acc[d]
            total = s["correct"] + s["wrong"]
            if total > 0:
                pct = s["correct"] / total
                click.echo(
                    f"    {d:6s}: {pct:.0%}  ({s['correct']}/{total})"
                )

    if verbose:
        click.echo("\n  Per-cup breakdown:")
        click.echo(
            f"    {'(wx,wy)':<22}  {'shots':>5}  {'outcome':<7}  trajectory"
        )
        for r, (wx, wy) in zip(results, cup_positions):
            outcome = "HIT" if r["hit"] else "TIMEOUT"
            traj_str = " → ".join(
                f"({x:+d},{y:+d})[{d or 'hit'}]"
                for x, y, d, _ in r["trajectory"]
            )
            click.echo(
                f"    ({wx:+d},{wy:+d})  "
                f"{r['shots']:>5}  {outcome:<7}  {traj_str}"
            )


# ---------------------------------------------------------------------------
# Cup position loader
# ---------------------------------------------------------------------------


def _load_cup_positions(
    input_path: Path,
    source_policy: str,
) -> list[tuple[int, int]]:
    """
    Load (winning_x, winning_y) from session_end records where outcome=hit
    and policy matches source_policy.
    """
    cups = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if (
                r.get("type") == "session_end"
                and r.get("outcome") == "hit"
                and r.get("policy") == source_policy
                and r.get("winning_x") is not None
                and r.get("winning_y") is not None
            ):
                cups.append((r["winning_x"], r["winning_y"]))
    return cups


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("pong-eval")
@click.option("--elastics", required=True, type=int,
              help="Number of elastics on the launcher (e.g. 2 or 4). "
                   "Determines data directory: data/{N}_elastics/")
@click.option("--source-policy", default=DEFAULT_SOURCE_POLICY, show_default=True,
              help="Which policy's session_end records to use as cup positions.")
@click.option("--sigma", "sigmas", multiple=True, type=float,
              help="Sigma value(s) for GRU evaluation. Pass multiple times for "
                   "multiple runs. Default: [0.0, checkpoint_sigma].")
@click.option("--hit-tolerance", default=DEFAULT_HIT_TOLERANCE, show_default=True,
              type=int, help="Step tolerance for simulated hit (both axes).")
@click.option("--max-shots", default=DEFAULT_MAX_SHOTS, show_default=True,
              type=int, help="Max shots per simulated episode.")
@click.option("--no-heuristic", is_flag=True, default=False,
              help="Skip heuristic baseline evaluation.")
@click.option("--no-gru", is_flag=True, default=False,
              help="Skip GRU evaluation (only run heuristic baseline).")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Print per-cup shot-by-shot trajectories.")
def cli(
    elastics: int,
    source_policy: str,
    sigmas: tuple[float, ...],
    hit_tolerance: int,
    max_shots: int,
    no_heuristic: bool,
    no_gru: bool,
    verbose: bool,
) -> None:
    """
    Simulate policy episodes against known cup positions - no robot required.

    Loads winning positions from historical session_end records and replays
    each cup against the specified policy, computing the true direction at
    each simulated shot from first principles.
    """
    logging.basicConfig(level=logging.WARNING)

    from utils.data_dir import elastics_data_dir, checkpoint_path as _ckpt_path
    data_dir = elastics_data_dir(elastics)

    input_file = data_dir / "rl_shots.jsonl"
    if not input_file.exists():
        raise click.ClickException(f"Input file not found: {input_file}")

    cups = _load_cup_positions(input_file, source_policy)
    if not cups:
        raise click.ClickException(
            f"No hit sessions found for policy={source_policy!r} in {input_file}"
        )

    click.echo(f"Cup positions loaded: {len(cups)} from {source_policy} sessions")
    click.echo(f"Hit tolerance: ±{hit_tolerance} steps  Max shots: {max_shots}")

    # --- Heuristic baseline ---
    if not no_heuristic:
        from rl.policy import HeuristicPolicy
        heuristic = HeuristicPolicy()
        h_results = [
            _sim_episode(heuristic, wx, wy, max_shots, hit_tolerance,
                         sigma_override=None, seed_offset=i)
            for i, (wx, wy) in enumerate(cups)
        ]
        _print_results("Heuristic baseline", h_results, cups,
                       verbose=verbose, show_dir_accuracy=False)

    # --- GRU evaluation ---
    if no_gru:
        return

    gru_ckpt = _ckpt_path(data_dir, "gru")
    if not gru_ckpt.exists():
        raise click.ClickException(
            f"GRU checkpoint not found: {gru_ckpt}\n"
            f"Run 'pong-train-rl bc --elastics {elastics}' first, "
            "or use --no-gru to skip GRU evaluation."
        )

    from rl.policy import GRUPolicy
    # Load checkpoint once to read stored sigma
    probe = GRUPolicy()
    probe.load(str(gru_ckpt))
    stored_sigma = probe.sigma_init

    # Determine sigma values to evaluate
    eval_sigmas: list[float] = list(sigmas) if sigmas else [0.0, stored_sigma]
    # Deduplicate while preserving order
    seen: set[float] = set()
    unique_sigmas: list[float] = []
    for s in eval_sigmas:
        if s not in seen:
            seen.add(s)
            unique_sigmas.append(s)
    eval_sigmas = unique_sigmas

    click.echo(f"\nGRU checkpoint: {gru_ckpt}")
    click.echo(f"Stored sigma: {stored_sigma:.4f}")
    click.echo(f"Evaluating at sigma: {eval_sigmas}")

    for sigma_val in eval_sigmas:
        # Fresh policy load for each sigma run to avoid state bleed
        pol = GRUPolicy()
        pol.load(str(gru_ckpt))

        results = [
            _sim_episode(pol, wx, wy, max_shots, hit_tolerance,
                         sigma_override=sigma_val, seed_offset=i)
            for i, (wx, wy) in enumerate(cups)
        ]
        _print_results(
            f"GRU sigma={sigma_val:.4f}",
            results, cups,
            verbose=verbose,
            show_dir_accuracy=True,
        )


if __name__ == "__main__":
    cli()
