"""
rl/gen_synthetic.py - pong-gen-synthetic: generate synthetic outer sessions.

Builds outer_session_end records from heuristic inner sessions for
bootstrapping the OuterGRUPolicy via BC. Writes to synthetic_outer.jsonl.
"""

from __future__ import annotations

import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

import click

from utils.data_dir import load_jsonl as _load_jsonl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading and reconstruction
# ---------------------------------------------------------------------------


def _reconstruct_mean_miss(
    shots: list[dict],
    start_x: int,
    start_y: int,
) -> tuple[int, int]:
    """
    Compute mean_miss_x/y from all non-hit shots in a session.

    Falls back to start_x/start_y if there are no miss shots (e.g. a
    1-shot hit).
    """
    miss_shots = [s for s in shots if not s.get("hit", False)]
    if not miss_shots:
        return start_x, start_y
    mean_x = int(sum(s["x_steps"] for s in miss_shots) / len(miss_shots))
    mean_y = int(sum(s["y_steps"] for s in miss_shots) / len(miss_shots))
    return mean_x, mean_y


def _build_cup_library(
    records: list[dict],
) -> list[dict]:
    """
    Build a cup library from heuristic hit sessions.

    Each entry represents one unique cup position with the features needed
    to build an outer_session cup_result:
        cup_x, cup_y, winning_x, winning_y, shots, start_x, start_y,
        mean_miss_x, mean_miss_y

    For cup positions with multiple sessions, the session with the lowest
    shot count is used (most reliable hit; fewest spurious misses).

    Returns a list of dicts, one per unique (cup_x, cup_y) position.
    Cups without cup_x/cup_y labels are skipped (not groupable).
    """
    # Index shot records by session_id
    shots_by_sid: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r.get("type") == "shot":
            shots_by_sid[r["session_id"]].append(r)

    # Collect heuristic hit session_end records
    session_ends = [
        r for r in records
        if r.get("type") == "session_end"
        and r.get("policy") == "heuristic"
        and r.get("outcome") == "hit"
        and r.get("winning_x") is not None
        and r.get("cup_x") is not None
    ]

    if not session_ends:
        return []

    # Group by cup grid position
    by_cup: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for se in session_ends:
        key = (se["cup_x"], se["cup_y"])
        by_cup[key].append(se)

    cup_library = []
    for (cx, cy), sessions in sorted(by_cup.items()):
        # Pick the representative session — lowest shot count
        best = min(sessions, key=lambda s: s.get("total_shots", 999))

        sid = best["session_id"]
        start_x = best.get("start_x", 0)
        start_y = best.get("start_y", 6500)
        winning_x = best["winning_x"]
        winning_y = best["winning_y"]
        total_shots = best.get("total_shots", 1)

        mean_miss_x, mean_miss_y = _reconstruct_mean_miss(
            shots_by_sid[sid], start_x, start_y
        )

        cup_library.append({
            "cup_x":       cx,
            "cup_y":       cy,
            "winning_x":   winning_x,
            "winning_y":   winning_y,
            "shots":       total_shots,
            "start_x":     start_x,
            "start_y":     start_y,
            "mean_miss_x": mean_miss_x,
            "mean_miss_y": mean_miss_y,
            "_sessions_available": len(sessions),
        })

    return cup_library


# ---------------------------------------------------------------------------
# Synthetic episode generation
# ---------------------------------------------------------------------------


def _generate_episodes(
    cup_library: list[dict],
    n_episodes: int,
    cups_per_episode: int,
    cap_pct: float,
    rng: random.Random,
) -> list[list[dict]]:
    """
    Generate synthetic outer episodes as lists of cup_result dicts.

    Each episode is a list of cups_per_episode cup_result dicts in a random
    order. Each dict matches the cup_results schema expected by
    _build_outer_episodes() in rl/train.py.

    The per-cup appearance cap ensures no single cup position dominates.
    """
    n_cups = len(cup_library)
    if n_cups < cups_per_episode:
        raise click.ClickException(
            f"Not enough unique cup positions ({n_cups}) to form episodes of "
            f"{cups_per_episode} cups. Lower --cups-per-episode or collect more data."
        )

    cap = max(1, int(n_episodes * cap_pct))
    cup_counts = defaultdict(int)
    episodes = []

    # Build a shuffled pool of all possible (cups_per_episode)-combinations
    # and cycle through them to fill the requested episode count without
    # concentrating on any one ordering.
    all_cups = list(range(n_cups))

    attempts = 0
    max_attempts = n_episodes * 20  # safety valve

    while len(episodes) < n_episodes and attempts < max_attempts:
        attempts += 1

        # Sample cups_per_episode distinct cups
        sample_indices = rng.sample(all_cups, cups_per_episode)

        # Check cap: skip if any cup has hit its limit
        if any(cup_counts[(cup_library[i]["cup_x"], cup_library[i]["cup_y"])] >= cap
               for i in sample_indices):
            continue

        # Build cup_result records for this episode using the new unordered schema.
        # found_index = discovery order (1-based), outer_attempt_num mirrors it
        # since synthetic data has no timeouts/retries.
        cup_results = []
        for found_index, idx in enumerate(sample_indices, start=1):
            cup = cup_library[idx]
            cup_results.append({
                "found_index":       found_index,
                "outer_attempt_num": found_index,
                "session_id":        f"synthetic_{len(episodes):04d}_cup{found_index}",
                "resolved_cup_x":    cup["cup_x"],
                "resolved_cup_y":    cup["cup_y"],
                "start_x":           cup["start_x"],
                "start_y":           cup["start_y"],
                "winning_x":         cup["winning_x"],
                "winning_y":         cup["winning_y"],
                "mean_miss_x":       cup["mean_miss_x"],
                "mean_miss_y":       cup["mean_miss_y"],
                "shots":             cup["shots"],
                "outcome":           "hit",
            })

        for idx in sample_indices:
            cup_counts[(cup_library[idx]["cup_x"], cup_library[idx]["cup_y"])] += 1

        episodes.append(cup_results)

    if len(episodes) < n_episodes:
        logger.warning(
            "Could only generate %d/%d episodes within cap constraints "
            "(cap_pct=%.2f). Consider lowering --cap-pct or increasing --episodes.",
            len(episodes),
            n_episodes,
            cap_pct,
        )

    return episodes


# ---------------------------------------------------------------------------
# JSONL output
# ---------------------------------------------------------------------------


def _episodes_to_jsonl_records(episodes: list[list[dict]]) -> list[dict]:
    """
    Wrap each synthetic episode list as an outer_session_end record
    compatible with _build_outer_episodes() in rl/train.py.

    Uses the new unordered schema: cup_results are in sampled discovery
    order with found_index and resolved_cup_x/y fields.
    No outer_attempt_end records are generated for synthetic data since
    there are no real timeouts or retries to simulate.
    """
    out = []
    for i, cup_results in enumerate(episodes):
        total_shots = sum(r["shots"] for r in cup_results)
        cups_hit = sum(1 for r in cup_results if r["outcome"] == "hit")
        out.append({
            "type":             "outer_session_end",
            "outer_session_id": f"synthetic_{i:04d}",
            "inner_policy":     "heuristic",
            "outer_policy":     "fixed",
            "total_shots":      total_shots,
            "cups_hit":         cups_hit,
            "cup_results":      cup_results,
        })
    return out


# ---------------------------------------------------------------------------
# Stats reporting
# ---------------------------------------------------------------------------


def _print_stats(
    cup_library: list[dict],
    episodes: list[list[dict]],
    cups_per_episode: int,
) -> None:
    """Print dataset statistics to stdout."""
    n_cups = len(cup_library)
    n_episodes = len(episodes)

    click.echo(f"\nCup library: {n_cups} unique cup positions")
    click.echo(f"Episodes:    {n_episodes} synthetic outer sessions ({cups_per_episode} cups each)")

    # Per-cup appearance counts — use resolved_cup_x/y from new schema
    cup_counts: dict[tuple[int, int], int] = defaultdict(int)
    for ep in episodes:
        for cr in ep:
            cx = cr.get("resolved_cup_x")
            cy = cr.get("resolved_cup_y")
            if cx is not None and cy is not None:
                cup_counts[(cx, cy)] += 1

    counts = list(cup_counts.values())
    if counts:
        click.echo(
            f"Per-cup appearances: min={min(counts)}  max={max(counts)}  "
            f"mean={sum(counts)/len(counts):.1f}"
        )

    # Shot count distribution
    all_shots = [cr["shots"] for ep in episodes for cr in ep]
    if all_shots:
        click.echo(
            f"Shots per cup: min={min(all_shots)}  max={max(all_shots)}  "
            f"mean={sum(all_shots)/len(all_shots):.1f}"
        )

    # Spatial coverage — winning position ranges
    all_wx = [cr["winning_x"] for ep in episodes for cr in ep]
    all_wy = [cr["winning_y"] for ep in episodes for cr in ep]
    if all_wx:
        click.echo(
            f"Winning X range: [{min(all_wx):+d} .. {max(all_wx):+d}]"
        )
        click.echo(
            f"Winning Y range: [{min(all_wy):+d} .. {max(all_wy):+d}]"
        )

    click.echo(f"\nCup library detail:")
    click.echo(f"  {'(cx,cy)':<8}  {'win_x':>7}  {'win_y':>7}  {'shots':>5}  {'n_src':>5}")
    for cup in cup_library:
        cx, cy = cup["cup_x"], cup["cup_y"]
        appearances = cup_counts.get((cx, cy), 0)
        click.echo(
            f"  ({cx},{cy})      "
            f"{cup['winning_x']:>+7}  {cup['winning_y']:>+7}  "
            f"{cup['shots']:>5}  {cup['_sessions_available']:>5}  "
            f"(appears {appearances}x)"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--elastics", required=True, type=int,
    help="Number of elastics on the launcher (e.g. 5). "
         "Determines data directory: data/{N}_elastics/",
)
@click.option(
    "--episodes", default=500, show_default=True, type=int,
    help="Number of synthetic outer episodes to generate.",
)
@click.option(
    "--cups-per-episode", default=3, show_default=True, type=int,
    help="Number of cups per synthetic episode. Must be <= number of unique "
         "cup positions in the heuristic data. Default 3 (recommended first pass).",
)
@click.option(
    "--cap-pct", default=0.15, show_default=True, type=float,
    help="Maximum fraction of total episodes any one cup can appear in. "
         "Prevents a single cup dominating the training distribution.",
)
@click.option(
    "--seed", default=42, show_default=True, type=int,
    help="Random seed for reproducibility.",
)
@click.option(
    "--output", default=None, type=click.Path(),
    help="Output JSONL path. Defaults to data/{N}_elastics/synthetic_outer.jsonl.",
)
@click.option("--verbose", "-v", is_flag=True, default=False)
def cli(
    elastics: int,
    episodes: int,
    cups_per_episode: int,
    cap_pct: float,
    seed: int,
    output: str | None,
    verbose: bool,
) -> None:
    """
    Generate synthetic outer_session_end records for OuterGRU BC warm-start.

    Reads heuristic hit sessions from data/{N}_elastics/rl_shots.jsonl,
    reconstructs per-session features, and generates synthetic multi-cup
    outer episodes. Output is written to synthetic_outer.jsonl (or --output).

    Train the outer GRU warm-start from the output with:

        pong-train-rl outer --elastics N --mode bc \\
            --input data/{N}_elastics/synthetic_outer.jsonl
    """
    logging.basicConfig(
        format="%(levelname)-8s %(name)s: %(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
        stream=sys.stdout,
    )

    from utils.data_dir import elastics_data_dir

    data_dir = elastics_data_dir(elastics)
    input_path = data_dir / "rl_shots.jsonl"
    output_path = Path(output) if output else data_dir / "synthetic_outer.jsonl"

    if not input_path.exists():
        raise click.ClickException(
            f"Input file not found: {input_path}\n"
            f"Run pong-tune --policy heuristic first to collect heuristic sessions."
        )

    click.echo(f"Loading records from {input_path} ...")
    records = _load_jsonl(input_path)
    click.echo(f"Loaded {len(records)} records.")

    # Build cup library
    cup_library = _build_cup_library(records)
    if not cup_library:
        raise click.ClickException(
            "No heuristic hit sessions with cup_x/cup_y labels found. "
            "Only 5_elastics data with cup grid labels is currently supported."
        )

    click.echo(f"Cup library: {len(cup_library)} unique cup positions found.")

    # Generate synthetic episodes
    rng = random.Random(seed)
    synthetic_episodes = _generate_episodes(
        cup_library,
        n_episodes=episodes,
        cups_per_episode=cups_per_episode,
        cap_pct=cap_pct,
        rng=rng,
    )
    click.echo(f"Generated {len(synthetic_episodes)} synthetic episodes.")

    # Convert to outer_session_end records and write
    jsonl_records = _episodes_to_jsonl_records(synthetic_episodes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for record in jsonl_records:
            f.write(json.dumps(record) + "\n")

    click.echo(f"Written to {output_path}")

    # Stats
    _print_stats(cup_library, synthetic_episodes, cups_per_episode)

    click.echo(
        f"\nNext step: train outer GRU BC warm-start with:\n"
        f"  pong-train-rl outer --elastics {elastics} --mode bc \\\n"
        f"      --input {output_path}"
    )


if __name__ == "__main__":
    cli()
