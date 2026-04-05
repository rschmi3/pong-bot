"""
rl/plots.py - pong-plot-rl: generate RL analysis graphs from rl_shots.jsonl.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from pathlib import Path

import click

from utils.data_dir import load_jsonl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

COLOR_HEURISTIC = "tab:blue"
COLOR_GRU = "tab:orange"
COLOR_HIT = "tab:green"
COLOR_MISS = "tab:red"
COLOR_TIMEOUT = "tab:gray"
FIG_DPI = 200
FIG_SIZE = (12, 8)
FIG_SIZE_WIDE = (16, 10)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_records(data_dir: Path) -> list[dict]:
    """Load all records from rl_shots.jsonl."""
    return load_jsonl(data_dir / "rl_shots.jsonl")


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------


def _get_sessions(
    records: list[dict],
    policy: str | None = None,
    sigma_filter: bool = True,
) -> list[dict]:
    """Get standalone inner session_end records. Excludes outer-session records."""
    sessions = [r for r in records if r.get("type") == "session_end"]

    # Exclude sessions that were part of an outer session
    sessions = [s for s in sessions if not s.get("outer_session_id")]

    if policy:
        sessions = [s for s in sessions if s.get("policy") == policy]

    # For GRU sessions, prefer sigma=0.0 when available
    if sigma_filter and policy == "gru":
        sigma_zero = [s for s in sessions if s.get("sigma") == 0.0]
        if sigma_zero:
            sessions = sigma_zero

    return sessions


def _get_shots_for_session(
    records: list[dict], session_id: str,
) -> list[dict]:
    """Get all shot records for a given session, sorted by shot number."""
    shots = [
        r for r in records
        if r.get("type") == "shot" and r.get("session_id") == session_id
    ]
    return sorted(shots, key=lambda s: s.get("shot", 0))


def _find_paired_episodes(
    heuristic_sessions: list[dict],
    gru_sessions: list[dict],
) -> list[tuple[dict, dict]]:
    """
    Pair heuristic and GRU episodes by exact (cup_x, cup_y) match.

    Only pairs sessions where both have cup_x and cup_y set.
    Returns list of (heuristic_session, gru_session) tuples.
    """
    # Group by cup position
    h_by_cup: dict[tuple, list[dict]] = defaultdict(list)
    g_by_cup: dict[tuple, list[dict]] = defaultdict(list)

    for s in heuristic_sessions:
        cx, cy = s.get("cup_x"), s.get("cup_y")
        if cx is not None and cy is not None:
            h_by_cup[(cx, cy)].append(s)

    for s in gru_sessions:
        cx, cy = s.get("cup_x"), s.get("cup_y")
        if cx is not None and cy is not None:
            g_by_cup[(cx, cy)].append(s)

    pairs = []
    for cup_pos in sorted(set(h_by_cup) & set(g_by_cup)):
        h_list = h_by_cup[cup_pos]
        g_list = g_by_cup[cup_pos]
        # Pick median-shots episode from each (representative, not cherry-picked)
        h_sorted = sorted(h_list, key=lambda s: s.get("total_shots", 999))
        g_sorted = sorted(g_list, key=lambda s: s.get("total_shots", 999))
        h_pick = h_sorted[len(h_sorted) // 2]
        g_pick = g_sorted[len(g_sorted) // 2]
        pairs.append((h_pick, g_pick))

    return pairs


def _distance(x: int, y: int, wx: int, wy: int) -> float:
    """Euclidean distance in step space."""
    return math.sqrt((x - wx) ** 2 + (y - wy) ** 2)



# ---------------------------------------------------------------------------
# Graph implementations
# ---------------------------------------------------------------------------


def _plot_trajectory(records: list[dict], plots_dir: Path) -> None:
    """
    Episode trajectory plot: X vs Y scatter with connected arrows.

    Generates two types of plots:
    1. Paired comparison (heuristic vs GRU for same cup position)
    2. Overlay (all episodes for a cup position, coloured by policy)
    """
    import matplotlib.pyplot as plt

    h_sessions = _get_sessions(records, "heuristic")
    g_sessions = _get_sessions(records, "gru")

    # Only include hit sessions (we need winning_x/y for the target marker)
    h_hits = [s for s in h_sessions if s.get("outcome") == "hit"]
    g_hits = [s for s in g_sessions if s.get("outcome") == "hit"]

    pairs = _find_paired_episodes(h_hits, g_hits)

    if not pairs:
        # No paired data - plot individual trajectories instead
        all_sessions = h_hits + g_hits
        if not all_sessions:
            logger.warning("No hit sessions found - skipping trajectory plot")
            return

        # Plot up to 6 individual trajectories
        n = min(6, len(all_sessions))
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=FIG_SIZE_WIDE, squeeze=False)

        for idx, session in enumerate(all_sessions[:n]):
            ax = axes[idx // cols][idx % cols]
            _draw_trajectory(ax, records, session)
            ax.invert_yaxis()
            policy = session.get("policy", "?")
            shots = session.get("total_shots", "?")
            ax.set_title(f"{policy} - {shots} shots", fontsize=11)

        # Hide empty subplots
        for idx in range(n, rows * cols):
            axes[idx // cols][idx % cols].set_visible(False)

        fig.suptitle("Episode Trajectories", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(plots_dir / "trajectory_individual.png", dpi=FIG_DPI)
        plt.close(fig)
        click.echo(f"  trajectory_individual.png ({n} episodes)")
        return

    # --- Paired comparison plots ---
    for h_sess, g_sess in pairs:
        cx = h_sess.get("cup_x", "?")
        cy = h_sess.get("cup_y", "?")

        fig, (ax_h, ax_g) = plt.subplots(1, 2, figsize=FIG_SIZE_WIDE)

        _draw_trajectory(ax_h, records, h_sess, color=COLOR_HEURISTIC)
        ax_h.invert_yaxis()
        ax_h.set_title(
            f"Heuristic - {h_sess['total_shots']} shots",
            fontsize=12, color=COLOR_HEURISTIC,
        )

        _draw_trajectory(ax_g, records, g_sess, color=COLOR_GRU)
        ax_g.invert_yaxis()
        ax_g.set_title(
            f"GRU - {g_sess['total_shots']} shots",
            fontsize=12, color=COLOR_GRU,
        )

        fig.suptitle(
            f"Trajectory Comparison - Cup ({cx},{cy})",
            fontsize=14, fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(plots_dir / f"trajectory_pair_cup{cx}_{cy}.png", dpi=FIG_DPI)
        plt.close(fig)

    click.echo(f"  trajectory_pair_*.png ({len(pairs)} pairs)")

    # --- Overlay plots per cup position ---
    cup_sessions: dict[tuple, list[dict]] = defaultdict(list)
    for s in h_hits + g_hits:
        cx, cy = s.get("cup_x"), s.get("cup_y")
        if cx is not None and cy is not None:
            cup_sessions[(cx, cy)].append(s)

    for cup_pos, sessions in sorted(cup_sessions.items()):
        if len(sessions) < 2:
            continue
        cx, cy = cup_pos
        fig, ax = plt.subplots(figsize=FIG_SIZE)

        for s in sessions:
            policy = s.get("policy", "?")
            color = COLOR_HEURISTIC if policy == "heuristic" else COLOR_GRU
            _draw_trajectory(ax, records, s, color=color, alpha=0.5, label_prefix=policy)

        ax.invert_yaxis()

        # Deduplicate legend
        handles, labels = ax.get_legend_handles_labels()
        seen: set[str] = set()
        unique = []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l)
                unique.append((h, l))
        if unique:
            ax.legend(*zip(*unique), fontsize=10)

        ax.set_title(
            f"All Episodes - Cup ({cx},{cy})",
            fontsize=14, fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(plots_dir / f"trajectory_overlay_cup{cx}_{cy}.png", dpi=FIG_DPI)
        plt.close(fig)

    click.echo(f"  trajectory_overlay_*.png ({len(cup_sessions)} positions)")


def _draw_trajectory(
    ax, records: list[dict], session: dict,
    color: str = "tab:blue", alpha: float = 1.0, label_prefix: str = "",
) -> None:
    """Draw a single episode trajectory on an axes."""
    shots = _get_shots_for_session(records, session["session_id"])
    if not shots:
        return

    xs = [s["x_steps"] for s in shots]
    ys = [s["y_steps"] for s in shots]
    hits = [s.get("hit", False) for s in shots]

    # Draw connected path with arrows
    label = label_prefix if label_prefix else None
    ax.plot(xs, ys, "-", color=color, alpha=alpha * 0.6, linewidth=1.5, label=label)

    # Draw shots: misses as circles, hit as star
    for i, (x, y, hit) in enumerate(zip(xs, ys, hits)):
        if hit:
            ax.plot(x, y, "*", color=COLOR_HIT, markersize=15, zorder=5)
        else:
            ax.plot(x, y, "o", color=COLOR_MISS, markersize=5, alpha=alpha * 0.8, zorder=4)
        # Number each shot
        ax.annotate(
            str(i + 1), (x, y), textcoords="offset points",
            xytext=(5, 5), fontsize=7, alpha=alpha * 0.7,
        )

    # Draw cup target position
    wx = session.get("winning_x")
    wy = session.get("winning_y")
    if wx is not None and wy is not None:
        ax.plot(wx, wy, "x", color="black", markersize=12, markeredgewidth=2, zorder=6)

    ax.set_xlabel("X steps", fontsize=11)
    ax.set_ylabel("Y steps", fontsize=11)
    ax.grid(True, alpha=0.3)


def _plot_histogram(records: list[dict], plots_dir: Path) -> None:
    """Shots-to-hit histogram grouped by policy."""
    import matplotlib.pyplot as plt

    h_sessions = _get_sessions(records, "heuristic")
    g_sessions = _get_sessions(records, "gru")

    h_hits = [s["total_shots"] for s in h_sessions if s.get("outcome") == "hit"]
    g_hits = [s["total_shots"] for s in g_sessions if s.get("outcome") == "hit"]
    h_timeouts = sum(1 for s in h_sessions if s.get("outcome") == "timeout")
    g_timeouts = sum(1 for s in g_sessions if s.get("outcome") == "timeout")

    if not h_hits and not g_hits:
        logger.warning("No hit sessions - skipping histogram")
        return

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    max_shots = max((h_hits + g_hits), default=20)
    bins = range(1, max_shots + 2)

    if h_hits:
        ax.hist(
            h_hits, bins=bins, alpha=0.6, color=COLOR_HEURISTIC,
            label=f"Heuristic (n={len(h_hits)}, mean={sum(h_hits)/len(h_hits):.1f})",
            edgecolor="white",
        )
        ax.axvline(
            sum(h_hits) / len(h_hits), color=COLOR_HEURISTIC,
            linestyle="--", linewidth=2, alpha=0.8,
        )

    if g_hits:
        ax.hist(
            g_hits, bins=bins, alpha=0.6, color=COLOR_GRU,
            label=f"GRU (n={len(g_hits)}, mean={sum(g_hits)/len(g_hits):.1f})",
            edgecolor="white",
        )
        ax.axvline(
            sum(g_hits) / len(g_hits), color=COLOR_GRU,
            linestyle="--", linewidth=2, alpha=0.8,
        )

    # Annotate timeouts
    timeout_text = []
    if h_timeouts:
        timeout_text.append(f"Heuristic timeouts: {h_timeouts}")
    if g_timeouts:
        timeout_text.append(f"GRU timeouts: {g_timeouts}")
    if timeout_text:
        ax.text(
            0.98, 0.95, "\n".join(timeout_text),
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, color=COLOR_TIMEOUT,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    ax.set_xlabel("Shots to Hit", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Shots-to-Hit Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(plots_dir / "shots_histogram.png", dpi=FIG_DPI)
    plt.close(fig)
    click.echo("  shots_histogram.png")


def _plot_convergence(records: list[dict], plots_dir: Path) -> None:
    """Convergence speed: distance to cup vs shot number, averaged per policy."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for policy, color, label in [
        ("heuristic", COLOR_HEURISTIC, "Heuristic"),
        ("gru", COLOR_GRU, "GRU"),
    ]:
        sessions = _get_sessions(records, policy)
        hits = [s for s in sessions if s.get("outcome") == "hit"]

        if not hits:
            continue

        # Collect distance-to-cup per shot number across all hit episodes
        max_shot = max(s.get("total_shots", 0) for s in hits)
        distances_by_shot: dict[int, list[float]] = defaultdict(list)

        for s in hits:
            wx, wy = s["winning_x"], s["winning_y"]
            shots = _get_shots_for_session(records, s["session_id"])
            for shot in shots:
                n = shot["shot"]
                d = _distance(shot["x_steps"], shot["y_steps"], wx, wy)
                distances_by_shot[n].append(d)

        # Compute mean and std per shot number
        shot_nums = sorted(distances_by_shot.keys())
        means = [np.mean(distances_by_shot[n]) for n in shot_nums]
        stds = [np.std(distances_by_shot[n]) for n in shot_nums]

        means_arr = np.array(means)
        stds_arr = np.array(stds)

        ax.plot(shot_nums, means, "-o", color=color, label=label, markersize=4)
        ax.fill_between(
            shot_nums,
            means_arr - stds_arr,
            means_arr + stds_arr,
            color=color, alpha=0.15,
        )

    ax.set_xlabel("Shot Number", fontsize=12)
    ax.set_ylabel("Distance to Cup (steps)", fontsize=12)
    ax.set_title("Convergence Speed", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "convergence_speed.png", dpi=FIG_DPI)
    plt.close(fig)
    click.echo("  convergence_speed.png")


def _plot_heatmap(records: list[dict], plots_dir: Path) -> None:
    """Shots-to-hit by cup position (winning_x, winning_y)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE_WIDE)

    for ax, policy, color, title in [
        (axes[0], "heuristic", COLOR_HEURISTIC, "Heuristic"),
        (axes[1], "gru", COLOR_GRU, "GRU"),
    ]:
        sessions = _get_sessions(records, policy)
        hits = [s for s in sessions if s.get("outcome") == "hit"]
        timeouts = [s for s in sessions if s.get("outcome") == "timeout"]

        if hits:
            wxs = [s["winning_x"] for s in hits]
            wys = [s["winning_y"] for s in hits]
            shots = [s["total_shots"] for s in hits]

            sc = ax.scatter(
                wxs, wys, c=shots, cmap="RdYlGn_r",
                s=80, edgecolors="black", linewidth=0.5,
                vmin=1, vmax=max(shots),
            )
            plt.colorbar(sc, ax=ax, label="Shots to Hit")

        if timeouts:
            # Use winning_x/y if available (some timeouts have them), else skip
            tx = [s.get("winning_x", s.get("start_x", 0)) for s in timeouts]
            ty = [s.get("winning_y", s.get("start_y", 0)) for s in timeouts]
            ax.scatter(tx, ty, c=COLOR_TIMEOUT, s=40, marker="x", label="Timeout")

        ax.set_xlabel("X steps", fontsize=11)
        ax.set_ylabel("Y steps", fontsize=11)
        ax.invert_yaxis()
        ax.set_title(f"{title} (n={len(hits)} hits, {len(timeouts)} timeouts)", fontsize=12)
        ax.grid(True, alpha=0.3)
        if timeouts:
            ax.legend(fontsize=10)

    # Shared axis limits for fair visual comparison
    all_hits = [s for p in ["heuristic", "gru"]
                for s in _get_sessions(records, p) if s.get("outcome") == "hit"]
    if all_hits:
        all_x = [s["winning_x"] for s in all_hits]
        all_y = [s["winning_y"] for s in all_hits]
        pad_x = (max(all_x) - min(all_x)) * 0.1
        pad_y = (max(all_y) - min(all_y)) * 0.1
        for ax in axes:
            ax.set_xlim(min(all_x) - pad_x, max(all_x) + pad_x)
            ax.set_ylim(max(all_y) + pad_y, min(all_y) - pad_y)  # inverted

    fig.suptitle("Shots-to-Hit by Cup Position", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / "shots_heatmap.png", dpi=FIG_DPI)
    plt.close(fig)
    click.echo("  shots_heatmap.png")


def _plot_hit_rate(records: list[dict], plots_dir: Path) -> None:
    """Cumulative hit rate over time."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for policy, color, label in [
        ("heuristic", COLOR_HEURISTIC, "Heuristic"),
        ("gru", COLOR_GRU, "GRU"),
    ]:
        sessions = _get_sessions(records, policy)
        if not sessions:
            continue

        # Sort chronologically
        sessions = sorted(sessions, key=lambda s: s.get("session_id", ""))

        cum_hits = 0
        cum_total = 0
        x_vals = []
        y_vals = []
        for i, s in enumerate(sessions):
            cum_total += 1
            if s.get("outcome") == "hit":
                cum_hits += 1
            x_vals.append(i + 1)
            y_vals.append(100.0 * cum_hits / cum_total)

        ax.plot(x_vals, y_vals, "-", color=color, label=label, linewidth=2)

    ax.set_xlabel("Session Number", fontsize=12)
    ax.set_ylabel("Cumulative Hit Rate (%)", fontsize=12)
    ax.set_title("Hit Rate Over Time", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "hit_rate_over_time.png", dpi=FIG_DPI)
    plt.close(fig)
    click.echo("  hit_rate_over_time.png")


def _plot_direction(records: list[dict], plots_dir: Path) -> None:
    """Directional accuracy bar chart grouped by policy."""
    import matplotlib.pyplot as plt
    import numpy as np

    dir_checks = {
        "left":  (lambda dx, dy: dx > 0),
        "right": (lambda dx, dy: dx < 0),
        "short": (lambda dx, dy: dy < 0),
        "long":  (lambda dx, dy: dy > 0),
    }

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    policies = ["heuristic", "gru"]
    colors = [COLOR_HEURISTIC, COLOR_GRU]
    bar_width = 0.35
    directions = ["left", "right", "short", "long"]
    x_pos = np.arange(len(directions))

    for i, (policy, color) in enumerate(zip(policies, colors)):
        sessions = _get_sessions(records, policy)
        if not sessions:
            continue

        stats: dict[str, dict] = {d: {"correct": 0, "total": 0} for d in directions}

        for s in sessions:
            shots = _get_shots_for_session(records, s["session_id"])
            for j in range(len(shots) - 1):
                cur = shots[j]
                nxt = shots[j + 1]
                d = cur.get("direction")
                if d not in dir_checks:
                    continue
                dx = nxt["x_steps"] - cur["x_steps"]
                dy = nxt["y_steps"] - cur["y_steps"]
                stats[d]["total"] += 1
                if dir_checks[d](dx, dy):
                    stats[d]["correct"] += 1

        accuracies = []
        for d in directions:
            t = stats[d]["total"]
            accuracies.append(100.0 * stats[d]["correct"] / t if t > 0 else 0)

        bars = ax.bar(
            x_pos + i * bar_width, accuracies, bar_width,
            color=color, label=policy.capitalize(), edgecolor="white",
        )
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            if acc > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{acc:.0f}%", ha="center", va="bottom", fontsize=9,
                )

    ax.set_xlabel("Direction", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Directional Accuracy by Policy", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos + bar_width / 2)
    ax.set_xticklabels([d.capitalize() for d in directions], fontsize=11)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(plots_dir / "directional_accuracy.png", dpi=FIG_DPI)
    plt.close(fig)
    click.echo("  directional_accuracy.png")


# ---------------------------------------------------------------------------
# Outer session graphs
# ---------------------------------------------------------------------------


def _plot_outer_trajectory(records: list[dict], plots_dir: Path) -> None:
    """
    Multi-cup outer session trajectory.

    Shows all cup searches in sequence with different colours per cup,
    lines connecting the end of one search to the start of the next.
    """
    import matplotlib.pyplot as plt

    outer_ends = [r for r in records if r.get("type") == "outer_session_end"]
    if not outer_ends:
        logger.warning("No outer session data - skipping outer-trajectory")
        return

    # Cup colour cycle
    cup_colors = [
        "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
        "tab:brown", "tab:pink", "tab:olive", "tab:cyan",
    ]

    for outer in outer_ends:
        oid = outer["outer_session_id"]
        cup_results = outer.get("cup_results", [])
        if not cup_results:
            continue

        fig, ax = plt.subplots(figsize=FIG_SIZE)

        prev_end = None  # last shot position of previous cup

        for i, cr in enumerate(cup_results):
            color = cup_colors[i % len(cup_colors)]
            sid = cr.get("session_id")
            # New schema: resolved_cup_x/y; found_index for display
            cx = cr.get("resolved_cup_x", "?")
            cy = cr.get("resolved_cup_y", "?")
            fi = cr.get("found_index", i + 1)
            outcome = cr.get("outcome", "?")
            n_shots = cr.get("shots", 0)

            # Get shots for this inner session
            shots = _get_shots_for_session(records, sid) if sid else []

            if not shots:
                continue

            xs = [s["x_steps"] for s in shots]
            ys = [s["y_steps"] for s in shots]

            # Draw connection from previous cup's last shot
            if prev_end is not None:
                ax.plot(
                    [prev_end[0], xs[0]], [prev_end[1], ys[0]],
                    "--", color="gray", linewidth=1, alpha=0.5,
                )

            # Draw trajectory
            ax.plot(
                xs, ys, "-o", color=color, markersize=4, linewidth=1.5,
                alpha=0.8,
                label=f"Cup #{fi} ({cx},{cy}) - {n_shots} shots, {outcome}",
            )

            # Mark hit with a star
            if outcome == "hit":
                ax.plot(xs[-1], ys[-1], "*", color=color, markersize=15, zorder=5)

            # Mark cup target
            wx = cr.get("winning_x")
            wy = cr.get("winning_y")
            if wx is not None and wy is not None:
                ax.plot(wx, wy, "x", color=color, markersize=10, markeredgewidth=2, zorder=6)

            prev_end = (xs[-1], ys[-1])

        ax.set_xlabel("X steps", fontsize=11)
        ax.set_ylabel("Y steps", fontsize=11)
        ax.invert_yaxis()
        ax.set_title(
            f"Outer Session {oid} - {len(cup_results)} cups",
            fontsize=14, fontweight="bold",
        )
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / f"outer_trajectory_{oid}.png", dpi=FIG_DPI)
        plt.close(fig)

    click.echo(f"  outer_trajectory_*.png ({len(outer_ends)} sessions)")


def _plot_outer_total_shots(records: list[dict], plots_dir: Path) -> None:
    """Total shots per outer session configuration bar chart."""
    import matplotlib.pyplot as plt
    import numpy as np

    outer_ends = [r for r in records if r.get("type") == "outer_session_end"]
    if not outer_ends:
        logger.warning("No outer session data - skipping outer-total-shots")
        return

    # Group by configuration using new outer_policy field
    configs: dict[str, list[int]] = defaultdict(list)
    for outer in outer_ends:
        inner_policy = outer.get("inner_policy", "?")
        outer_pol    = outer.get("outer_policy", "fixed")

        if inner_policy == "heuristic":
            config_name = "Heuristic + Fixed Start"
        elif inner_policy == "gru" and outer_pol == "fixed":
            config_name = "GRU + Fixed Start"
        elif inner_policy == "gru" and outer_pol == "gru":
            config_name = "GRU + Outer GRU"
        else:
            config_name = f"{inner_policy} + {outer_pol}"

        configs[config_name].append(outer.get("total_shots", 0))

    if not configs:
        return

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    config_names = list(configs.keys())
    config_colors = [COLOR_HEURISTIC, COLOR_GRU, "tab:green"]
    x_pos = np.arange(len(config_names))

    means = [np.mean(configs[c]) for c in config_names]
    stds = [np.std(configs[c]) for c in config_names]
    counts = [len(configs[c]) for c in config_names]

    bars = ax.bar(
        x_pos, means, yerr=stds,
        color=[config_colors[i % len(config_colors)] for i in range(len(config_names))],
        edgecolor="white", capsize=5,
    )

    # Add labels
    for bar, mean, count in zip(bars, means, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{mean:.1f} (n={count})", ha="center", va="bottom", fontsize=10,
        )

    ax.set_xlabel("Configuration", fontsize=12)
    ax.set_ylabel("Total Shots (all cups)", fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(config_names, fontsize=11)
    ax.set_title(
        "Outer Session: Total Shots Comparison",
        fontsize=14, fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(plots_dir / "outer_total_shots.png", dpi=FIG_DPI)
    plt.close(fig)
    click.echo("  outer_total_shots.png")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _print_summary(records: list[dict]) -> None:
    """Print a summary table to stdout."""
    click.echo("\n" + "=" * 65)
    click.echo(f"{'Policy':<20} {'Sessions':>10} {'Hit Rate':>10} {'Mean Shots':>12} {'Median':>8}")
    click.echo("-" * 65)

    for policy in ("heuristic", "gru"):
        sessions = _get_sessions(records, policy)
        if not sessions:
            continue
        hits = [s for s in sessions if s.get("outcome") == "hit"]
        hit_shots = sorted(s["total_shots"] for s in hits)
        n = len(sessions)
        hit_rate = 100.0 * len(hits) / n if n else 0
        mean = sum(hit_shots) / len(hit_shots) if hit_shots else float("nan")
        median = hit_shots[len(hit_shots) // 2] if hit_shots else float("nan")

        # Show sigma info for GRU
        label = policy.capitalize()
        if policy == "gru":
            sigma_zero = [s for s in sessions if s.get("sigma") == 0.0]
            all_gru = _get_sessions(records, "gru", sigma_filter=False)
            if sigma_zero:
                label = f"GRU (σ=0, n={len(sigma_zero)})"
            elif len(all_gru) > 0:
                label = f"GRU (all, n={len(all_gru)})"

        click.echo(
            f"{label:<20} {n:>10} {hit_rate:>9.1f}% {mean:>12.1f} {median:>8}"
        )

    click.echo("=" * 65)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.group("pong-plot-rl")
@click.option("--elastics", required=True, type=int,
              help="Number of elastics - determines data directory.")
@click.pass_context
def cli(ctx: click.Context, elastics: int) -> None:
    """Generate RL analysis graphs from session data."""
    from utils.data_dir import elastics_data_dir
    data_dir = elastics_data_dir(elastics)
    if not data_dir.exists():
        raise click.ClickException(f"Data directory not found: {data_dir}")
    ctx.ensure_object(dict)
    ctx.obj["data_dir"] = data_dir
    ctx.obj["plots_dir"] = data_dir / "plots"


def _setup_rl(ctx: click.Context) -> tuple[list[dict], Path]:
    """Load RL records and ensure plots directory exists."""
    data_dir = ctx.obj["data_dir"]
    plots_dir = ctx.obj["plots_dir"]
    plots_dir.mkdir(parents=True, exist_ok=True)
    records = _load_records(data_dir)
    click.echo(f"Loaded {len(records)} RL records from {data_dir / 'rl_shots.jsonl'}")
    return records, plots_dir


# --- RL graphs ---

@cli.command("trajectory")
@click.pass_context
def trajectory_cmd(ctx: click.Context) -> None:
    """Episode trajectory comparison (X vs Y scatter)."""
    records, plots_dir = _setup_rl(ctx)
    _plot_trajectory(records, plots_dir)


@cli.command("histogram")
@click.pass_context
def histogram_cmd(ctx: click.Context) -> None:
    """Shots-to-hit distribution by policy."""
    records, plots_dir = _setup_rl(ctx)
    _plot_histogram(records, plots_dir)


@cli.command("convergence")
@click.pass_context
def convergence_cmd(ctx: click.Context) -> None:
    """Convergence speed: distance to cup vs shot number."""
    records, plots_dir = _setup_rl(ctx)
    _plot_convergence(records, plots_dir)


@cli.command("heatmap")
@click.pass_context
def heatmap_cmd(ctx: click.Context) -> None:
    """Shots-to-hit by cup position on the table."""
    records, plots_dir = _setup_rl(ctx)
    _plot_heatmap(records, plots_dir)


@cli.command("hit-rate")
@click.pass_context
def hit_rate_cmd(ctx: click.Context) -> None:
    """Cumulative hit rate over time."""
    records, plots_dir = _setup_rl(ctx)
    _plot_hit_rate(records, plots_dir)


@cli.command("direction")
@click.pass_context
def direction_cmd(ctx: click.Context) -> None:
    """Directional accuracy bar chart by policy."""
    records, plots_dir = _setup_rl(ctx)
    _plot_direction(records, plots_dir)


# --- Outer session graphs ---

@cli.command("outer-trajectory")
@click.pass_context
def outer_trajectory_cmd(ctx: click.Context) -> None:
    """Multi-cup outer session trajectory."""
    records, plots_dir = _setup_rl(ctx)
    _plot_outer_trajectory(records, plots_dir)


@cli.command("outer-total-shots")
@click.pass_context
def outer_total_shots_cmd(ctx: click.Context) -> None:
    """Outer session total shots comparison bar chart."""
    records, plots_dir = _setup_rl(ctx)
    _plot_outer_total_shots(records, plots_dir)


# --- Generate all ---

@cli.command("all")
@click.pass_context
def all_cmd(ctx: click.Context) -> None:
    """Generate all RL graphs and print summary table."""
    data_dir = ctx.obj["data_dir"]
    plots_dir = ctx.obj["plots_dir"]
    plots_dir.mkdir(parents=True, exist_ok=True)

    records = _load_records(data_dir)
    if records:
        click.echo(f"\nLoaded {len(records)} RL records")
        click.echo("Generating RL graphs...")
        _plot_trajectory(records, plots_dir)
        _plot_histogram(records, plots_dir)
        _plot_convergence(records, plots_dir)
        _plot_heatmap(records, plots_dir)
        _plot_hit_rate(records, plots_dir)
        _plot_direction(records, plots_dir)
        _plot_outer_trajectory(records, plots_dir)
        _plot_outer_total_shots(records, plots_dir)
        _print_summary(records)
    else:
        click.echo("\nNo RL data found - skipping RL graphs.")

    click.echo(f"\nAll graphs saved to {plots_dir}/")


if __name__ == "__main__":
    cli()
