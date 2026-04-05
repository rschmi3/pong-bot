"""
rl/train.py - pong-train-rl: offline GRU policy training.

Subcommands: bc (behaviour cloning), rl (REINFORCE fine-tuning), outer (outer GRU training).
Reads from data/{N}_elastics/rl_shots.jsonl.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

logger = logging.getLogger(__name__)

NORMALISE_STEPS = 50_000  # must match rl/policy.py


# ---------------------------------------------------------------------------
# Shared data loading helpers
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict]:
    """Load all records from a JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning("Skipping malformed line: %s", e)
    return records


def _build_episodes(
    records: list[dict],
    policy_filter: str | None,
) -> list[list[dict]]:
    """Group shot records into complete inner episodes. Skips interrupted sessions."""
    session_ends: dict[str, dict] = {}
    for r in records:
        if r.get("type") == "session_end":
            session_ends[r["session_id"]] = r

    shots_by_session: dict[str, list[dict]] = {}
    for r in records:
        if r.get("type") != "shot":
            continue
        sid = r.get("session_id")
        if sid is None:
            continue
        if policy_filter and r.get("policy") != policy_filter:
            continue
        shots_by_session.setdefault(sid, []).append(r)

    episodes = []
    skipped = 0
    for sid, shots in shots_by_session.items():
        if sid not in session_ends:
            skipped += 1
            continue
        shots_sorted = sorted(shots, key=lambda r: r["shot"])
        episodes.append(shots_sorted)

    if skipped:
        logger.info("Skipped %d interrupted sessions (no session_end record)", skipped)

    return episodes


def _build_state_vec(
    x: int,
    y: int,
    last_dir: str | None,
    last_hit: bool,
    shot_num: int,
    max_shots: int,
) -> list[float]:
    """Build the 8-dim state vector for one shot. Must match GRUPolicy._build_state."""
    dir_vec = [0.0, 0.0, 0.0, 0.0]  # left, right, short, long
    if last_dir == "left":
        dir_vec[0] = 1.0
    elif last_dir == "right":
        dir_vec[1] = 1.0
    elif last_dir == "short":
        dir_vec[2] = 1.0
    elif last_dir == "long":
        dir_vec[3] = 1.0
    return [
        x / NORMALISE_STEPS,
        y / NORMALISE_STEPS,
        dir_vec[0],
        dir_vec[1],
        dir_vec[2],
        dir_vec[3],
        float(last_hit),
        shot_num / max_shots,
    ]


def _episodes_to_sequences(
    episodes: list[list[dict]],
    max_shots: int,
) -> list[dict]:
    """
    Convert raw episode shot records into training sequences.

    Each returned dict has:
        states   : list of T 8-dim state vectors
        actions  : list of T 2-dim normalised delta vectors (BC target)
        rewards  : list of T float rewards
        hit      : bool - did this episode end in a hit?
    """
    sequences = []
    for shots in episodes:
        if len(shots) < 2:
            continue  # need at least 2 shots to have one delta

        states = []
        actions = []
        rewards = []

        for i, shot in enumerate(shots):
            x = shot["x_steps"]
            y = shot["y_steps"]

            # Use THIS shot's direction/hit in the state. Although inference
            # sees the previous shot's direction in _last_dir, the GRU's
            # hidden state accumulates the full sequence - pairing the
            # current direction with its corrective action teaches a direct
            # reactive mapping that the hidden state bridges at inference.
            cur_dir = shot.get("direction")
            cur_hit = shot.get("hit", False)

            state = _build_state_vec(x, y, cur_dir, cur_hit, i, max_shots)
            states.append(state)

            # BC action target: delta to next position (zero for last shot)
            if i + 1 < len(shots):
                nx = shots[i + 1]["x_steps"]
                ny = shots[i + 1]["y_steps"]
                action = [
                    (nx - x) / NORMALISE_STEPS,
                    (ny - y) / NORMALISE_STEPS,
                ]
            else:
                action = [0.0, 0.0]
            actions.append(action)

            reward = 1.0 if cur_hit else -0.05
            rewards.append(reward)

        hit = any(s["hit"] for s in shots)
        sequences.append({
            "states":  states,
            "actions": actions,
            "rewards": rewards,
            "hit":     hit,
        })

    return sequences


# ---------------------------------------------------------------------------
# Behaviour Cloning
# ---------------------------------------------------------------------------


def _augment_flip_x(sequences: list[dict]) -> list[dict]:
    """
    Double the dataset by reflecting the X axis.

    For each sequence, creates a mirrored version where:
      - state[0] (x_norm) is negated
      - state[2] (left flag) and state[3] (right flag) are swapped
      - action[0] (dx_norm) is negated
      - Y, short/long flags, last_hit, shot_num are unchanged

    This exploits left-right symmetry: a cup to the right at +X is
    physically equivalent to a cup to the left at -X.
    """
    augmented = []
    for seq in sequences:
        flipped_states = []
        flipped_actions = []
        for state, action in zip(seq["states"], seq["actions"]):
            s = list(state)
            s[0] = -s[0]              # negate x_norm
            s[2], s[3] = s[3], s[2]  # swap left <-> right flags
            a = list(action)
            a[0] = -a[0]              # negate dx_norm
            flipped_states.append(s)
            flipped_actions.append(a)
        augmented.append({
            "states":  flipped_states,
            "actions": flipped_actions,
            "rewards": list(seq["rewards"]),
            "hit":     seq["hit"],
        })
    return augmented


def _eval_directional_accuracy(pol, sequences: list[dict]) -> None:
    """
    Compute and print directional accuracy on training sequences.

    For each timestep where a direction hint is present (state[2-5] != 0),
    checks whether the GRU's predicted delta (no noise) has the correct sign.
    Printed as a diagnostic after BC training.
    """
    import torch
    from collections import defaultdict

    # State indices: [2]=left, [3]=right, [4]=short, [5]=long
    # Correct sign for each: left→+dx, right→-dx, short→-dy, long→+dy
    dir_map = {
        2: ("left",  0, +1),   # (name, axis: 0=x/1=y, correct_sign)
        3: ("right", 0, -1),
        4: ("short", 1, -1),
        5: ("long",  1, +1),
    }
    stats: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})

    pol.gru.eval()
    pol.head.eval()
    with torch.no_grad():
        for seq in sequences:
            states_t = torch.tensor(seq["states"], dtype=torch.float32)
            inp = states_t.unsqueeze(0)
            out, _ = pol.gru(inp)
            pred = pol.head(out[0])  # (T, 2)

            for t in range(len(seq["states"])):
                for state_idx, (dir_name, axis, correct_sign) in dir_map.items():
                    if seq["states"][t][state_idx] > 0.5:
                        predicted_val = pred[t, axis].item()
                        correct = (predicted_val * correct_sign) > 0
                        stats[dir_name]["correct"] += int(correct)
                        stats[dir_name]["total"] += 1

    click.echo("\nDirectional accuracy on training data (no noise):")
    for d in ("left", "right", "short", "long"):
        s = stats[d]
        if s["total"] > 0:
            acc = s["correct"] / s["total"]
            click.echo(f"  {d:6s}: {acc:.0%}  ({s['correct']}/{s['total']})")
        else:
            click.echo(f"  {d:6s}: no samples")


def _run_bc(
    input_path: Path,
    checkpoint_path: Path,
    epochs: int,
    lr: float,
    max_shots: int,
    dir_loss_weight: float = 2.0,
) -> None:
    import torch
    import torch.nn as nn
    from rl.policy import GRUPolicy

    records = _load_jsonl(input_path)
    episodes = _build_episodes(records, policy_filter="heuristic")
    if not episodes:
        raise click.ClickException(
            "No complete heuristic sessions found in the input file. "
            "Run pong-tune --policy heuristic first to collect training data."
        )

    sequences = _episodes_to_sequences(episodes, max_shots)
    augmented = _augment_flip_x(sequences)
    all_sequences = sequences + augmented
    click.echo(
        f"BC: {len(sequences)} episodes + {len(augmented)} X-flipped = "
        f"{len(all_sequences)} total, "
        f"{sum(len(s['states']) for s in all_sequences)} timesteps"
    )

    # Load or initialise policy
    pol = GRUPolicy(max_shots=max_shots)
    if checkpoint_path.exists():
        pol.load(str(checkpoint_path))
        click.echo(f"Loaded existing checkpoint from {checkpoint_path}")
    else:
        click.echo("No existing checkpoint - initialising new GRU weights")

    optimiser = torch.optim.Adam(
        list(pol.gru.parameters()) + list(pol.head.parameters()), lr=lr
    )
    loss_fn = nn.L1Loss()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_steps = 0

        for seq in all_sequences:
            states_t  = torch.tensor(seq["states"],  dtype=torch.float32)  # (T, 8)
            actions_t = torch.tensor(seq["actions"], dtype=torch.float32)  # (T, 2)

            # Full sequence forward pass
            inp = states_t.unsqueeze(0)          # (1, T, 8)
            out, _ = pol.gru(inp)                 # (1, T, hidden)
            pred = pol.head(out[0])               # (T, 2)

            loss = loss_fn(pred, actions_t)

            # Directional sign penalty - penalise predicted deltas with the
            # wrong sign given the direction hint in the state.
            # State indices: [2]=left, [3]=right, [4]=short, [5]=long
            # Correct directions: left→+dx, right→-dx, short→-dy, long→+dy
            # relu(wrong_sign_output) > 0 only when sign is wrong.
            # When last_dir=None (first shot), all flags are 0 → no penalty.
            # Mask out the last timestep per sequence - its action target is
            # [0, 0] by construction (no next position), so applying the sign
            # penalty there fights the L1 loss and corrupts training.
            if dir_loss_weight > 0.0:
                T = states_t.shape[0]
                # mask: 1 for all timesteps except the last
                mask = torch.ones(T, dtype=torch.float32)
                mask[-1] = 0.0

                dir_penalty = (
                    mask * (
                        states_t[:, 3] * torch.relu( pred[:, 0]) +  # right → punish +dx
                        states_t[:, 2] * torch.relu(-pred[:, 0]) +  # left  → punish -dx
                        states_t[:, 4] * torch.relu( pred[:, 1]) +  # short → punish +dy
                        states_t[:, 5] * torch.relu(-pred[:, 1])    # long  → punish -dy
                    )
                ).mean()
                loss = loss + dir_loss_weight * dir_penalty

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(pol.gru.parameters()) + list(pol.head.parameters()), 1.0
            )
            optimiser.step()

            total_loss  += loss.item() * len(seq["states"])
            total_steps += len(seq["states"])

        avg_loss = total_loss / max(total_steps, 1)
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            click.echo(f"  Epoch {epoch:4d}/{epochs}  loss={avg_loss:.6f}")

    # Directional accuracy diagnostic on training data (no noise)
    _eval_directional_accuracy(pol, all_sequences)

    # Mark as trained and save
    pol._trained = True
    pol.save(str(checkpoint_path))
    click.echo(f"\nBC complete. Checkpoint saved to {checkpoint_path}")
    click.echo(
        "Next step: run GRU sessions with:\n"
        f"  pong-tune --policy gru --elastics N"
    )


# ---------------------------------------------------------------------------
# REINFORCE fine-tuning
# ---------------------------------------------------------------------------


def _run_rl(
    input_path: Path,
    checkpoint_path: Path,
    epochs: int,
    lr: float,
    gamma: float,
    min_shots: int,
    max_shots: int,
    sigma: float = 0.01,
) -> None:
    import torch
    from rl.policy import GRUPolicy

    if not checkpoint_path.exists():
        raise click.ClickException(
            f"Checkpoint {checkpoint_path} not found. "
            "Run 'pong-train-rl bc' first to warm-start the GRU."
        )

    records = _load_jsonl(input_path)
    episodes = _build_episodes(records, policy_filter="gru")
    if not episodes:
        raise click.ClickException(
            "No complete GRU sessions found in the input file. "
            "Run pong-tune --policy gru first to collect GRU trajectories."
        )

    sequences = _episodes_to_sequences(episodes, max_shots)
    sequences = [s for s in sequences if len(s["states"]) >= min_shots]

    if not sequences:
        raise click.ClickException(
            f"No episodes with >= {min_shots} shots after filtering. "
            "Lower --min-shots or collect more data."
        )

    hit_rate = sum(1 for s in sequences if s["hit"]) / len(sequences)
    mean_shots = sum(len(s["states"]) for s in sequences) / len(sequences)
    click.echo(
        f"RL: {len(sequences)} episodes  "
        f"hit_rate={hit_rate:.1%}  mean_shots={mean_shots:.1f}"
    )

    pol = GRUPolicy(max_shots=max_shots)
    pol.load(str(checkpoint_path))

    optimiser = torch.optim.Adam(
        list(pol.gru.parameters()) + list(pol.head.parameters()), lr=lr
    )

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_steps = 0

        for seq in sequences:
            states_t  = torch.tensor(seq["states"],  dtype=torch.float32)  # (T, 8)
            actions_t = torch.tensor(seq["actions"], dtype=torch.float32)  # (T, 2)
            rewards   = seq["rewards"]
            T = len(rewards)

            # Compute discounted returns
            returns = []
            G = 0.0
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)

            returns_t = torch.tensor(returns, dtype=torch.float32)

            # Normalise returns across this episode
            if T > 1:
                returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

            # Forward pass to get log-probs
            inp = states_t.unsqueeze(0)   # (1, T, 8)
            out, _ = pol.gru(inp)          # (1, T, hidden)
            mean_delta = pol.head(out[0])  # (T, 2)

            # Log-prob under Gaussian with sigma floor.
            # pol.sigma may be 0.0 (sigma=0 checkpoint) which would make
            # Normal(scale=0) invalid. Use max(pol.sigma, sigma) so there
            # is always a positive scale for log-prob computation.
            sigma_val = torch.tensor(
                max(float(pol.sigma), sigma), dtype=torch.float32
            )
            dist = torch.distributions.Normal(mean_delta, sigma_val)
            log_probs = dist.log_prob(actions_t).sum(dim=-1)  # (T,)

            # REINFORCE loss
            loss = -(log_probs * returns_t).mean()

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(pol.gru.parameters()) + list(pol.head.parameters()), 1.0
            )
            optimiser.step()

            total_loss  += loss.item()
            total_steps += 1

        avg_loss = total_loss / max(total_steps, 1)
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            click.echo(f"  Epoch {epoch:4d}/{epochs}  loss={avg_loss:.6f}")

    pol.save(str(checkpoint_path))
    click.echo(f"\nRL fine-tuning complete. Checkpoint saved to {checkpoint_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.group()
@click.option("--verbose", "-v", is_flag=True, default=False, help="Debug logging.")
def cli(verbose: bool) -> None:
    """Offline GRU policy training from logged shot data."""
    logging.basicConfig(
        format="%(levelname)-8s %(name)s: %(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
        stream=sys.stdout,
    )


@cli.command("bc")
@click.option("--elastics", required=True, type=int,
              help="Number of elastics on the launcher (e.g. 2 or 4). "
                   "Determines data directory: data/{N}_elastics/")
@click.option("--epochs", default=100, show_default=True, type=int,
              help="Training epochs.")
@click.option("--lr", default=1e-3, show_default=True, type=float,
              help="Learning rate.")
@click.option("--max-shots", default=20, show_default=True, type=int,
              help="Expected max shots per session (used to normalise shot_num in state).")
@click.option("--dir-loss-weight", default=2.0, show_default=True, type=float,
              help="Weight for directional sign penalty loss. Penalises predicted deltas "
                   "with the wrong sign given the last direction hint. Set 0 to disable.")
def bc_cmd(
    elastics: int,
    epochs: int,
    lr: float,
    max_shots: int,
    dir_loss_weight: float,
) -> None:
    """
    Behaviour Cloning: train the inner GRU to imitate heuristic trajectories.

    Reads heuristic sessions from the elastic-config data directory, trains
    the GRU to predict the same position deltas the heuristic used, and
    saves weights to data/{N}_elastics/checkpoints/gru.pt.

    Example:
        pong-train-rl bc --elastics 2
    """
    from utils.data_dir import elastics_data_dir, checkpoint_path as _ckpt_path
    data_dir = elastics_data_dir(elastics)
    input_path = str(data_dir / "rl_shots.jsonl")
    checkpoint = str(_ckpt_path(data_dir, "gru"))
    click.echo(
        f"Behaviour cloning: input={input_path}  checkpoint={checkpoint}  "
        f"epochs={epochs}  lr={lr}  dir_loss_weight={dir_loss_weight}"
    )
    _run_bc(
        input_path=Path(input_path),
        checkpoint_path=Path(checkpoint),
        epochs=epochs,
        lr=lr,
        max_shots=max_shots,
        dir_loss_weight=dir_loss_weight,
    )


@cli.command("rl")
@click.option("--elastics", required=True, type=int,
              help="Number of elastics on the launcher (e.g. 2 or 4). "
                   "Determines data directory: data/{N}_elastics/")
@click.option("--epochs", default=20, show_default=True, type=int,
              help="Training epochs.")
@click.option("--lr", default=1e-4, show_default=True, type=float,
              help="Learning rate (lower than BC to avoid overwriting warm-start).")
@click.option("--gamma", default=0.95, show_default=True, type=float,
              help="Discount factor for computing returns.")
@click.option("--min-shots", default=3, show_default=True, type=int,
              help="Skip episodes shorter than this many shots.")
@click.option("--max-shots", default=20, show_default=True, type=int,
              help="Expected max shots per session (used to normalise shot_num in state).")
@click.option("--sigma", default=0.01, show_default=True, type=float,
              help="Sigma floor for REINFORCE log-prob computation. "
                   "If the loaded checkpoint has sigma=0.0 (final eval policy), "
                   "this value is used instead to keep Normal(scale>0) valid. "
                   "Has no effect if checkpoint sigma exceeds this value.")
def rl_cmd(
    elastics: int,
    epochs: int,
    lr: float,
    gamma: float,
    min_shots: int,
    max_shots: int,
    sigma: float,
) -> None:
    """
    REINFORCE fine-tuning: improve the inner GRU using GRU session trajectories.

    Requires a BC-trained checkpoint and at least a few GRU sessions.
    Updates data/{N}_elastics/checkpoints/gru.pt in place.

    Example:
        pong-train-rl rl --elastics 2
    """
    from utils.data_dir import elastics_data_dir, checkpoint_path as _ckpt_path
    data_dir = elastics_data_dir(elastics)
    input_path = str(data_dir / "rl_shots.jsonl")
    checkpoint = str(_ckpt_path(data_dir, "gru"))
    click.echo(
        f"REINFORCE fine-tuning: input={input_path}  checkpoint={checkpoint}  "
        f"epochs={epochs}  lr={lr}  gamma={gamma}"
    )
    _run_rl(
        input_path=Path(input_path),
        checkpoint_path=Path(checkpoint),
        epochs=epochs,
        lr=lr,
        gamma=gamma,
        min_shots=min_shots,
        max_shots=max_shots,
        sigma=sigma,
    )


# ---------------------------------------------------------------------------
# Outer GRU training helpers
# ---------------------------------------------------------------------------


def _build_outer_episodes(records: list[dict]) -> list[list[dict]]:
    """
    Group outer_session_end records and reconstruct outer episodes.

    Each outer episode is a list of cup_result dicts in actual discovery
    order (sorted by found_index), sourced from the outer_session_end
    record's cup_results field.

    New schema fields expected per cup_result:
        found_index, outer_attempt_num, session_id,
        resolved_cup_x, resolved_cup_y,
        start_x, start_y, winning_x, winning_y,
        mean_miss_x, mean_miss_y, shots, outcome

    Episodes with fewer than 2 cup results are skipped (need at least 2
    cups to have one start recommendation to train on).
    """
    episodes = []
    skipped_no_cups = 0
    skipped_too_few = 0
    for r in records:
        if r.get("type") != "outer_session_end":
            continue
        cup_results = r.get("cup_results", [])
        if not cup_results:
            skipped_no_cups += 1
            continue
        if len(cup_results) < 2:
            skipped_too_few += 1
            continue
        # Sort by found_index (discovery order)
        sorted_cups = sorted(cup_results, key=lambda c: c.get("found_index", 0))
        episodes.append(sorted_cups)

    if skipped_no_cups:
        logger.info(
            "Outer: skipped %d outer_session_end records with no cup_results",
            skipped_no_cups,
        )
    if skipped_too_few:
        logger.info(
            "Outer: skipped %d outer_session_end records with fewer than 2 cups",
            skipped_too_few,
        )
    return episodes


def _farthest_point_from(
    known: list[tuple[float, float]],
    n_candidates: int = 1000,
) -> tuple[float, float]:
    """
    Return the normalised (x, y) that maximises minimum distance to all known points.

    Used as the BC target for the outer GRU: given cups already found,
    recommend starting as far as possible from all of them.

    Returns values in normalised space [X_MIN/NS, X_MAX/NS] x [Y_MIN/NS, Y_MAX/NS].
    """
    from rl.policy import NORMALISE_STEPS, X_MAX, X_MIN, Y_MAX, Y_MIN

    x_lo = X_MIN / NORMALISE_STEPS
    x_hi = X_MAX / NORMALISE_STEPS
    y_lo = Y_MIN / NORMALISE_STEPS
    y_hi = Y_MAX / NORMALISE_STEPS

    if not known:
        return 0.0, Y_MIN / NORMALISE_STEPS  # default start normalised

    import random as _random
    best_dist = -1.0
    best_pt = (0.0, 0.0)
    for _ in range(n_candidates):
        cx = _random.uniform(x_lo, x_hi)
        cy = _random.uniform(y_lo, y_hi)
        min_d = min(
            ((cx - kx) ** 2 + (cy - ky) ** 2) ** 0.5
            for kx, ky in known
        )
        if min_d > best_dist:
            best_dist = min_d
            best_pt = (cx, cy)
    return best_pt


def _run_outer(
    input_path: Path,
    checkpoint_path: Path,
    mode: str,
    epochs: int,
    lr: float,
    gamma: float,
    max_shots_per_cup: int,
) -> None:
    import torch
    import torch.nn as nn
    from rl.policy import NORMALISE_STEPS, OuterGRUPolicy

    records = _load_jsonl(input_path)
    episodes = _build_outer_episodes(records)

    if not episodes:
        raise click.ClickException(
            "No complete outer sessions found. "
            "Run pong-tune-outer first to collect outer session data."
        )

    click.echo(f"Outer {mode.upper()}: {len(episodes)} outer episodes loaded.")

    pol = OuterGRUPolicy(max_shots_per_cup=max_shots_per_cup)
    if checkpoint_path.exists():
        pol.load(str(checkpoint_path))
        click.echo(f"Loaded checkpoint from {checkpoint_path}")
    else:
        click.echo("No checkpoint - initialising fresh OuterGRUPolicy.")

    optimiser = torch.optim.Adam(
        list(pol.gru.parameters()) + list(pol.head.parameters()), lr=lr
    )

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_seqs = 0

        for ep_sessions in episodes:
            # Build input sequence and targets
            states: list[list[float]] = []
            targets: list[list[float]] = []  # BC targets or sampled actions
            rewards: list[float] = []

            known_norm: list[tuple[float, float]] = []

            for i, sess in enumerate(ep_sessions):
                wx = sess.get("winning_x") if sess.get("winning_x") is not None else 0
                wy = sess.get("winning_y") if sess.get("winning_y") is not None else 0
                shots = sess.get("shots", sess.get("total_shots", max_shots_per_cup))
                sx = sess.get("start_x", 0)
                sy = sess.get("start_y", 0)
                mmx = sess.get("mean_miss_x", sx)
                mmy = sess.get("mean_miss_y", sy)

                vec = [
                    wx / NORMALISE_STEPS,
                    wy / NORMALISE_STEPS,
                    shots / max_shots_per_cup,
                    sx / NORMALISE_STEPS,
                    sy / NORMALISE_STEPS,
                    mmx / NORMALISE_STEPS,
                    mmy / NORMALISE_STEPS,
                ]
                states.append(vec)

                # BC target: farthest point from all cups found so far
                if mode == "bc":
                    tx, ty = _farthest_point_from(known_norm)
                    targets.append([tx, ty])

                # RL: negative normalised shots as reward
                rewards.append(-shots / max_shots_per_cup)

                known_norm.append((wx / NORMALISE_STEPS, wy / NORMALISE_STEPS))

            if len(states) < 2:
                continue  # need at least 2 cups to have one start recommendation

            states_t = torch.tensor(states, dtype=torch.float32)  # (T, 7)

            if mode == "bc":
                # Supervised: predict BC target starts from GRU over input sequence
                # We predict start for cup K+1 given cups 1..K in hidden state,
                # so we run GRU over states[0..T-2] and predict targets[1..T-1]
                inp = states_t[:-1].unsqueeze(0)              # (1, T-1, 7)
                out, _ = pol.gru(inp)                          # (1, T-1, hidden)
                pred = pol.head(out[0])                        # (T-1, 2)
                tgt = torch.tensor(targets[1:], dtype=torch.float32)
                loss = nn.functional.mse_loss(pred, tgt)

            else:  # rl
                # REINFORCE: compute discounted returns
                T = len(rewards)
                returns = []
                G = 0.0
                for r in reversed(rewards[1:]):  # rewards for cups 2..T
                    G = r + gamma * G
                    returns.insert(0, G)

                if not returns:
                    continue

                returns_t = torch.tensor(returns, dtype=torch.float32)
                if len(returns_t) > 1:
                    returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

                # Run GRU over first T-1 states, predict starts for cups 2..T
                inp = states_t[:-1].unsqueeze(0)
                out, _ = pol.gru(inp)
                mean_start = pol.head(out[0])   # (T-1, 2)

                # Actual starts used for cups 2..T (from session_end records)
                actual_starts = torch.tensor(
                    [
                        [
                            ep_sessions[k].get("start_x", 0) / NORMALISE_STEPS,
                            ep_sessions[k].get("start_y", 0) / NORMALISE_STEPS,
                        ]
                        for k in range(1, len(ep_sessions))
                    ],
                    dtype=torch.float32,
                )  # (T-1, 2)

                sigma = torch.tensor(pol.sigma, dtype=torch.float32)
                dist = torch.distributions.Normal(mean_start, sigma)
                log_probs = dist.log_prob(actual_starts).sum(dim=-1)  # (T-1,)
                loss = -(log_probs * returns_t).mean()

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(pol.gru.parameters()) + list(pol.head.parameters()), 1.0
            )
            optimiser.step()
            total_loss += loss.item()
            total_seqs += 1

        avg_loss = total_loss / max(total_seqs, 1)
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            click.echo(f"  Epoch {epoch:4d}/{epochs}  loss={avg_loss:.6f}")

    pol._trained = True
    pol.save(str(checkpoint_path))
    click.echo(f"\nOuter {mode.upper()} complete. Saved to {checkpoint_path}")


# ---------------------------------------------------------------------------
# outer subcommand
# ---------------------------------------------------------------------------


@cli.command("outer")
@click.option("--elastics", required=True, type=int,
              help="Number of elastics on the launcher (e.g. 2 or 4). "
                   "Determines data directory: data/{N}_elastics/")
@click.option(
    "--mode",
    type=click.Choice(["bc", "rl"], case_sensitive=False),
    default="bc",
    show_default=True,
    help=(
        "bc - behaviour cloning: train outer GRU to start as far as possible\n"
        "     from already-found cups (geometric farthest-point target).\n"
        "rl - REINFORCE fine-tuning: train outer GRU using actual shot outcomes\n"
        "     from pong-tune-outer sessions."
    ),
)
@click.option("--epochs", default=50, show_default=True, type=int,
              help="Training epochs.")
@click.option("--lr", default=1e-3, show_default=True, type=float,
              help="Learning rate.")
@click.option("--gamma", default=0.95, show_default=True, type=float,
              help="Discount factor for returns. Only used with --mode rl.")
@click.option("--max-shots-per-cup", default=20, show_default=True, type=int,
              help="Expected max shots per cup (used for normalisation).")
@click.option(
    "--input", "input_override", default=None, type=click.Path(exists=True),
    help="Override input JSONL path. Defaults to data/{N}_elastics/rl_shots.jsonl. "
         "Use this to point at synthetic_outer.jsonl for BC warm-start training: "
         "pong-train-rl outer --mode bc --input data/5_elastics/synthetic_outer.jsonl",
)
def outer_cmd(
    elastics: int,
    mode: str,
    epochs: int,
    lr: float,
    gamma: float,
    max_shots_per_cup: int,
    input_override: str | None,
) -> None:
    """
    Train the OuterGRUPolicy from outer session logs.

    Two modes:
      bc  - warm-start using geometric farthest-point targets.
      rl  - REINFORCE fine-tuning using actual pong-tune-outer trajectories.

    For synthetic BC warm-start (before real outer sessions exist), generate
    synthetic data first with pong-gen-synthetic-outer and pass it via --input:

        pong-gen-synthetic-outer --elastics 5
        pong-train-rl outer --elastics 5 --mode bc \\
            --input data/5_elastics/synthetic_outer.jsonl

    Example (real data):
        pong-train-rl outer --elastics 2 --mode bc
        pong-train-rl outer --elastics 2 --mode rl
    """
    from utils.data_dir import elastics_data_dir, checkpoint_path as _ckpt_path
    data_dir = elastics_data_dir(elastics)
    input_path = input_override if input_override else str(data_dir / "rl_shots.jsonl")
    checkpoint = str(_ckpt_path(data_dir, "outer_gru"))
    click.echo(
        f"Outer {mode.upper()}: input={input_path}  checkpoint={checkpoint}  "
        f"epochs={epochs}  lr={lr}"
        + (f"  gamma={gamma}" if mode == "rl" else "")
    )
    _run_outer(
        input_path=Path(input_path),
        checkpoint_path=Path(checkpoint),
        mode=mode,
        epochs=epochs,
        lr=lr,
        gamma=gamma,
        max_shots_per_cup=max_shots_per_cup,
    )


if __name__ == "__main__":
    cli()
