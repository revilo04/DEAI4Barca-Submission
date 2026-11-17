import numpy as np

from dataclasses import dataclass

from pipeline.pauses.constants import *

@dataclass
class Match:
    """Per-frame numpy arrays for the whole video, positions in metres."""
    ball_x: np.ndarray        # (n_frames,)
    ball_y: np.ndarray        # (n_frames,)
    ball_seen: np.ndarray     # (n_frames,) bool
    player_x: np.ndarray      # (n_frames, n_slots)
    player_y: np.ndarray      # (n_frames, n_slots)
    speed: np.ndarray         # (n_frames, n_slots) m/s
    team: np.ndarray          # (n_slots,) 0 = home, 1 = away
    half: np.ndarray          # (n_frames,) 0 = not graded, 1 / 2


def get_runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Start and end indices (both inclusive) of each run of consecutive True values.

    Example: [F, T, T, F, T] -> starts [1, 4], ends [2, 4]
    """
    # Pad with False so runs touching either end still have a start and an end edge
    padded = np.concatenate([[False], mask, [False]])
    # Indices where the value flips; they alternate: start, end+1, start, end+1, ...
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return edges[0::2], edges[1::2] - 1


def ball_frames_around(start: int, seen_frames: np.ndarray) -> tuple[int | None, int | None]:
    """
    Last frame with the ball seen before a pause, and first one after it.

    `seen_frames` is the sorted array of frames where the ball is visible. Since
    every frame inside the pause has no ball, the first seen frame >= start is
    the first one after the pause, and the one right before it is the last one
    before the pause.

    Returns (None, None) unless both exist (a pause at the very start or end of
    the tracking has no neighbour on one side).
    """
    i = np.searchsorted(seen_frames, start)
    if i == 0 or i == len(seen_frames):
        return None, None
    return int(seen_frames[i - 1]), int(seen_frames[i])


def keeper_distance(
    frame: int | None, half: int, match: Match, keepers: dict[tuple[int, int], int]
) -> float:
    """
    Distance (m) from the ball to the nearest of the two goalkeepers at `frame`.

    Keepers come from identify_keepers for this half. A keeper that was not
    found, or is not tracked on this frame, is ignored. NaN if no distance can
    be computed.
    """
    if frame is None:
        return np.nan

    distances = []
    for team in (0, 1):
        slot = keepers.get((half, team))
        if slot is None:
            continue  # no keeper identified for this team in this half
        d = np.hypot(
            match.ball_x[frame] - match.player_x[frame, slot],
            match.ball_y[frame] - match.player_y[frame, slot],
        )
        if not np.isnan(d):  # NaN if the keeper (or ball) is not tracked here
            distances.append(d)
    return float(min(distances)) if distances else np.nan


def mean_player_speed(start: int, end: int, match: Match) -> float:
    """
    Average speed (m/s) of the tracked players during frames [start, end].

    Sampled once per second to keep it cheap. Sampled frames with fewer than
    MIN_TRACKED_PLAYERS tracked players are skipped, since a mean over a handful
    of players is not representative. NaN if no sampled frame qualifies.
    """
    sampled = match.speed[start : end + 1 : max(1, int(FPS))]  # (n_samples, n_slots)
    n_tracked = np.sum(~np.isnan(sampled), axis=1)
    sampled = sampled[n_tracked >= MIN_TRACKED_PLAYERS]
    if len(sampled) == 0:
        return np.nan
    # Mean over players on each sampled frame, then mean over frames
    return float(np.nanmean(sampled, axis=1).mean())


def distance_to_touchline(frame: int | None, match: Match) -> float:
    """Distance (m) from the ball at `frame` to the nearest touchline (y = 0 or y = width)."""
    if frame is None:
        return np.nan
    y = match.ball_y[frame]
    return float(min(abs(y), abs(PITCH_WIDTH - y)))


def ball_displacement(before: int | None, after: int | None, match: Match) -> float:
    """Straight-line distance (m) between the ball positions at `before` and `after`."""
    if before is None or after is None:
        return np.nan
    return float(np.hypot(
        match.ball_x[after] - match.ball_x[before],
        match.ball_y[after] - match.ball_y[before],
    ))