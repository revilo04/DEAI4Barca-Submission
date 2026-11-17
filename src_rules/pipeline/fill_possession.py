"""
Label which team has the ball (H / A) on every in-play frame.

Pipeline (see fill_possession):
    tracking               -> Frames arrays (metres, in-play frames only)
    Frames                 -> touches                          (T1-T3)
    pauses + Frames        -> dead-ball intervals, keeper holds
    dead balls + touches   -> restarts (R1-R2) and kickoffs     (R5)
    all events             -> possession state machine         (P1-P5)
    possession starts      -> one label per frame              (L1)

Frames where possession is unknown get "?"; main fills them afterwards.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from constants import FIRST_HALF_END, FIRST_HALF_START, SECOND_HALF_END, SECOND_HALF_KICKOFF
from games import GAME, GAMES
from pipeline.pauses.fill_pauses import get_runs

# --- Match / pitch -----------------------------------------------------------
FPS = GAMES[GAME]["fps"]
# The original notebook used 105 x 68 m, while games.py says 105.3 m long. Kept
# at 105.0 so results match the notebook; changing it shifts every distance.
PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0
N_SLOTS = 46                                   # tracking columns 1_x .. 46_x

# --- Labels ------------------------------------------------------------------
HOME, AWAY, UNKNOWN = "H", "A", "?"
PAUSE_DEAD_BALL = "D"                          # pause label meaning dead ball

# --- Touches (T1-T3) ---------------------------------------------------------
SMOOTH_WINDOW = 5              # frames, centred rolling median of the ball position
TOUCH_LAG = 5                  # frames before / after a frame to measure ball motion
MAX_TOUCH_DISTANCE = 1.5       # m, nearest player to the smoothed ball
TOUCH_SPEED_CHANGE = 1.0       # m/s
TOUCH_DIRECTION_CHANGE = 20.0  # degrees
CLUSTER_GAP_FRAMES = 5         # candidate frames this close belong to the same touch
SCORE_SPEED_SCALE = 2.5        # touch frame score = speed change / 2.5 ...
SCORE_DIRECTION_SCALE = 40.0   # ... + direction change / 40

# --- Restarts (R1-R2) and kickoffs (R5) --------------------------------------
MAX_RESTART_FRAMES = int(10.0 * FPS)   # how long after a dead ball to look for the restart
RESTART_SPEED_LAG = 5          # frames, causal ball speed
RESTART_SPEED = 1.0            # m/s, the ball counts as moving above this
RESTART_CHECK_WINDOW = 5       # frames
RESTART_MIN_MOVING = 3         # moving frames needed within the check window
PLAYER_NEAR_RADIUS = 10.0      # m, nearest player must be this close to vote
TEAM_EVIDENCE_FRAMES = int(0.5 * FPS)  # vote until this long after the observed restart
TOUCH_TEAM_WEIGHT = 2.0        # each touch is worth this many votes...
MAX_TOUCHES_COUNTED = 2        # ...for at most this many touches per team
MIN_TEAM_CONFIDENCE = 0.25     # below this, fall back to simpler evidence

# --- Possession (P4) ---------------------------------------------------------
MIN_TOUCHES = 2                # touches in a row a team needs to take possession
MAX_TOUCH_GAP_S = 10.0         # max time between two touches of the same sequence

# --- Events, in the order they are processed when on the same frame -----------
DEAD_BALL, KEEPER_HOLD, RESTART, TOUCH = "DEAD_BALL", "GOALKEEPER_HOLD", "RESTART", "TOUCH"
EVENT_PRIORITY = {DEAD_BALL: 0, KEEPER_HOLD: 1, RESTART: 2, TOUCH: 3}


@dataclass
class Frames:
    """Per-frame arrays. Positions are in metres and NaN outside in-play frames."""
    half: np.ndarray          # (n_frames,) 1 / 2 from kickoff to end of half, 0 elsewhere
    ball_x: np.ndarray        # (n_frames,)
    ball_y: np.ndarray        # (n_frames,)
    player_x: np.ndarray      # (n_frames, n_slots)
    player_y: np.ndarray      # (n_frames, n_slots)
    slot_team: np.ndarray     # (n_slots,) H / A
    nearest_dist: np.ndarray  # (n_frames,) m from the ball to the nearest player, NaN if none
    nearest_team: np.ndarray  # (n_frames,) H / A of that player, None if none

    @property
    def n(self) -> int:
        return len(self.half)

    @property
    def in_play(self) -> np.ndarray:
        return self.half > 0


@dataclass
class Intervals:
    """Sorted, disjoint [start, end] frame intervals (both ends included)."""
    starts: np.ndarray
    ends: np.ndarray

    def __len__(self) -> int:
        return len(self.starts)

    def contains(self, frames) -> np.ndarray | bool:
        """Whether each frame lies inside one of the intervals."""
        if len(self) == 0:
            return np.zeros(np.shape(frames), dtype=bool)
        pos =np.searchsorted(self.starts, frames, side="right") - 1
        return (pos >= 0) & (frames <= self.ends[np.maximum(pos, 0)])

    def next_start_after(self, frame: int, default: int) -> int:
        """First interval start strictly after `frame`, or `default` if none."""
        pos = np.searchsorted(self.starts, frame, side="right")
        return int(self.starts[pos]) if pos < len(self.starts) else default


# =============================================================================
# Entry point
# =============================================================================
def fill_possession(
    submission: pd.DataFrame, tracking: pd.DataFrame, team: np.ndarray, pauses: pd.DataFrame
) -> pd.DataFrame:
    """
    Copy of `submission` with every in-play frame labelled H, A or "?".

    Args:
        submission: one row per video frame, columns `frame` and `label`.
        tracking:   tactical tracking dataframe, row i = video frame i.
        team:       team (0 = home, 1 = away) of each tracking slot.
        pauses:     labelled pauses from fill_pauses.label_pauses
                    (columns start, end, label).
    """
    frames = build_frames(tracking, team)
    dead_balls = build_dead_balls(pauses, frames)
    keeper_holds = pauses[pauses["label"].isin([HOME, AWAY])]

    touches = detect_touches(frames)
    touches = touches[~dead_balls.contains(touches["frame"].to_numpy())]  # no live touches in a dead ball

    restarts = detect_restarts(dead_balls, touches, frames)
    kickoffs = detect_kickoffs(dead_balls, touches, frames)

    events = build_events(touches, restarts + kickoffs, keeper_holds, dead_balls, frames.n)
    starts = track_possession(events, frames)
    possession = expand_possession(starts, dead_balls, frames)

    labels = submission["label"].to_numpy(copy=True)
    labels[frames.in_play] = possession[frames.in_play]
    return submission.assign(label=labels)


# =============================================================================
# Frame arrays
# =============================================================================
def build_frames(tracking: pd.DataFrame, team: np.ndarray) -> Frames:
    """
    Per-frame arrays in metres, keeping positions only on in-play frames.

    Also finds the player nearest to the (raw) ball on every frame, which the
    restart rules use.
    """
    def columns(suffix: str) -> np.ndarray:
        cols = [f"{slot}_{suffix}" for slot in range(1, N_SLOTS + 1)]
        return tracking[cols].to_numpy(dtype=float, na_value=np.nan)

    half = play_half_index(len(tracking))
    in_play = half > 0

    ball_x = tracking["ball_x"].to_numpy(dtype=float, na_value=np.nan) * PITCH_LENGTH
    ball_y = tracking["ball_y"].to_numpy(dtype=float, na_value=np.nan) * PITCH_WIDTH
    ball_ok = in_play & np.isfinite(ball_x) & np.isfinite(ball_y)

    player_x = columns("x") * PITCH_LENGTH
    player_y = columns("y") * PITCH_WIDTH
    player_ok = in_play[:, None] & np.isfinite(player_x) & np.isfinite(player_y)

    ball_x, ball_y = np.where(ball_ok, ball_x, np.nan), np.where(ball_ok, ball_y, np.nan)
    player_x, player_y = np.where(player_ok, player_x, np.nan), np.where(player_ok, player_y, np.nan)

    slot_team = np.where(np.asarray(team) == 0, HOME, AWAY)
    slot, dist = nearest_player(player_x, player_y, ball_x, ball_y)
    nearest_team = np.where(slot >= 0, slot_team[slot], None)

    return Frames(half, ball_x, ball_y, player_x, player_y, slot_team, dist, nearest_team)


def play_half_index(n_frames: int) -> np.ndarray:
    """Half (1 / 2) of each frame from kickoff to the end of that half; 0 elsewhere."""
    half = np.zeros(n_frames, dtype=int)
    half[FIRST_HALF_START : FIRST_HALF_END + 1] = 1
    half[SECOND_HALF_KICKOFF : SECOND_HALF_END + 1] = 2
    return half


def nearest_player(
    player_x: np.ndarray, player_y: np.ndarray, ball_x: np.ndarray, ball_y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slot of the player closest to the ball on each frame, and their distance (m).

    Slot is -1 and distance NaN when the ball or every player is missing.
    """
    dist = np.hypot(player_x - ball_x[:, None], player_y - ball_y[:, None])
    dist[np.isnan(dist)] = np.inf
    slot = np.argmin(dist, axis=1)
    nearest = dist[np.arange(len(dist)), slot]
    found = np.isfinite(nearest)
    return np.where(found, slot, -1), np.where(found, nearest, np.nan)


# =============================================================================
# Touches (T1-T3)
# =============================================================================
def detect_touches(frames: Frames) -> pd.DataFrame:
    """
    One row per touch: the frame it happens on and the team that made it.

    T1: a frame is a touch candidate if a player is close to the ball and the
        ball changes speed or direction there.
    T2: candidate frames close together are one touch; keep its best frame.
    T3: the touch belongs to the team that is nearest across the cluster.
    """
    ball_x, ball_y = smooth_ball(frames)
    speed_change, direction_change = ball_turn(ball_x, ball_y)
    slot, dist = nearest_player(frames.player_x, frames.player_y, ball_x, ball_y)

    candidates = np.flatnonzero(is_touch_candidate(speed_change, direction_change, dist))
    score = speed_change / SCORE_SPEED_SCALE + direction_change / SCORE_DIRECTION_SCALE

    rows = [
        {"frame": best_touch_frame(cluster, score),
         "team": majority_team(frames.slot_team[slot[cluster]])}
        for cluster in cluster_frames(candidates, CLUSTER_GAP_FRAMES)
    ]
    return pd.DataFrame(rows, columns=["frame", "team"])


def smooth_ball(frames: Frames) -> tuple[np.ndarray, np.ndarray]:
    """Ball position smoothed with a centred rolling median (NaN unless the whole window has a ball)."""
    def smooth(values):
        return (pd.Series(values)
                .rolling(SMOOTH_WINDOW, center=True, min_periods=SMOOTH_WINDOW)
                .median().to_numpy())
    return smooth(frames.ball_x), smooth(frames.ball_y)


def ball_turn(ball_x: np.ndarray, ball_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    How much the ball changes speed (m/s) and direction (degrees, 0-180) at each frame.

    Compares the ball's velocity over the TOUCH_LAG frames before the frame with
    the velocity over the TOUCH_LAG frames after it. NaN unless the ball is seen
    on every frame of that window.
    """
    lag, dt = TOUCH_LAG, TOUCH_LAG / FPS
    ok = np.isfinite(ball_x) & np.isfinite(ball_y)
    f = np.flatnonzero(all_valid_around(ok, lag))

    vx_pre, vy_pre = (ball_x[f] - ball_x[f - lag]) / dt, (ball_y[f] - ball_y[f - lag]) / dt
    vx_post, vy_post = (ball_x[f + lag] - ball_x[f]) / dt, (ball_y[f + lag] - ball_y[f]) / dt

    speed_change = np.full(len(ball_x), np.nan)
    direction_change = np.full(len(ball_x), np.nan)
    speed_change[f] = np.abs(np.hypot(vx_post, vy_post) - np.hypot(vx_pre, vy_pre))
    direction_change[f] = angle_between(
        np.degrees(np.arctan2(vy_pre, vx_pre)), np.degrees(np.arctan2(vy_post, vx_post))
    )
    return speed_change, direction_change


def is_touch_candidate(
    speed_change: np.ndarray, direction_change: np.ndarray, nearest_dist: np.ndarray
) -> np.ndarray:
    """
    T1. A player is within MAX_TOUCH_DISTANCE of the ball and the ball changes
    speed by TOUCH_SPEED_CHANGE or direction by TOUCH_DIRECTION_CHANGE.

    NaN inputs (no ball motion or no player) compare as False.
    """
    player_close = nearest_dist <= MAX_TOUCH_DISTANCE
    ball_turns = (speed_change >= TOUCH_SPEED_CHANGE) | (direction_change >= TOUCH_DIRECTION_CHANGE)
    return player_close & ball_turns


def cluster_frames(frames: np.ndarray, max_gap: int) -> list[np.ndarray]:
    """T2. Split sorted frames into groups wherever two neighbours are more than `max_gap` apart."""
    if len(frames) == 0:
        return []
    return np.split(frames, np.flatnonzero(np.diff(frames) > max_gap) + 1)


def best_touch_frame(cluster: np.ndarray, score: np.ndarray) -> int:
    """T2. The frame of the cluster where the ball turns the most (highest score)."""
    return int(cluster[np.nanargmax(score[cluster])])


def majority_team(teams: np.ndarray) -> str:
    """T3. The team that appears most often (a tie goes to the home team)."""
    return HOME if np.sum(teams == HOME) >= np.sum(teams == AWAY) else AWAY


# =============================================================================
# Dead balls
# =============================================================================
def build_dead_balls(pauses: pd.DataFrame, frames: Frames) -> Intervals:
    """
    Every stretch of frames where the ball is dead: the pauses labelled D, plus
    every run of in-play frames where the ball is outside the pitch. Overlapping
    or touching intervals are merged.
    """
    dead_pauses = pauses[pauses["label"] == PAUSE_DEAD_BALL]
    out_starts, out_ends = get_runs(ball_out_of_pitch(frames))
    return merge_intervals(
        np.concatenate([dead_pauses["start"].to_numpy(), out_starts]),
        np.concatenate([dead_pauses["end"].to_numpy(), out_ends]),
    )


def ball_out_of_pitch(frames: Frames) -> np.ndarray:
    """Frames where the ball is seen outside the pitch lines."""
    x, y = frames.ball_x, frames.ball_y
    return (x < 0) | (x > PITCH_LENGTH) | (y < 0) | (y > PITCH_WIDTH)


def merge_intervals(starts: np.ndarray, ends: np.ndarray) -> Intervals:
    """Sort intervals and merge the ones that overlap or touch (end + 1 == next start)."""
    merged: list[list[int]] = []
    for s, e in sorted(zip(starts.tolist(), ends.tolist())):
        if merged and s <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    arr = np.array(merged, dtype=int).reshape(-1, 2)
    return Intervals(arr[:, 0], arr[:, 1])


# =============================================================================
# Restarts (R1-R2) and kickoffs (R5)
# =============================================================================
def detect_restarts(
    dead_balls: Intervals, touches: pd.DataFrame, frames: Frames
) -> list[tuple[int, str]]:
    """
    (frame, team) of each restart: the team that restarts play after each dead
    ball. The restart is placed on the first frame after the dead ball.
    """
    speed = causal_ball_speed(frames)
    restarts = []
    for i in range(len(dead_balls)):
        team = restart_team_after(i, dead_balls, touches, speed, frames)
        if team is not None:
            restarts.append((int(dead_balls.ends[i]) + 1, team))
    return restarts


def restart_team_after(
    i: int, dead_balls: Intervals, touches: pd.DataFrame, speed: np.ndarray, frames: Frames
) -> str | None:
    """
    R1-R2 for dead ball `i`: who restarts play, or None if no restart is seen.

    R1: the restart is observed where the ball starts moving, or else at the
        first touch after the dead ball.
    R2: the team is decided by player-proximity votes and touch counts around
        the restart, with fallbacks when that is inconclusive.
    """
    window = restart_search_window(i, dead_balls, frames.n)
    if window is None:
        return None
    start, end, limit = window

    moving_from = ball_starts_moving(speed, start, end)
    first_touch = first_touch_between(touches, start, limit)
    if moving_from is None and first_touch is None:
        return None  # NOT_DETECTED
    observed = moving_from if moving_from is not None else first_touch[0]

    # R2: votes from the nearest player, plus weighted touches in the search window
    votes_end = min(observed + TEAM_EVIDENCE_FRAMES, end)
    votes = proximity_votes(frames, start, votes_end)
    touch_counts = touches_per_team(touches, start, end)
    team, confidence = team_from_evidence(votes, touch_counts)

    if team is None or confidence < MIN_TEAM_CONFIDENCE:
        fallback = first_known(
            first_touch[1] if first_touch else None,
            frames.nearest_team[observed],
            frames.nearest_team[start],
        )
        team = fallback or team
    return team


def restart_search_window(i: int, dead_balls: Intervals, n_frames: int) -> tuple[int, int, int] | None:
    """
    R1. Frames to look for the restart after dead ball `i`.

    Returns (start, end, limit): the window runs from the first frame after the
    dead ball for up to MAX_RESTART_FRAMES, stopping before the next dead ball.
    `limit` is the last frame before the next dead ball (used for touches that
    come late). None if the window is empty.
    """
    start = int(dead_balls.ends[i]) + 1
    limit = (int(dead_balls.starts[i + 1]) - 1) if i + 1 < len(dead_balls) else n_frames - 1
    end = min(start - 1 + MAX_RESTART_FRAMES, n_frames - 1, limit)
    return (start, end, limit) if end > start else None


def causal_ball_speed(frames: Frames) -> np.ndarray:
    """
    Ball speed (m/s) at each frame, from its position RESTART_SPEED_LAG frames earlier.

    Uses only past frames, so it marks the moment the ball starts moving. NaN
    unless the ball is seen on every frame from f - lag to f.
    """
    lag = RESTART_SPEED_LAG
    ok = np.isfinite(frames.ball_x) & np.isfinite(frames.ball_y)
    seen_so_far = np.concatenate([[0], np.cumsum(ok)])
    f = np.arange(lag, frames.n)
    f = f[seen_so_far[f + 1] - seen_so_far[f - lag] == lag + 1]

    speed = np.full(frames.n, np.nan)
    speed[f] = np.hypot(
        frames.ball_x[f] - frames.ball_x[f - lag], frames.ball_y[f] - frames.ball_y[f - lag]
    ) / (lag / FPS)
    return speed


def ball_starts_moving(speed: np.ndarray, start: int, end: int) -> int | None:
    """
    R1. First frame of the first RESTART_CHECK_WINDOW frames in which at least
    RESTART_MIN_MOVING have a causal ball speed of RESTART_SPEED or more.

    Returns the first moving frame of that window, or None if the ball never
    moves within [start, end].
    """
    first_check = max(start + RESTART_SPEED_LAG, RESTART_CHECK_WINDOW - 1)
    for f in range(first_check, end + 1):
        lo = max(start, f - RESTART_CHECK_WINDOW + 1)
        moving = speed[lo : f + 1] >= RESTART_SPEED  # NaN -> False
        if moving.sum() >= RESTART_MIN_MOVING:
            return lo + int(np.argmax(moving))
    return None


def first_touch_between(touches: pd.DataFrame, lo: int, hi: int) -> tuple[int, str] | None:
    """(frame, team) of the first touch in frames [lo, hi], or None."""
    inside = touches[touches["frame"].between(lo, hi)]
    if inside.empty:
        return None
    first = inside.iloc[0]
    return int(first["frame"]), first["team"]


def touches_per_team(touches: pd.DataFrame, lo: int, hi: int) -> dict[str, int]:
    """Number of touches by each team in frames [lo, hi]."""
    teams = touches.loc[touches["frame"].between(lo, hi), "team"]
    return {HOME: int((teams == HOME).sum()), AWAY: int((teams == AWAY).sum())}


def proximity_votes(frames: Frames, lo: int, hi: int) -> dict[str, int]:
    """R2. One vote per frame in [lo, hi] for the team of the nearest player, if within PLAYER_NEAR_RADIUS."""
    near = frames.nearest_dist[lo : hi + 1] <= PLAYER_NEAR_RADIUS
    teams = frames.nearest_team[lo : hi + 1]
    return {HOME: int(np.sum(near & (teams == HOME))), AWAY: int(np.sum(near & (teams == AWAY)))}


def team_from_evidence(votes: dict[str, int], touches: dict[str, int]) -> tuple[str | None, float]:
    """
    R2. Team with the higher score, and how clear the win is (0 = tie, 1 = unanimous).

    score = proximity votes + TOUCH_TEAM_WEIGHT x touches (at most MAX_TOUCHES_COUNTED).
    Team is None on a tie.
    """
    score = {
        team: votes[team] + TOUCH_TEAM_WEIGHT * min(touches[team], MAX_TOUCHES_COUNTED)
        for team in (HOME, AWAY)
    }
    total = score[HOME] + score[AWAY]
    confidence = abs(score[HOME] - score[AWAY]) / total if total > 0 else 0.0
    if score[HOME] == score[AWAY]:
        return None, confidence
    return (HOME if score[HOME] > score[AWAY] else AWAY), confidence


def first_known(*teams: str | None) -> str | None:
    """R2 fallback. The first of `teams` that is H or A."""
    return next((t for t in teams if t in (HOME, AWAY)), None)


def detect_kickoffs(
    dead_balls: Intervals, touches: pd.DataFrame, frames: Frames
) -> list[tuple[int, str]]:
    """
    R5. (frame, team) of each half's kickoff, unless it falls inside a dead ball.

    The team is the one making the first touch from the kickoff on, or else the
    team of the player nearest the ball at kickoff.
    """
    kickoffs = []
    for frame in (FIRST_HALF_START, SECOND_HALF_KICKOFF):
        if dead_balls.contains(frame):
            continue
        later = touches[touches["frame"] >= frame]
        team = later.iloc[0]["team"] if not later.empty else frames.nearest_team[frame]
        if team in (HOME, AWAY):
            kickoffs.append((frame, team))
    return kickoffs


# =============================================================================
# Possession state machine (P1-P5)
# =============================================================================
@dataclass
class PossessionState:
    """What the state machine remembers between events."""
    team: str = UNKNOWN                     # current possession
    candidate_team: str | None = None       # team of the touch sequence being built
    candidate_frames: list[int] = field(default_factory=list)
    half: int | None = None                 # half of the previous event
    starts: list[tuple[int, str]] = field(default_factory=list)  # (frame, team) of each possession start


def build_events(
    touches: pd.DataFrame,
    restarts: list[tuple[int, str]],
    keeper_holds: pd.DataFrame,
    dead_balls: Intervals,
    n_frames: int,
) -> list[tuple[int, str, str | None]]:
    """
    Every event as (frame, type, team), sorted by frame and then EVENT_PRIORITY.

    A keeper hold produces two events: at its start, and again on the frame
    after it ends (unless that frame is a dead ball), so possession is kept.
    """
    events = [(int(f), TOUCH, t) for f, t in zip(touches["frame"], touches["team"])]
    events += [(f, RESTART, t) for f, t in restarts]
    for hold in keeper_holds.itertuples():
        events.append((int(hold.start), KEEPER_HOLD, hold.label))
        after = int(hold.end) + 1
        if after < n_frames and not dead_balls.contains(after):
            events.append((after, KEEPER_HOLD, hold.label))
    events += [(int(s), DEAD_BALL, None) for s in dead_balls.starts]
    return sorted(events, key=lambda e: (e[0], EVENT_PRIORITY[e[1]]))


def track_possession(events: list[tuple[int, str, str | None]], frames: Frames) -> list[tuple[int, str]]:
    """Run the possession state machine over the events; returns (frame, team) of each possession start."""
    state = PossessionState()
    for frame, kind, team in events:
        half = int(frames.half[frame])
        on_half_change(state, half)                 # P5
        if kind == DEAD_BALL:
            on_dead_ball(state)                     # P1
        elif half == 0:
            continue                                # only dead balls count outside the halves
        elif kind in (KEEPER_HOLD, RESTART):
            on_possession_start(state, frame, team)  # P2, P3
        else:
            on_touch(state, frame, team)             # P4
    return state.starts


def reset_possession(state: PossessionState) -> None:
    """Forget the current possession and any touch sequence being built."""
    state.team = UNKNOWN
    state.candidate_team = None
    state.candidate_frames = []


def on_half_change(state: PossessionState, half: int) -> None:
    """P5. Possession never carries over from one half (or from outside the halves) to the next."""
    if state.half is not None and half != state.half:
        reset_possession(state)
    state.half = half


def on_dead_ball(state: PossessionState) -> None:
    """P1. Nobody has the ball while it is dead."""
    reset_possession(state)


def on_possession_start(state: PossessionState, frame: int, team: str) -> None:
    """P2 / P3. A keeper hold or a restart hands the ball to `team` straight away."""
    reset_possession(state)
    state.team = team
    state.starts.append((frame, team))


def on_touch(state: PossessionState, frame: int, team: str) -> None:
    """
    P4. MIN_TOUCHES touches in a row by the same team, each at most
    MAX_TOUCH_GAP_S after the previous one, give that team the ball.

    The new possession starts at the first touch of the sequence.
    """
    continues_sequence = (
        team == state.candidate_team
        and state.candidate_frames
        and (frame - state.candidate_frames[-1]) / FPS <= MAX_TOUCH_GAP_S
    )
    if continues_sequence:
        state.candidate_frames.append(frame)
    else:
        state.candidate_team, state.candidate_frames = team, [frame]

    confirmed = len(state.candidate_frames) >= MIN_TOUCHES
    if confirmed and state.candidate_team != state.team:
        state.team = state.candidate_team
        state.starts.append((state.candidate_frames[0], state.team))


# =============================================================================
# Frame labels (L1)
# =============================================================================
def expand_possession(starts: list[tuple[int, str]], dead_balls: Intervals, frames: Frames) -> np.ndarray:
    """
    L1. One label per frame: each possession start holds until the next start,
    the next dead ball or the end of its half. Dead balls and frames outside
    play are "?", as is anything no possession reaches.
    """
    starts = one_start_per_frame(starts)
    labels = np.full(frames.n, UNKNOWN, dtype=object)
    for i, (start, team) in enumerate(starts):
        next_start = starts[i + 1][0] if i + 1 < len(starts) else frames.n
        labels[start : possession_end(start, next_start, dead_balls, frames.n)] = team

    labels[dead_balls.contains(np.arange(frames.n))] = UNKNOWN
    labels[~frames.in_play] = UNKNOWN
    return labels


def one_start_per_frame(starts: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Sort starts by frame; when several share a frame, keep the last one recorded."""
    by_frame = {}
    for frame, team in sorted(starts, key=lambda s: s[0]):  # stable: keeps recording order
        by_frame[frame] = team
    return list(by_frame.items())


def possession_end(start: int, next_start: int, dead_balls: Intervals, n_frames: int) -> int:
    """First frame (exclusive) a possession starting at `start` no longer covers."""
    half_end = FIRST_HALF_END + 1 if start <= FIRST_HALF_END else SECOND_HALF_END + 1
    next_dead = dead_balls.next_start_after(start, default=n_frames)
    return min(next_start, next_dead, half_end)


# =============================================================================
# Helpers
# =============================================================================
def all_valid_around(valid: np.ndarray, lag: int) -> np.ndarray:
    """True on frames where `valid` holds for every frame in [f - lag, f + lag]."""
    width = 2 * lag + 1
    return np.convolve(valid.astype(int), np.ones(width, dtype=int), mode="same") == width


def angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Absolute difference between two angles in degrees, in [0, 180]."""
    return np.abs((b - a + 180) % 360 - 180)
