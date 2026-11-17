import json
import pandas as pd
import numpy as np
import pickle

from dataclasses import dataclass

from pipeline.skillcorner.constants import *
from pipeline.skillcorner.utils import match_file

def read_meta(match_id: int) -> dict:
    # Read the match.json file, which has the home team, pitch size, and player
    # list
    raw = json.loads(match_file(match_id, "match.json").read_text())

    # Our tracking uses 0 for the home team and 1 for the away team (see
    # team_of_each_track() in starter/baseline.py)
    home_id = raw["home_team"]["id"]

    # One slot per player (1-based, like our column names), in the order the
    # file lists them
    players = pd.DataFrame({
        "player_id": [p["id"] for p in raw["players"]],
        "team": [0 if p["team_id"] == home_id else 1 for p in raw["players"]],
    })
    # This will be our 1_x, 1_y, 2_x, 2_y, ... column name
    players["slot"] = np.arange(1, len(players) + 1)
    assert len(players) <= N_SLOTS, f"match {match_id} has more than {N_SLOTS} players"

    return {
        "match_id": match_id,
        "fps": SOURCE_FPS,
        "pitch_length": raw["pitch_length"],
        "pitch_width": raw["pitch_width"],
        "home_id": home_id,
        "home": raw["home_team"]["short_name"],
        "away": raw["away_team"]["short_name"],
        "home_side_by_period": raw["home_team_side"],      # e.g. ["right_to_left", "left_to_right"]
        "periods": [(p["period"], p["start_frame"], p["end_frame"]) for p in raw["match_periods"]],
        "players": players,
    }

def read_tracking(match_id: int, meta: dict) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Tracking in our format, plus SkillCorner's per-frame possession (H / A / None).

    Frames where the broadcast shows no pitch have no players and no ball (all NaN).
    """
    length, width = meta["pitch_length"], meta["pitch_width"]
    slot_of = dict(zip(meta["players"]["player_id"], meta["players"]["slot"]))
    group_label = {"home team": HOME, "away team": AWAY}

    lines = match_file(match_id, "tracking_extrapolated.jsonl").read_text().splitlines()
    n = len(lines)
    period = np.zeros(n, dtype=int)
    ball = np.full((n, 2), np.nan)

    # SkillCorner matches have one slot per player (22 in total)
    # Our tracking has 46 slots
    # The extra slots are always NaN, but we keep them so the columns match our tracking
    player_x = np.full((n, N_SLOTS), np.nan)
    player_y = np.full((n, N_SLOTS), np.nan)
    possession = np.full(n, None, dtype=object)

    for i, line in enumerate(lines):
        frame = json.loads(line)
        period[i] = frame["period"] or 0
        possession[i] = group_label.get(frame["possession"]["group"])

        b = frame["ball_data"]
        if b["x"] is not None and (b["is_detected"] or not BALL_DETECTED_ONLY):
            # x and y in SkillCorner start from the middle of the pitch
            ball[i] = b["x"], b["y"]

        for p in frame["player_data"]:
            if p["is_detected"] or not PLAYERS_DETECTED_ONLY:
                col = slot_of[p["player_id"]] - 1
                player_x[i, col], player_y[i, col] = p["x"], p["y"]

    tracking = pd.DataFrame({
        "frame_id": np.arange(1, n + 1),            # 1-based, like ours: row i is frame i + 1
        "period_id": period,
        # ball[:, 0] is in [-length/2, length/2] Dividing by length scales
        # metres to a fraction, giving -0.5 to 0.5. Adding 0.5 then shifts the
        # origin from the centre to the corner, giving 0 to 1.
        "ball_x": ball[:, 0] / length + 0.5,
        # ball[:, 1] is in [-width/2, width/2]
        "ball_y": ball[:, 1] / width + 0.5,
    })
    # player_x contains the x positions of each player in each frame
    # player_y contains the y positions of each player in each frame
    speed = player_speed(player_x, player_y, SOURCE_FPS)
    slot_columns = {}
    for col in range(N_SLOTS):
        slot_columns[f"{col + 1}_x"] = player_x[:, col] / length + 0.5
        slot_columns[f"{col + 1}_y"] = player_y[:, col] / width + 0.5
        slot_columns[f"{col + 1}_s"] = speed[:, col]
    return pd.concat([tracking, pd.DataFrame(slot_columns)], axis=1), possession


def player_speed(x: np.ndarray, y: np.ndarray, fps: float) -> np.ndarray:
    """Speed (m/s) of each player from one frame to the next (positions in metres). NaN where either frame is missing."""
    speed = np.full_like(x, np.nan)
    speed[1:] = np.hypot(np.diff(x, axis=0), np.diff(y, axis=0)) * fps
    return speed


def slot_teams(meta: dict) -> np.ndarray:
    """Team (0 = home, 1 = away) of each of the N_SLOTS slots. Unused slots are 0 and always NaN."""
    team = np.zeros(N_SLOTS, dtype=np.int8)
    team[meta["players"]["slot"] - 1] = meta["players"]["team"]
    return team


def read_phase_team(match_id: int, meta: dict, n_frames: int) -> np.ndarray:
    """Team in possession (H / A) of the phase of play each frame belongs to; None outside phases."""
    phases = pd.read_csv(match_file(match_id, "phases_of_play.csv"))
    phase_team = np.full(n_frames, None, dtype=object)
    # A phase's last frame is the next phase's first; writing in order lets the later one win.
    for phase in phases.sort_values("frame_start").itertuples():
        team = HOME if phase.team_in_possession_id == meta["home_id"] else AWAY
        phase_team[phase.frame_start : phase.frame_end + 1] = team
    return phase_team


def build_labels(meta: dict, phase_team: np.ndarray, frame_possession: np.ndarray) -> pd.DataFrame:
    """One row per frame with the H / A / D / X label and the two signals it comes from."""
    n = len(phase_team)
    in_period = np.zeros(n, dtype=bool)
    for _, start, end in meta["periods"]:
        in_period[start : end + 1] = True

    in_phase = pd.notna(phase_team)
    label = np.full(n, NO_ANSWER, dtype=object)
    label[in_period] = DEAD                                         # in a period, ball not in play
    label[in_period & in_phase] = phase_team[in_period & in_phase]  # in play: the phase's team...
    on_ball = in_period & in_phase & pd.notna(frame_possession)
    label[on_ball] = frame_possession[on_ball]                      # ...unless a player is on the ball

    return pd.DataFrame({
        "frame": np.arange(n),
        "label": label,
        "phase_team": phase_team,
        "frame_possession": frame_possession,
    })

@dataclass
class SkillCornerMatch:
    meta: dict
    tracking: pd.DataFrame   # our tracking format, row i = frame i
    team: np.ndarray         # (N_SLOTS,) 0 = home, 1 = away
    labels: pd.DataFrame     # frame, label, phase_team, frame_possession


def load_native_match(match_id: int) -> SkillCornerMatch:
    """Convert one downloaded match at its native 10 fps (or read it from the cache)."""
    cache = CACHE_DIR / f"{match_id}.pkl"
    if cache.exists():
        cached = pickle.loads(cache.read_bytes())
        # Caches are shared with skillcorner_loader.ipynb, which also stores `tags`: ignore them
        return SkillCornerMatch(**{name: cached[name] for name in SkillCornerMatch.__dataclass_fields__})

    meta = read_meta(match_id)
    tracking, frame_possession = read_tracking(match_id, meta)
    phase_team = read_phase_team(match_id, meta, len(tracking))
    labels = build_labels(meta, phase_team, frame_possession)
    match = SkillCornerMatch(meta, tracking, slot_teams(meta), labels)

    # Cached as a plain dict of dataframes / arrays, so any notebook or script can read it
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(vars(match)))
    return match


def resample_match(match: SkillCornerMatch, fps: float) -> SkillCornerMatch:
    """Copy of `match` at `fps` frames per second (see the rules above)."""
    source_fps = match.meta["fps"]
    before, after, weight = resample_positions(len(match.tracking), source_fps, fps)

    old = match.tracking
    position_columns = [c for c in old.columns if c.endswith(("_x", "_y"))]
    positions = interpolate(old[position_columns].to_numpy(), before, after, weight)
    tracking = pd.DataFrame(positions, columns=position_columns)
    tracking.insert(0, "period_id", old["period_id"].to_numpy()[before])
    tracking.insert(0, "frame_id", np.arange(1, len(before) + 1))

    # Speeds from the resampled positions, in metres
    length, width = match.meta["pitch_length"], match.meta["pitch_width"]
    x = tracking[[f"{s}_x" for s in range(1, N_SLOTS + 1)]].to_numpy() * length
    y = tracking[[f"{s}_y" for s in range(1, N_SLOTS + 1)]].to_numpy() * width
    speed = pd.DataFrame(player_speed(x, y, fps), columns=[f"{s}_s" for s in range(1, N_SLOTS + 1)])
    tracking = pd.concat([tracking, speed], axis=1)[list(old.columns)]   # same column order as before

    labels = match.labels.iloc[before].reset_index(drop=True)
    labels["frame"] = np.arange(len(labels))

    meta = {**match.meta, "fps": fps, "periods": period_ranges(tracking["period_id"].to_numpy())}
    return SkillCornerMatch(meta, tracking, match.team, labels)


def resample_positions(n_frames: int, source_fps: float, fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Where each new frame falls in the original frames.

    Returns, per new frame: the original frame at or before it, the one after it, and how far
    between the two it lies (0 = exactly on `before`, close to 1 = almost at `after`).
    """
    n_new = int((n_frames - 1) * fps / source_fps) + 1
    position = np.arange(n_new) * source_fps / fps        # in original-frame units, e.g. 0, 0.4, 0.8, ...
    before = np.floor(position + 1e-9).astype(int)        # 1e-9 guards against 1.9999999 for a true 2.0
    after = np.minimum(before + 1, n_frames - 1)
    weight = np.clip(position - before, 0.0, 1.0)
    weight[weight < 1e-9] = 0.0
    return before, after, weight


def interpolate(values: np.ndarray, before: np.ndarray, after: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Linear interpolation of each column between rows `before` and `after`. NaN if either is NaN."""
    w = weight[:, None]
    blended = (1 - w) * values[before] + w * values[after]
    # A new frame exactly on an original frame takes its value, whatever the next frame holds
    return np.where(w == 0, values[before], blended)


def period_ranges(period_id: np.ndarray) -> list[tuple[int, int, int]]:
    """(period, first frame, last frame) of each period found in `period_id`."""
    ranges = []
    for period in sorted(set(period_id[period_id > 0].tolist())):
        frames = np.flatnonzero(period_id == period)
        ranges.append((period, int(frames[0]), int(frames[-1])))
    return ranges


def load_match(match_id: int, fps: float | None = TARGET_FPS) -> SkillCornerMatch:
    """One match in our format, resampled to `fps` (None keeps the native 10 fps)."""
    match = load_native_match(match_id)
    if fps is None or fps == match.meta["fps"]:
        return match
    return resample_match(match, fps)