"""
Label the frames where the ball is missing during play (pauses) as D / X / H / A.

Pipeline:
    label_pauses: tracking -> Match arrays -> pauses -> per-pause features -> rules -> labels
    fill_pauses:  write each pause's label onto its frames of the submission
"""

import numpy as np
import pandas as pd

from pipeline.pauses.constants import *
from pipeline.pauses.utils import *
from pipeline.pauses.rules import RULES

from games import half_video_blocks

# =============================================================================
# Entry point
# =============================================================================
def label_pauses(tracking: pd.DataFrame, team: np.ndarray) -> pd.DataFrame:
    """
    Detect every pause and label it D / X / H / A.

    A pause is a run of graded frames where the ball is not tracked.

    Args:
        tracking: tactical tracking dataframe, row i = video frame i.
        team:     team (0 = home, 1 = away) of each tracking slot, from
                  starter.baseline.team_of_each_track.

    Returns one row per pause with (at least) `start`, `end` (inclusive video
    frames) and `label`.
    """
    match = build_match(tracking, team)
    pauses = detect_pauses(match)
    keepers = identify_keepers(match)
    pauses = add_pause_features(pauses, match, keepers)
    pauses["label"] = pauses.apply(label_pause, axis=1)
    return pauses


def fill_pauses(submission: pd.DataFrame, pauses: pd.DataFrame) -> pd.DataFrame:
    """
    Copy of `submission` with each pause's label written on frames [start, end].

    Frames outside pauses keep their label. Relies on row i of the submission
    being video frame i.
    """
    labels = submission["label"].to_numpy(copy=True)
    for p in pauses.itertuples():
        labels[p.start : p.end + 1] = p.label
    return submission.assign(label=labels)


# =============================================================================
# Steps
# =============================================================================
def build_match(tracking: pd.DataFrame, team: np.ndarray) -> Match:
    """
    Turn the tracking dataframe into per-frame numpy arrays (see Match).

    Tracking coordinates are normalized to [0, 1]; they are converted to metres
    here once so every feature and rule can work in metres.
    """
    def columns(suffix: str) -> np.ndarray:
        cols = [f"{slot}_{suffix}" for slot in range(1, N_SLOTS + 1)]
        return tracking[cols].to_numpy(dtype=float, na_value=np.nan)

    # ball_x and ball_y are normalized to [0, 1] in the tracking; convert to
    # metres to get interpretable distances in the features and rules
    ball_x = tracking["ball_x"].to_numpy(dtype=float, na_value=np.nan) * PITCH_LENGTH
    ball_y = tracking["ball_y"].to_numpy(dtype=float, na_value=np.nan) * PITCH_WIDTH

    # Half of each frame: 1 / 2 inside the graded halves, 0 everywhere else
    half = np.zeros(len(tracking), dtype=int)
    # half_video_blocks = [(1, start1, end1), (2, start2, end2)] for the two
    # halves of the video
    for h, first, last in half_video_blocks(GAME):
        half[first : last + 1] = h

    return Match(
        ball_x=ball_x,
        ball_y=ball_y,
        ball_seen=~np.isnan(ball_x),
        player_x=columns("x") * PITCH_LENGTH,
        player_y=columns("y") * PITCH_WIDTH,
        speed=columns("s"),
        team=np.asarray(team),
        half=half,
    )


def detect_pauses(match: Match) -> pd.DataFrame:
    """
    One row per pause: a run of consecutive graded frames without the ball.

    Columns: id, start, end (both inclusive video frames), half, seconds.
    """
    no_ball = (match.half > 0) & ~match.ball_seen
    starts, ends = get_runs(no_ball)
    return pd.DataFrame({
        "id": np.arange(len(starts)),
        "start": starts,
        "end": ends,
        "half": match.half[starts],
        "seconds": (ends - starts + 1) / FPS,
    })


def identify_keepers(match: Match) -> dict[tuple[int, int], int]:
    """
    Find the goalkeeper's track slot for each team in each half.

    A team's keeper is the slot that is seen inside a penalty box (either one,
    since teams switch goals at halftime) on the largest share of the half's
    frames. The share is over every frame of the half, not only the frames
    where the slot is tracked, so a short tracklet cannot outscore the keeper. This is done per half because a slot is a tracklet, not a person,
    so the keeper's slot can change between halves.

    A slot is only accepted if that share is at least KEEPER_MIN_BOX_SHARE, so a
    team whose keeper is not tracked gets no entry instead of a random outfield
    player.

    Returns {(half, team): slot}, where slot is a column index into
    match.player_x / match.player_y. Missing keys mean "no keeper found".
    """
    # (n_frames, n_slots) bool: is each player inside either penalty box?
    # NaN positions compare as False, so untracked frames are never "in the box".
    in_left_box  = match.player_x <= PENALTY_BOX_DEPTH
    in_right_box = match.player_x >= (PITCH_LENGTH - PENALTY_BOX_DEPTH)
    within_box_width = np.abs(match.player_y - PITCH_WIDTH / 2) <= PENALTY_BOX_WIDTH / 2
    # Acá nos quedamos con todos los jugadores que estuvieron alguna vez dentro
    # del área de penalti, sin importar si es el área del equipo local o
    # visitante
    in_penalty_box = (in_left_box | in_right_box) & within_box_width

    # (n_frames, n_slots) bool: is each slot tracked on that frame?
    visible = ~np.isnan(match.player_x)

    keepers = {}
    for half in (1, 2):
        in_half = match.half == half
        for team in (0, 1):
            # Pick the team's slot that is seen in a box on the largest share
            # of the half's frames
            best_slot, best_share = None, 0.0
            for slot in np.flatnonzero(match.team == team):
                seen = in_half & visible[:, slot]
                if not seen.any():
                    continue  # slot never tracked in this half
                share = in_penalty_box[seen, slot].sum() / in_half.sum()
                if share > best_share:
                    best_slot, best_share = slot, share

            if best_share >= KEEPER_MIN_BOX_SHARE:
                keepers[(half, team)] = int(best_slot)
    return keepers


def add_pause_features(
    pauses: pd.DataFrame, match: Match, keepers: dict[tuple[int, int], int]
) -> pd.DataFrame:
    """
    Add the columns the rules look at, one row per pause.

    Each pause is described by what the ball and players do around it, using
    the last frame where the ball was seen before the pause ("before") and the
    first frame where it is seen again ("after"):

        ball_x_after, ball_y_after : where the ball reappears (m)
        keeper_dist_before         : ball-to-nearest-keeper distance before (m)
        keeper_dist_after          : ball-to-nearest-keeper distance after (m)
        players_speed_mean         : mean player speed during the pause (m/s)
        touchline_dist_before      : ball distance to the nearest touchline before (m)
        ball_displacement          : distance between the before and after ball positions (m)

    Features that cannot be computed (no keeper found, no ball frame on one
    side of the pause, ...) are NaN, and every rule treats NaN as "no match".
    """
    # Sorted frames where the ball is visible, to find each pause's neighbours
    seen_frames = np.flatnonzero(match.ball_seen)

    rows = []
    for p in pauses.itertuples():
        half = int(p.half)
        before, after = ball_frames_around(p.start, seen_frames)
        rows.append({
            "ball_x_after": match.ball_x[after] if after is not None else np.nan,
            "ball_y_after": match.ball_y[after] if after is not None else np.nan,
            "keeper_dist_before": keeper_distance(before, half, match, keepers),
            "keeper_dist_after": keeper_distance(after, half, match, keepers),
            "players_speed_mean": mean_player_speed(p.start, p.end, match),
            "touchline_dist_before": distance_to_touchline(before, match),
            "ball_displacement": ball_displacement(before, after, match),
        })
    return pd.concat([pauses, pd.DataFrame(rows, index=pauses.index)], axis=1)


def label_pause(pause: pd.Series) -> str:
    """
    Final submission label for one pause: D, X, H or A.

    The rules give a category (Dead Ball, Error or Keeper). Keeper pauses are
    then resolved to the team whose keeper restarts play (H = home, A = away).
    If that team cannot be determined, the pause is labelled X.
    """
    category = classify_pause(pause)
    if category != KEEPER:
        return category

    team = keeper_team(pause)
    if team == 0:
        return HOME_KEEPER
    if team == 1:
        return AWAY_KEEPER
    return ERROR  # keeper pause, but no goal close enough to tell whose


def classify_pause(p: pd.Series) -> str:
    """
    Category of a pause (DEAD_BALL, ERROR or KEEPER): the first rule in RULES
    that matches.
    """
    for rule, category in RULES:
        if rule(p):
            return category
    return DEAD_BALL  # R6: nothing suggests play went on, so assume a stoppage


def keeper_team(pause: pd.Series) -> int | None:
    """
    Team (0 = home, 1 = away) whose keeper restarts play after a Keeper pause.

    Picks the team whose defended goal line (from KEEPER_SIDE) is closest to
    where the ball reappears. None if the ball never reappears, or if it
    reappears more than KEEPER_MAX_GOAL_DIST_M from both goal lines.
    """
    x = pause["ball_x_after"]
    if pd.isna(x):
        return None

    # Distance from the ball to the goal line each team defends in this half
    half = int(pause["half"])
    goal_dist = {
        team: x if KEEPER_SIDE[(half, team)] == "LEFT" else PITCH_LENGTH - x
        for team in (0, 1)
    }
    team = min(goal_dist, key=goal_dist.get)
    return team if goal_dist[team] <= KEEPER_MAX_GOAL_DIST_M else None
