import pandas as pd

from pipeline.pauses.constants import *

# =============================================================================
# Rules: each takes one pause row (with features) and returns True if it
# matches. Comparisons with NaN are False, so a missing feature never matches.
# =============================================================================
def is_long_pause(p: pd.Series) -> bool:
    """
    R0. The ball is gone for more than LONG_PAUSE_S.

    The tracker does not lose the ball for that long during live play, so this
    is a real stoppage (injury, substitution, VAR, goal celebration, ...).
    """
    return p["seconds"] > LONG_PAUSE_S


def is_short_pause(p: pd.Series) -> bool:
    """
    R1. The ball is gone for less than SHORT_PAUSE_S.

    No restart (throw-in, goal kick, free kick) is taken that fast, so this is
    the tracker briefly losing the ball while play continues.
    """
    return p["seconds"] < SHORT_PAUSE_S


def restarts_in_goal_area(p: pd.Series) -> bool:
    """
    R2. The ball reappears inside one of the two goal areas (6-yard boxes).

    That is where goal kicks are taken, so play was stopped.
    """
    x, y = p["ball_x_after"], p["ball_y_after"]
    in_goal_area_depth = x <= GOAL_AREA_DEPTH or PITCH_LENGTH - x <= GOAL_AREA_DEPTH
    in_goal_area_width = abs(y - PITCH_WIDTH / 2) <= GOAL_AREA_HALF_WIDTH
    return in_goal_area_depth and in_goal_area_width


def keeper_has_ball(p: pd.Series) -> bool:
    """
    R3. A goalkeeper holds the ball during the pause.

    Typical case: the keeper catches the ball (the tracker loses it in their
    hands) and play restarts with the ball at the keeper's feet. The clauses
    below are empirically tuned combinations of:

        kb = keeper_dist_before   ka = keeper_dist_after       (m)
        sp = players_speed_mean   (m/s)
        db = ball_displacement    (m)
    """
    kb, ka = p["keeper_dist_before"], p["keeper_dist_after"]
    sp, db = p["players_speed_mean"], p["ball_displacement"]
    return (
        # Ball near the keeper before, at their feet after, players still moving
        (kb <= 7.0 and ka <= 0.9 and sp > 1.25)
        # Ball far from the keeper before (a shot or cross), at their feet after
        or (ka <= 1.0 and kb >= 15.0 and db <= 22.0)
        # Ball at mid distance before, at their feet after, barely moved
        or (ka <= 1.0 and 5.0 <= kb <= 12.0 and db <= 5.0)
        # Ball very close to the keeper on both sides, players moving
        or (ka <= 1.5 and kb <= 4.0 and sp >= 1.8)
        # Keeper restarts a few metres from the ball after a long ball, fast play
        or (5.0 <= ka <= 7.0 and kb >= 15.0 and sp >= 2.3)
    )


def is_throw_in(p: pd.Series) -> bool:
    """
    R4. The ball goes out near a touchline and reappears close to that spot.

    That is a throw-in, so play was stopped.
    """
    return (
        p["touchline_dist_before"] <= THROW_IN_TOUCHLINE_M
        and p["ball_displacement"] <= THROW_IN_DISPLACEMENT_M
    )


def players_kept_moving(p: pd.Series) -> bool:
    """
    R5. Players keep running while the ball is missing.

    During a stoppage players slow down; if they keep moving, play went on and
    the tracker just lost the ball. Either they are very fast overall, or
    reasonably fast while the ball was away from the keeper and the touchlines.
    """
    sp = p["players_speed_mean"]
    away_from_keeper_and_line = (
        p["keeper_dist_before"] >= 10.0 and p["touchline_dist_before"] >= 6.0
    )
    return sp >= PLAY_ON_SPEED_MS or (sp >= 1.5 and away_from_keeper_and_line)


# Rules in priority order; the first one that matches decides the category.
RULES = [
    (is_long_pause, DEAD_BALL),          # R0
    (is_short_pause, ERROR),             # R1
    (restarts_in_goal_area, DEAD_BALL),  # R2 goal kick
    (keeper_has_ball, KEEPER),           # R3
    (is_throw_in, DEAD_BALL),            # R4
    (players_kept_moving, ERROR),        # R5 tracker lost the ball, play went on
]