import numpy as np
import pandas as pd

from dataclasses import dataclass

@dataclass
class MatchInput:
    name: str
    tracking: pd.DataFrame          # our format, positions normalized to [0, 1]
    team: np.ndarray                # (N_SLOTS,) 0 = home, 1 = away
    pitch_length: float
    pitch_width: float
    home_attacks_right: np.ndarray  # (n_frames,) bool
    graded: np.ndarray              # (n_frames,) frames to predict
    periods: list                   # [(period, first frame, last frame)]
    labels: np.ndarray | None       # (n_frames,) H / A / D / X, None if unknown

# ---------------------------------------------------------------------------
    match_id: int | None = None     # SkillCorner match id; None for BAR-ATH


from games import GAME, GAMES, half_video_blocks, require
from src_rules.pipeline.pauses.constants import KEEPER_SIDE
from src_rules.utils.data_utils import load_tracking
from starter.baseline import team_of_each_track

from src_model.pipeline.model.constants import NO_ANSWER
from src_model.pipeline.skillcorner.format_data import SkillCornerMatch


def skillcorner_input(match: SkillCornerMatch) -> MatchInput:
    """A SkillCorner match (already loaded and resampled to 25 fps), with its labels."""
    meta = match.meta
    period = match.tracking["period_id"].to_numpy()
    # home_side_by_period[p - 1] is "left_to_right" when home attacks towards x
    # = length in period p
    attacks_right = np.array([side == "left_to_right" for side in meta["home_side_by_period"]])
    labels = match.labels["label"].to_numpy()
    return MatchInput(
        name=f"{meta['home']} v {meta['away']}",
        tracking=match.tracking,
        team=match.team,
        pitch_length=float(meta["pitch_length"]),
        pitch_width=float(meta["pitch_width"]),
        home_attacks_right=np.where(period > 0, attacks_right[np.maximum(period, 1) - 1], False),
        graded=labels != NO_ANSWER,
        periods=meta["periods"],
        labels=labels,
        match_id=int(meta["match_id"]),
    )


def bar_ath_input() -> MatchInput:
    """Our match. Labels are unknown here; the public ground truth is scored separately."""
    # Loads the cached dataframe, building it from the gated tracking assets on first use.
    tracking = load_tracking()
    half = np.zeros(len(tracking), dtype=int)
    for h, first, last in half_video_blocks(GAME):
        half[first : last + 1] = h
    # KEEPER_SIDE gives the goal each team defends; home defending LEFT means attacking right
    defends_left = {h: KEEPER_SIDE[(h, 0)] == "LEFT" for h in (1, 2)}
    length, width = GAMES[GAME]["pitch"]
    return MatchInput(
        name="BAR-ATH",
        tracking=tracking,
        team=np.asarray(team_of_each_track(require(["atd_meta"])[0])),
        pitch_length=float(length),
        pitch_width=float(width),
        home_attacks_right=np.array([defends_left.get(h, False) for h in half]),
        graded=half > 0,
        periods=half_video_blocks(GAME),
        labels=None,
    )
