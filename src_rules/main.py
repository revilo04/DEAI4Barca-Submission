import sys

import numpy as np
import pandas as pd

from constants import ROOT, SECOND_HALF_KICKOFF, SECOND_HALF_START
sys.path.append(str(ROOT))

import games
from utils.data_utils import load_tracking
from pipeline.pauses.fill_pauses import fill_pauses, label_pauses
from pipeline.fill_possession import UNKNOWN, fill_possession
from starter.baseline import team_of_each_track


def mark_pre_kickoff_dead_ball(submission: pd.DataFrame) -> pd.DataFrame:
    """
    D from the start of the graded second half up to its kickoff.

    Citing FAQ.md: ### Which frames are actually graded?
        - One trap worth knowing: the second half is graded from frame
            110,808 but does not kick off until 113,742. Those 2,934 frames
            are graded and all of them are `D`.
    """
    before_kickoff = submission["frame"].between(
        SECOND_HALF_START, SECOND_HALF_KICKOFF, inclusive="left"
    )
    return submission.assign(label=submission["label"].mask(before_kickoff, "D"))


def forward_fill_unknown(submission: pd.DataFrame) -> pd.DataFrame:
    """Replace each unknown ("?") label with the last known label before it, or X if none."""
    labels = submission["label"].replace(UNKNOWN, np.nan).ffill().fillna("X")
    return submission.assign(label=labels)


def build_submission(tracking: pd.DataFrame, team: np.ndarray) -> pd.DataFrame:
    """
    One (frame, label) row per tracking row, labelled in priority order (later
    steps overwrite earlier ones):

        1. X everywhere.
        2. Possession (H / A / ?) on in-play frames.
        3. Pauses (D / X / H / A) on frames where the ball is missing.
        4. D before the second-half kickoff.
        5. Remaining "?" take the previous label (X if there is none).
    """
    submission = pd.DataFrame({"frame": range(len(tracking)), "label": "X"})
    pauses = label_pauses(tracking, team)
    submission = fill_possession(submission, tracking, team, pauses)
    submission = fill_pauses(submission, pauses)
    submission = mark_pre_kickoff_dead_ball(submission)
    return forward_fill_unknown(submission)


def main():
    meta = games.require(["atd_meta"])[0]
    tracking = load_tracking()

    team = team_of_each_track(meta)
    submission = build_submission(tracking, team)
    submission.to_csv("submission.csv", index=False)
    print(submission["label"].value_counts())


if __name__ == "__main__":
    main()
