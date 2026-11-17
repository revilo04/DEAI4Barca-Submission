import logging
import sys

import pandas as pd

from constants import ROOT, SECOND_HALF_KICKOFF, SECOND_HALF_START
sys.path.append(str(ROOT))

from pipeline.skillcorner.constants import MATCH_IDS, TARGET_FPS
from pipeline.skillcorner.get_data import get_skillcorner_data, list_match_ids
from pipeline.skillcorner.format_data import SkillCornerMatch, load_match

from src_model.pipeline.model.constants import DEAD
from src_model.pipeline.model.match_input import bar_ath_input
from src_model.pipeline.model.predict import predict
from src_model.pipeline.model.train import train_model

logger = logging.getLogger(__name__)

def load_skillcorner_matches() -> list[SkillCornerMatch]:
    """
    Steps 1 + 2: download the SkillCorner matches (skips files already on disk)
    and convert each one to our tracking format at TARGET_FPS.
    """
    get_skillcorner_data()
    match_ids = MATCH_IDS if MATCH_IDS is not None else list_match_ids()
    matches = [load_match(match_id, fps=TARGET_FPS) for match_id in match_ids]
    # Positions are normalised to [0, 1]: float32 is precise to well under a millimetre and
    # halves the memory of keeping all matches loaded at once.
    for m in matches:
        float_cols = m.tracking.select_dtypes("float64").columns
        m.tracking[float_cols] = m.tracking[float_cols].astype("float32")
    return matches


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
    return submission.assign(label=submission["label"].mask(before_kickoff, DEAD))


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # 1 + 2. SkillCorner data, in our format
    matches = load_skillcorner_matches()
    logger.info(f"Loaded {len(matches)} SkillCorner matches")

    # 3. Train (tuning, then the final model on every match)
    logger.info("Training model...")
    model, smooth_seconds = train_model(matches)
    logger.info(f"Trained model with smooth seconds: {smooth_seconds}")

    # 4. Run on our tracking data
    submission = predict(model, smooth_seconds, bar_ath_input())
    submission = mark_pre_kickoff_dead_ball(submission)
    submission.to_csv("submission.csv", index=False)
    logger.info(f"Label counts: {submission['label'].value_counts()}")


if __name__ == "__main__":
    main()
