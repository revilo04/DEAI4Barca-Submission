import logging

import numpy as np
import pandas as pd

from src_model.pipeline.model.constants import SEED, VALID_COUNT, SAMPLE_EVERY
from src_model.pipeline.model.features import build_features, swap_teams
from src_model.pipeline.model.match_input import MatchInput

logger = logging.getLogger(__name__)


def split(inputs: list[MatchInput]) -> tuple[list[MatchInput], list[MatchInput]]:
    """
    Random (SEED) split into training and held-out matches; VALID_COUNT matches
    are held out. Matches are sorted by id first, so the split does not depend on
    the order they were loaded in and matches the notebook's split.
    """
    if len(inputs) <= VALID_COUNT:
        raise ValueError(
            f"Need more than {VALID_COUNT} matches to hold {VALID_COUNT} out, got {len(inputs)}"
        )
    inputs = sorted(inputs, key=lambda m: m.match_id)
    order = np.random.default_rng(SEED).permutation(len(inputs))
    valid = [inputs[i] for i in sorted(order[:VALID_COUNT])]
    train = [inputs[i] for i in sorted(order[VALID_COUNT:])]
    return train, valid


def training_rows(inputs: list[MatchInput]) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Every SAMPLE_EVERY-th graded frame of each match, plus its team-swapped copy.
    Returns (features, labels) for all matches in `inputs`.
    """
    features, labels = [], []
    for i, m in enumerate(inputs, start=1):
        rows = np.flatnonzero(m.graded)[::SAMPLE_EVERY]
        logger.info("[%d/%d] %s (%d): %s sampled frames", i, len(inputs), m.name, m.match_id, f"{len(rows):,}")
        x, y = build_features(m).iloc[rows].reset_index(drop=True), m.labels[rows]
        x_swapped, y_swapped = swap_teams(x, y)
        features += [x, x_swapped]
        labels += [y, y_swapped]
    return pd.concat(features, ignore_index=True), np.concatenate(labels)


def validation_features(inputs: list[MatchInput]) -> dict[int, pd.DataFrame]:
    """Features of every frame of each held-out match, keyed by match id."""
    features = {}
    for i, m in enumerate(inputs, start=1):
        logger.info("[%d/%d] %s (%d): features of all %s frames", i, len(inputs), m.name, m.match_id, f"{len(m.tracking):,}")
        features[m.match_id] = build_features(m)
    return features
