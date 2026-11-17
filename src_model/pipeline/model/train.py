import logging

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src_model.pipeline.skillcorner.format_data import SkillCornerMatch

from src_model.pipeline.model.constants import FINAL_MODEL_PATH, RESUME_SAVED_MODELS
from src_model.pipeline.model.dataset import (
    split, training_rows, validation_features
)
from src_model.pipeline.model.match_input import MatchInput, skillcorner_input
from src_model.pipeline.model.tuning import (
    TrainingData, atomic_joblib_save, run_settings, select_config
)

logger = logging.getLogger(__name__)


def fit_final_model(inputs: list[MatchInput], params: dict, smooth_seconds: float,
                    feature_columns: list[str]) -> HistGradientBoostingClassifier:
    """
    The selected configuration retrained on every SkillCorner match.

    Reloaded from FINAL_MODEL_PATH when a model with the same settings, matches
    and features was saved before (and RESUME_SAVED_MODELS is on).
    """
    match_ids = sorted(m.match_id for m in inputs)
    signature = {
        "version": 3,                    # 3: training rows in match-id order
        "model_params": params,
        "smooth_seconds": smooth_seconds,
        "train_match_ids": match_ids,
        **run_settings(),
        "feature_columns": feature_columns,
    }

    if RESUME_SAVED_MODELS and FINAL_MODEL_PATH.exists():
        saved = joblib.load(FINAL_MODEL_PATH)
        if saved.get("signature") == signature:
            logger.info("Loaded final all-match model from %s", FINAL_MODEL_PATH)
            return saved["model"]

    logger.info("Fitting final model on %d matches", len(inputs))
    X_all, y_all = training_rows(inputs)
    logger.info("%s training rows x %d features", f"{len(X_all):,}", X_all.shape[1])
    if list(X_all.columns) != feature_columns:
        raise ValueError("Features of the full training set differ from the ones used for tuning")
    model = HistGradientBoostingClassifier(**params).fit(X_all, y_all)
    atomic_joblib_save({
        "signature": signature,
        "model": model,
        "model_params": params,
        "smooth_seconds": smooth_seconds,
        "train_match_ids": match_ids,
    }, FINAL_MODEL_PATH)
    logger.info("Saved final all-match model to %s", FINAL_MODEL_PATH)
    return model


def train_model(matches: list[SkillCornerMatch]) -> tuple[HistGradientBoostingClassifier, float]:
    # Sorted by match id, like the notebook: the row order changes which rows the
    # model samples to choose its bin edges, and so the final model
    inputs = sorted((skillcorner_input(m) for m in matches), key=lambda m: m.match_id)
    train, valid = split(inputs)
    logger.info("Split: %d training matches, %d held-out matches", len(train), len(valid))
    logger.info("Train ids: %s", sorted(m.match_id for m in train))
    logger.info("Held-out ids: %s", sorted(m.match_id for m in valid))

    logger.info("Building training rows")
    X_train, y_train = training_rows(train)
    logger.info("%s training rows x %d features", f"{len(X_train):,}", X_train.shape[1])
    logger.info("Label shares: %s",
                pd.Series(y_train).value_counts(normalize=True).round(3).to_dict())

    logger.info("Building held-out features")
    data = TrainingData(
        X_train, y_train, train, valid, validation_features(valid)
    )
    params, smooth_s = select_config(data)

    # Free the tuning data before building the (larger) all-match training set
    feature_columns = list(X_train.columns)
    del data, X_train, y_train

    model = fit_final_model(inputs, params, smooth_s, feature_columns)
    logger.info("Training done")
    return model, smooth_s
