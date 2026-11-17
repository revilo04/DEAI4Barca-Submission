"""
Hyperparameter and smoothing selection on held-out SkillCorner matches.

    select_config(data) -> (model params, smoothing seconds)

Three ways to get there, in this order (see the flags in constants.py):
    1. RUN_HYPERPARAMETER_TUNING: grid search over TUNING_PARAM_GRID x SMOOTHING_SECONDS.
       Each candidate is checkpointed, so an interrupted run resumes where it stopped.
    2. USE_SAVED_TUNING_RESULT: reuse best_config.json from an earlier tuning run.
    3. Otherwise: MODEL_PARAMS as they are (no early stopping); the smoothing window
       is still chosen from SMOOTHING_SECONDS on the held-out matches.
"""

import hashlib
import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

from evaluation import possession_metrics as pm   # the official scorer

from src_model.pipeline.model.constants import (
    BEST_CONFIG_PATH, EARLY_STOP_PATIENCE, EARLY_STOP_STEP, EARLY_STOP_TOL,
    EARLY_STOP_VALID_EVERY, FPS, MODEL_PARAMS, NO_ANSWER,
    RESUME_TUNING, ROLLING_SECONDS, RUN_HYPERPARAMETER_TUNING, SAMPLE_EVERY,
    SHIFT_SECONDS, SMOOTHING_SECONDS, TUNING_DIR, TUNING_PARAM_GRID,
    USE_SAVED_TUNING_RESULT,
)
from src_model.pipeline.model.match_input import MatchInput

logger = logging.getLogger(__name__)


@dataclass
class TrainingData:
    X_train: pd.DataFrame
    y_train: np.ndarray
    train: list[MatchInput]
    valid: list[MatchInput]
    valid_features: dict[int, pd.DataFrame]   # match_id -> features of every frame

    @property
    def train_ids(self) -> list[int]:
        return sorted(m.match_id for m in self.train)

    @property
    def valid_ids(self) -> list[int]:
        return sorted(m.match_id for m in self.valid)


# --- Saving ------------------------------------------------------------------

def atomic_joblib_save(value, destination: Path) -> None:
    """Write a checkpoint without leaving a partial file if the process stops."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    joblib.dump(value, temporary, compress=3)
    temporary.replace(destination)


def atomic_json_save(value: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)


def run_settings() -> dict:
    """Feature and training settings a saved result depends on."""
    return {
        "sample_every": SAMPLE_EVERY,
        "fps": FPS,
        "rolling_seconds": list(ROLLING_SECONDS),
        "shift_seconds": list(SHIFT_SECONDS),
        "early_stopping": {
            "method": "held_out_match_log_loss",
            "step": EARLY_STOP_STEP,
            "patience": EARLY_STOP_PATIENCE,
            "tol": EARLY_STOP_TOL,
            "validation_every": EARLY_STOP_VALID_EVERY,
        },
    }


def tuning_signature(params: dict, data: TrainingData) -> dict:
    """Settings that must match before a saved candidate can be reused."""
    return {
        "version": 3,
        "params": params,
        "train_ids": data.train_ids,
        "valid_ids": data.valid_ids,
        **run_settings(),
        "feature_columns": list(data.X_train.columns),
    }


# --- Labels and scores -------------------------------------------------------

def labels_from_probabilities(proba: np.ndarray, classes: np.ndarray, graded: np.ndarray,
                              smooth_seconds: float) -> np.ndarray:
    """Most likely label after centred probability smoothing; X off the graded frames."""
    if smooth_seconds > 0:
        window = max(1, int(round(smooth_seconds * FPS)))
        proba = pd.DataFrame(proba).rolling(window, center=True, min_periods=1).mean().to_numpy()
    labels = classes[proba.argmax(axis=1)].astype(object)
    labels[~graded] = NO_ANSWER
    return labels


def score(truth: np.ndarray, predicted: np.ndarray) -> dict:
    """Headline numbers from the official scorer."""
    result = pm.evaluate(pm.label_codes(truth), pm.label_codes(predicted))
    return {
        "spellF1": round(result["spell_f1_mean"], 4),
        "macroF1": round(result["macro_f1"], 4),
        "poss_pred": result["n_pred_spells"],
        "poss_true": result["n_gt_spells"],
        "dur_pred_s": round(result["mean_spell_duration_s"], 1),
        "dur_true_s": round(result["gt_mean_spell_duration_s"], 1),
    }


# --- Fitting -----------------------------------------------------------------

def validation_log_loss(model, data: TrainingData) -> float:
    """Mean log loss on sampled graded frames from whole held-out matches."""
    weighted_loss, total_frames = 0.0, 0
    for m in data.valid:
        rows = np.flatnonzero(m.graded)[::EARLY_STOP_VALID_EVERY]
        probabilities = model.predict_proba(data.valid_features[m.match_id].iloc[rows])
        loss = log_loss(m.labels[rows], probabilities, labels=model.classes_)
        weighted_loss += loss * len(rows)
        total_frames += len(rows)
    if total_frames == 0:
        raise ValueError("No graded validation frames available for early stopping")
    return weighted_loss / total_frames


def fit_with_early_stopping(params: dict, data: TrainingData):
    """
    Select boosting iterations on held-out matches, then refit at the best iteration.

    Returns (model, best iteration, best validation log loss, {match_id: validation probabilities}).
    """
    max_iter = params["max_iter"]
    model = HistGradientBoostingClassifier(
        **{**params, "max_iter": min(EARLY_STOP_STEP, max_iter), "warm_start": True}
    )
    best_loss, best_iteration, checks_without_improvement = np.inf, 0, 0
    for iteration in range(EARLY_STOP_STEP, max_iter + EARLY_STOP_STEP, EARLY_STOP_STEP):
        iteration = min(iteration, max_iter)
        model.set_params(max_iter=iteration)
        model.fit(data.X_train, data.y_train)
        current_loss = validation_log_loss(model, data)
        if current_loss < best_loss - EARLY_STOP_TOL:
            best_loss, best_iteration = current_loss, iteration
            checks_without_improvement = 0
        else:
            checks_without_improvement += 1
        if checks_without_improvement >= EARLY_STOP_PATIENCE or iteration == max_iter:
            break

    best_params = {**params, "max_iter": best_iteration}
    best_model = HistGradientBoostingClassifier(**best_params).fit(data.X_train, data.y_train)
    validation_probabilities = {
        match_id: best_model.predict_proba(features)
        for match_id, features in data.valid_features.items()
    }
    return best_model, best_iteration, best_loss, validation_probabilities


# --- Grid search -------------------------------------------------------------

def run_grid_search(data: TrainingData) -> dict[str, dict]:
    """Fit (or load from checkpoint) every combination in TUNING_PARAM_GRID."""
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    names = tuple(TUNING_PARAM_GRID)
    overrides_list = [
        dict(zip(names, values))
        for values in itertools.product(*(TUNING_PARAM_GRID[name] for name in names))
    ]

    records = {}
    for trial, overrides in enumerate(overrides_list, start=1):
        params = {**MODEL_PARAMS, **overrides}
        signature = tuning_signature(params, data)
        candidate_id = hashlib.sha256(
            json.dumps(signature, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        checkpoint = TUNING_DIR / f"trial_{candidate_id}.joblib"
        prefix = f"[{trial}/{len(overrides_list)}]"

        if RESUME_TUNING and checkpoint.exists():
            record = joblib.load(checkpoint)
            if record.get("signature") != signature:
                raise ValueError(f"Checkpoint settings do not match current run: {checkpoint}")
            logger.info("%s loaded checkpoint %s (best iteration %d)", prefix, candidate_id, record["best_iteration"])
        else:
            logger.info("%s fitting %s with match-level early stopping: %s", prefix, candidate_id, overrides)
            model, best_iteration, best_loss, valid_proba = fit_with_early_stopping(params, data)
            logger.info("%s selected %d iterations (validation log loss %.5f)", prefix, best_iteration, best_loss)
            record = {
                "signature": signature,
                "params": params,
                "best_iteration": best_iteration,
                "best_validation_log_loss": best_loss,
                "model": model,
                "valid_proba": valid_proba,
            }
            atomic_joblib_save(record, checkpoint)
        records[candidate_id] = record
    return records


def rank_candidates(records: dict[str, dict], data: TrainingData) -> pd.DataFrame:
    """Mean held-out scores of every candidate x smoothing window, best first."""
    logger.info("Scoring %d candidates x %d smoothing windows on %d held-out matches",
                len(records), len(SMOOTHING_SECONDS), len(data.valid))
    rows = []
    for candidate_id, record in records.items():
        for smooth_seconds in SMOOTHING_SECONDS:
            for m in data.valid:
                predicted = labels_from_probabilities(
                    record["valid_proba"][m.match_id], record["model"].classes_,
                    m.graded, smooth_seconds,
                )
                rows.append({
                    "candidate": candidate_id,
                    "smooth_s": smooth_seconds,
                    "match_id": m.match_id,
                    **score(m.labels, predicted),
                })
    scores = (
        pd.DataFrame(rows)
        .groupby(["candidate", "smooth_s"])[["spellF1", "macroF1", "poss_pred"]]
        .mean().reset_index()
    )
    return scores.sort_values(["spellF1", "macroF1"], ascending=False, ignore_index=True)


# --- Selection ---------------------------------------------------------------

def tune(data: TrainingData) -> tuple[dict, float]:
    """Option 1: grid search, pick the best candidate x smoothing, save best_config.json."""
    records = run_grid_search(data)
    ranking = rank_candidates(records, data)
    logger.info("Top candidates on held-out matches:\n%s", ranking.head(10).to_string(index=False))

    best = ranking.iloc[0]
    candidate_id = str(best["candidate"])
    record = records[candidate_id]
    params = {**record["params"], "max_iter": record["best_iteration"]}
    smooth_seconds = float(best["smooth_s"])

    atomic_json_save({
        "model_params": params,
        "smooth_seconds": smooth_seconds,
        "candidate_id": candidate_id,
        "train_ids": data.train_ids,
        "valid_ids": data.valid_ids,
        "run_settings": run_settings(),
        "best_iteration": record["best_iteration"],
        "selection_spell_f1": float(best["spellF1"]),
        "selection_macro_f1": float(best["macroF1"]),
        "selection_mean_predicted_possessions": float(best["poss_pred"]),
    }, BEST_CONFIG_PATH)
    logger.info("Saved best configuration to %s", BEST_CONFIG_PATH)
    return params, smooth_seconds


def load_saved_config(data: TrainingData) -> tuple[dict, float]:
    """Option 2: best_config.json from an earlier tuning run, checked against this run."""
    config = json.loads(BEST_CONFIG_PATH.read_text(encoding="utf-8"))
    if config["train_ids"] != data.train_ids or config["valid_ids"] != data.valid_ids:
        raise ValueError("Saved best configuration was selected with a different train/validation split")
    if config.get("run_settings") != run_settings():
        raise ValueError("Saved best configuration was selected with different feature/training settings")
    params = config["model_params"]
    if config.get("best_iteration", params["max_iter"]) != params["max_iter"]:
        raise ValueError("Saved best iteration does not match saved model parameters")
    logger.info("Loaded tuned settings from %s", BEST_CONFIG_PATH)
    return params, float(config["smooth_seconds"])


def fixed_config(data: TrainingData) -> tuple[dict, float]:
    """
    Option 3: MODEL_PARAMS as they are (max_iter included, no early stopping). The
    smoothing window is chosen as in the notebook: mean held-out spell F1 per window,
    rounded to 3 decimals, and the first (smallest) window with the highest value.
    """
    logger.info("Fitting MODEL_PARAMS (%d iterations) on the training matches", MODEL_PARAMS["max_iter"])
    model = HistGradientBoostingClassifier(**MODEL_PARAMS).fit(data.X_train, data.y_train)
    valid_proba = {
        match_id: model.predict_proba(features)
        for match_id, features in data.valid_features.items()
    }
    ranking = rank_candidates({"fixed": {"model": model, "valid_proba": valid_proba}}, data)
    by_window = (
        ranking.drop(columns="candidate").set_index("smooth_s").sort_index().round(3)
    )
    logger.info("Smoothing windows on held-out matches:\n%s", by_window.to_string())
    return dict(MODEL_PARAMS), float(by_window["spellF1"].idxmax())


def select_config(data: TrainingData) -> tuple[dict, float]:
    """Model parameters and smoothing window to train the final model with."""
    if RUN_HYPERPARAMETER_TUNING:
        logger.info("Tuning: grid search over %d combinations", len(list(itertools.product(*TUNING_PARAM_GRID.values()))))
        params, smooth_seconds = tune(data)
    elif USE_SAVED_TUNING_RESULT and BEST_CONFIG_PATH.exists():
        logger.info("Tuning off: using the saved configuration")
        params, smooth_seconds = load_saved_config(data)
    else:
        logger.info("Tuning off, no saved configuration: using MODEL_PARAMS")
        params, smooth_seconds = fixed_config(data)
    logger.info("Selected model parameters: %s", params)
    logger.info("Selected smoothing: %g seconds", smooth_seconds)
    return params, smooth_seconds
