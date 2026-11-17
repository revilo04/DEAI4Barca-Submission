import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src_model.pipeline.model.features import build_features
from src_model.pipeline.model.match_input import MatchInput
from src_model.pipeline.model.tuning import labels_from_probabilities


def predict(model: HistGradientBoostingClassifier, smooth_seconds: float,
            m: MatchInput) -> pd.DataFrame:
    """
    One (frame, label) row per tracking frame of `m`: the most likely label after
    smoothing the model's probabilities over `smooth_seconds`, X off the graded frames.
    """
    proba = model.predict_proba(build_features(m))
    labels = labels_from_probabilities(proba, model.classes_, m.graded, smooth_seconds)
    return pd.DataFrame({"frame": np.arange(len(labels)), "label": labels})
