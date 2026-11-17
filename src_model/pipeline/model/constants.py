from src_model.constants import ROOT

FPS = 25                     # everything runs at our frame rate (SkillCorner is resampled to it)
N_SLOTS = 46
VALID_COUNT = 4              # held-out SkillCorner matches
SAMPLE_EVERY = 5             # train on every 5th graded frame (0.2 s apart; neighbours are near copies)
SEED = 0

# Toggle tuning off after a completed tuning run to reuse its saved best configuration.
RUN_HYPERPARAMETER_TUNING = False
USE_SAVED_TUNING_RESULT = True # if tuning is off, load prior best settings when available
RESUME_TUNING = True         # load completed candidate checkpoints instead of fitting them again
RESUME_SAVED_MODELS = True   # reuse compatible validation/final models after a kernel restart
TUNING_DIR = ROOT / "data" / "derived" / "possession_model_tuning"
BEST_CONFIG_PATH = TUNING_DIR / "best_config.json"
VALIDATION_MODEL_PATH = TUNING_DIR / "best_validation_model.joblib"
FINAL_MODEL_PATH = TUNING_DIR / "final_model.joblib"

ROLLING_SECONDS = (1, 4)     # centred rolling means over these windows
SHIFT_SECONDS = (1,)         # values this long before and after the frame
SMOOTHING_SECONDS = (0, 1, 2, 4, 8)   # candidate windows (same as the notebook), chosen on held-out SkillCorner matches
EARLY_STOP_STEP = 10         # add this many boosting iterations between checks
EARLY_STOP_PATIENCE = 5      # stop after this many checks without validation-loss improvement
EARLY_STOP_TOL = 1e-4        # minimum log-loss improvement to reset patience
EARLY_STOP_VALID_EVERY = 5   # subsample validation frames for faster stopping checks

TUNING_PARAM_GRID = {
    "learning_rate": (0.1, 0.05, 0.01),
    "max_leaf_nodes": (15, 31),
    "min_samples_leaf": (100, 200),
    "l2_regularization": (1.0, 10.0),
}

MODEL_PARAMS = dict(
    max_iter=300,
    learning_rate=0.1,
    max_leaf_nodes=31,
    min_samples_leaf=200,     # large leaves: frames are highly correlated, so be conservative
    l2_regularization=1.0,
    early_stopping=False,     # stopping is performed on held-out matches below
    random_state=SEED,
)

HOME, AWAY, DEAD, NO_ANSWER = "H", "A", "D", "X"