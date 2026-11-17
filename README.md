# Football Tracking Challenge – Possession Detection (`DEAI4Barca`)

## 1. Project Overview

This repository contains our end-to-end machine learning pipeline for reconstructing frame-level ball possession from football player and ball tracking data (`BAR-ATH` match).

The objective is to assign a state label to every video frame (`0..199,318` at **25 FPS**):
- **`H`** – Home team (FC Barcelona) has possession.
- **`A`** – Away team (Athletic Club) has possession.
- **`D`** – Dead ball / play is paused (throw-in, free kick, corner, goal kick, kickoff, foul, offside, etc.).
- **`X`** – Frame is outside the graded playing periods (pre-match, half-time, post-match).

### Core Challenges & Our Solution
In the target `BAR-ATH` tactical tracking dataset, ground-truth labels are held out for evaluation, and the ball is only detected in **~56.9%** of frames (with non-extrapolated player dropouts and frequent track ID reassignments).

To overcome this without relying on brittle hand-crafted heuristics, we built a **supervised transfer-learning pipeline**:
1. **External Training Data (SkillCorner Open Data):** Automatically downloads and harmonizes **20 full broadcast-tracked matches** from the SkillCorner Open Data repository into our normalized 25 FPS, 46-slot tracking format.
2. **Rich Spatio-Temporal Feature Engineering (265 Features):** Extracts ball kinematics, gap-bridging features across ball-visibility dropouts, team collective shape/pressure, inferred touch dynamics, and multi-scale rolling/lag/lead temporal context.
3. **Symmetric Data Augmentation (`swap_teams`):** Doubles the training set to **945,714 rows** while guaranteeing zero home/away bias by mirroring team-relative features and flipping labels (`H` <-> `A`).
4. **Gradient Boosted Decision Trees (`HistGradientBoostingClassifier`):** Natively handles `NaN` values during ball dropouts and learns complex interactions between player collective movement and ball kinematics.
5. **Temporal Probability Smoothing & Domain Post-Processing:** Smooths predicted class probabilities over a **4.0-second** centred window (selected on held-out validation matches to maximize **Spell F1**) and enforces the graded second-half pre-kickoff dead-ball window.

---

## 2. Repository Structure

```text
DEAI4Barca-Submission/
├── main.py                                    # Root entry point (runs the full pipeline)
├── DEAI4BARCA.csv                             # Final generated submission file (official name)
├── submission.csv                             # Final generated submission file (default output)
├── DEAI4Barca.pdf                             # Technical report
├── games.py                                   # Match metadata, frame block definitions, and asset paths
├── pyproject.toml / uv.lock                   # Dependencies and reproducible environment lockfile
├── data/
│   ├── README.md                              # Dataset placement documentation
│   └── derived/possession_model_tuning/
│       ├── best_config.json                   # Saved hyperparameters and optimal smoothing window (4.0s)
│       └── final_model.joblib                 # Pre-trained 20-match HistGradientBoostingClassifier
├── src_model/
│   ├── main.py                                # Core pipeline orchestration
│   ├── constants.py                           # Match frame boundaries and root path definitions
│   └── pipeline/
│       ├── skillcorner/
│       │   ├── constants.py                   # SkillCorner resampling and filtering constants
│       │   ├── get_data.py                    # Automatic downloader for 20 SkillCorner matches
│       │   ├── format_data.py                 # Coordinate normalization, 10->25 FPS resampling, label builder
│       │   └── utils.py                       # File path helpers
│       └── model/
│           ├── constants.py                   # Model hyperparameters, sampling rate, and window constants
│           ├── match_input.py                 # Unified MatchInput wrapper for SkillCorner and BAR-ATH
│           ├── features.py                    # 265 spatio-temporal features & swap_teams augmentation
│           ├── dataset.py                     # Match-level train/validation split and row subsampling
│           ├── tuning.py                      # Hyperparameter & smoothing window selection via Spell F1
│           ├── train.py                       # Model training, checkpoint caching, and reloading
│           └── predict.py                     # Frame-level probability prediction and temporal smoothing
├── starter/                                   # Provided baseline utilities and EPTS/XML loaders
├── evaluation/                                # Official validation and scoring scripts
└── info/                                      # Challenge documentation (CHALLENGE.md, DATA.md, FAQ.md)
```

---

## 3. Quick Start & Reproduction

### Prerequisites
- Python **3.12+**
- Required BAR-ATH match tracking files placed in `data/tactical/` (`BAR-ATH_tactical_atd.txt`, `BAR-ATH_tactical_atd.xml`).

### Installation
Using `uv` (recommended) or `pip`:
```bash
# Option A: Using uv
uv sync

# Option B: Using standard pip / venv
pip install joblib "kloppy==3.18.0" numpy pandas scikit-learn
```

### Running the Pipeline
Execute `main.py` from the repository root:
```bash
python main.py
```

### What `python main.py` Does:
1. **SkillCorner Data Preparation:** Checks `data/external/skillcorner/` for the 20 training matches (downloading any missing files automatically) and resamples them from 10 FPS to 25 FPS (caching converted matches in `data/derived/skillcorner/`).
2. **Pre-Trained Model Loading:** Because `data/derived/possession_model_tuning/best_config.json` and `final_model.joblib` are included in the repository (`RESUME_SAVED_MODELS = True` and `USE_SAVED_TUNING_RESULT = True`), the pipeline verifies the configuration signature and **loads the pre-trained model directly** in ~18 seconds without re-running the full training loop.
   - *Note:* If `final_model.joblib` is deleted, `python main.py` will automatically re-train the model from scratch on all 20 SkillCorner matches and save the new artifact.
3. **BAR-ATH Prediction & Post-Processing:** Computes the 265 features on `BAR-ATH`, applies 4.0-second probability smoothing, enforces the second-half pre-kickoff dead-ball rule (`frames 110,808..113,741`), and writes `submission.csv`.

### Validating & Evaluating the Submission
```bash
# 1. Verify submission format and frame coverage
python evaluation/validate_submission.py DEAI4BARCA.csv

# 2. Score against public ground truth
cp DEAI4BARCA.csv evaluation/results/our_submission.csv
python evaluation/run_evaluation.py --no-detail
rm evaluation/results/our_submission.csv
```

---

## 4. Methodology & Pipeline Architecture

### 4.1 Cross-Dataset Harmonization (`src_model/pipeline/skillcorner/`)
To train a supervised model that generalizes to `BAR-ATH`, we align SkillCorner's open tracking data with the target Metrica EPTS format:
- **Spatial Normalization:** SkillCorner pitch coordinates (centred at `(0, 0)` in metres) are scaled by match-specific pitch dimensions and shifted to `[0, 1] x [0, 1]`. Attacking directions are standardized per half so features are invariant to which side a team defends.
- **Temporal Resampling (10 FPS -> 25 FPS):** Player and ball coordinates are linearly interpolated to 25 FPS (`resample_match`), and velocities are recomputed in metres per second.
- **Detection Filtering:** Only detected ball and player observations (`is_detected == True`) are retained (`BALL_DETECTED_ONLY = True`, `PLAYERS_DETECTED_ONLY = True`) so the model experiences realistic visibility dropouts similar to `BAR-ATH`.
- **Ground-Truth Label Construction:** Frame labels (`H`, `A`, `D`, `X`) are synthesized from SkillCorner's `phases_of_play.csv` and per-frame `possession.group` annotations.

### 4.2 Spatio-Temporal Feature Engineering (`src_model/pipeline/model/features.py`)
For every frame, `build_features()` constructs **265 `float32` features** organized into six functional blocks:

1. **Ball Kinematics & Geometry (`ball_block`):**
   - Visibility indicator (`ball_seen`), normalized forward progression (`ball_fwd`), distance to touchlines, goal lines, and nearest goal, penalty-box presence (`ball_in_box`), out-of-bounds margin (`ball_out_margin`).
   - Incoming/outgoing speed (`0.2s` window), speed change, acceleration (`ball_accel`), turn angle (`ball_turn_deg`), cumulative displacement over `1s` and `3s`, and continuous stationary duration (`ball_still_s`).
2. **Visibility Gap Bridging (`gap_block`):**
   - Explicitly models periods when the ball disappears (`~43.1%` of frames in `BAR-ATH`): duration of the current invisibility gap (`gap_len_s`), relative position within the gap (`gap_position`), spatial jump distance between last-seen and next-seen coordinates (`gap_jump`), and boundary/speed/proximity states immediately before (`gap_prev_*`) and after (`gap_next_*`) the dropout.
3. **Nearest-Player & On-Ball Interaction (`touch_block`):**
   - Distance, speed, relative velocity, and velocity cosine alignment (`nearest_alignment`) of the closest player to the ball, plus persistence duration (`nearest_persistence_s`).
   - Inferred `on_ball` (player within `2.0m`) and kinematic `touch` events (player within `2.0m` coinciding with a ball turn >= 20 deg or speed change >= 2.0 m/s), including time elapsed since (`since_touch`) and until (`until_touch`) the nearest touch and rolling touch counts over +/- 3s.
4. **Team-Relative Proximity, Pressure & Collective Shape (`team_block`):**
   - Computed symmetrically from each team's perspective: nearest, second-nearest, and mean top-3 player distances to the ball; player counts within `2m`, `3m`, `5m`, `10m`, and `20m`; defensive pressure count (`opponents within 3.0m` of the ball-carrier).
   - **Ball-independent collective geometry** (crucial when the ball is invisible): defensive line (`defline`), attacking line (`topline`), team length, width, compactness, forward collective velocity (`vel_attack`), share of slow/walking players (`< 1.0 m/s`, highly indicative of dead balls `D`), players per pitch third, and players inside own/opponent penalty boxes.
5. **Cross-Team Differentials (`both_teams_block`):**
   - Explicit `home_minus_away_*` differentials across nearest distances, pressure, collective forward velocity, defensive/attacking lines, compactness, and touch recency, plus overall speed quantiles (`q20`, `q80`) and centroid separation.
6. **Multi-Scale Temporal Context (`add_time_context` & `add_past_future_context`):**
   - Centred rolling means over **1s** and **4s** windows (`_r1s`, `_r4s`).
   - Exact lead/lag shifts at **+/- 1s** (`_m1s`, `_p1s`).
   - Past-only (`_past2s`, `_past6s`), future-only (`_next2s`, `_next6s`), and regime-change differences (`_change2s`, `_change6s`) over **2s** and **6s** windows.

### 4.3 Symmetric Augmentation & Model Training (`dataset.py`, `train.py`, `tuning.py`)
- **Match Split & Subsampling:** The 20 SkillCorner matches are split at the match level into **16 training matches** and **4 held-out validation matches**. Training frames are sampled every 5th graded frame (`SAMPLE_EVERY = 5`, i.e., `0.2s` intervals) to reduce redundant adjacent frames.
- **Team-Swap Augmentation (`swap_teams`):** Every sampled training frame is duplicated with Home and Away roles exchanged (`home_*` <-> `away_*`, `home_minus_away_*` negated, `_fwd` coordinates inverted to `1 - x`, and labels `H` <-> `A`). This produces **945,714 training rows** and guarantees exact symmetry between Home and Away predictions.
- **Classifier:** `HistGradientBoostingClassifier` (`max_iter=300`, `learning_rate=0.1`, `max_leaf_nodes=31`, `min_samples_leaf=200`, `l2_regularization=1.0`).
- **Smoothing Selection & Post-Processing:** On the 4 held-out matches (evaluated across all frames without subsampling), predicted probabilities are smoothed with a centred rolling average across candidate windows `(0, 1, 2, 4, 8)` seconds. A **4.0-second window** achieves the highest held-out **Spell F1 (`0.456`)** alongside **Macro F1 (`0.868`)**. The final model is then fit on all 20 matches.

### 4.4 Domain Post-Processing on `BAR-ATH`
1. **Graded Period Masking:** Frames outside the official half intervals (`11,185..90,651` and `110,808..190,197`) are assigned `X`.
2. **Pre-Kickoff Dead Ball Rule:** As noted in `info/FAQ.md`, the second half is graded starting at frame `110,808`, whereas the actual kickoff occurs at frame `113,742`. `mark_pre_kickoff_dead_ball()` deterministically assigns `D` to `[110,808, 113,742)`.

---

## 5. Final Output Summary (`DEAI4BARCA.csv` / `submission.csv`)

| Period | Start Frame | End Frame |
| :--- | ---: | ---: |
| First half | 11,185 | 90,651 |
| Second-half pre-kickoff (`D`) | 110,808 | 113,741 |
| Second half | 113,742 | 190,197 |

- **Total Frames:** `199,319` (`frame 0..199318`)
- **Predicted Label Distribution:**
  - `D` (Dead Ball): `83,026` (`41.7%`)
  - `H` (Home Possession): `47,197` (`23.7%`)
  - `X` (Ungraded / Out of Period): `40,462` (`20.3%`)
  - `A` (Away Possession): `28,634` (`14.4%`)
- **Reconstructed Possessions:** `231` spells with a mean duration of `13.1s`.
