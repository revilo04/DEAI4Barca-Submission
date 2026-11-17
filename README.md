# Football Tracking Challenge – Possession Detection

## 1. Project Overview

This project aims to reconstruct ball possession from football tracking data, as part of a challenge involving player and ball tracking information.

The objective is to assign a label to every video frame, identifying whether:

- **H** – The home team has possession.
- **A** – The away team has possession.
- **D** – The ball is dead (e.g. throw-in, corner, free kick, goal kick, kickoff, or another interruption).
- **X** – The frame is outside the evaluated playing periods or cannot be assigned to one of the other states.

The main difficulty is that possession is not explicitly provided by the tracking data. It must be inferred from the positions and movements of the ball and players, together with additional event information.

Importantly, the ball is not visible in every frame. Therefore, the pipeline cannot rely exclusively on ball tracking or detected touches.

The project combines:
1. Player and ball tracking data.
2. Smart Tagging information.
3. Ball movement and player proximity.
4. Detected touch events.
5. A state-based approach to reconstruct possession over time.

The current implementation is a working baseline intended to be further improved.

---

## 2. Data

The tracking dataset contains **199,319 frames**, with a frame rate of **25 FPS**.

The available information includes:
- Ball coordinates.
- Player coordinates.
- Player team identification.
- Ball and player movement information derived from tracking.
- Smart Tagging events.
- Detected ball-touch events.

The ball is visible in approximately **56.9% of all frames**. This is one of the main challenges of the task.

### Evaluated periods

The pipeline distinguishes between the following periods:

| Period | Start frame | End frame |
|---|---:|---:|
| First half | 11,185 | 90,651 |
| Second-half pre-kickoff | 110,808 | 113,741 |
| Second half | 113,742 | 190,197 |

The first and second halves are treated separately. Frames outside the relevant playing periods are not considered active-play possession frames.

The second-half pre-kickoff interval is explicitly labelled as dead ball (`D`).

---

## 3. Project Structure

The project is organised into four Jupyter notebooks, each covering a different stage of the possession reconstruction pipeline.

| Notebook | Description |
|---|---|
| `pauses_analysis.ipynb` | Analyses Smart Tagging information and tracking data to identify pauses in play. Produces pause episodes and frame-level pause labels (`D`, `H`, `A`, `X`). |
| `frame_features.ipynb` | Processes the raw tracking data and computes the features used throughout the project, including ball and player positions, velocities, distances and proximity indicators. |
| `merge_labels.ipynb` | Combines pause labels and possession labels into the final frame-level prediction used for evaluation. |
| `possession_video.ipynb` | Detects ball touches and restarts, combines tracking evidence with touch events, and reconstructs team possession frame by frame. |

### Supporting files and outputs

- `touch_events_v3.csv`: input file containing the detected touch events used by the possession detection pipeline.
- `analysis/pauses/`: contains pause detection outputs, including `pause_frame_labels.csv` and `pause_episodes.csv`.
- `analysis/touches/`: contains possession labels, touch and restart information, and diagnostic/audit files used to inspect the detection results.
- `analysis/final/`: contains the final output, `final_frame_labels.csv`, together with videos used for visual inspection.

The final prediction is stored in `analysis/final/final_frame_labels.csv`. It contains one label for each video frame and is the file used by the challenge evaluation script.

---

## 4. Pipeline Overview

The processing is divided into two main parts:

1. **Pause detection:** identifying periods in which ball is not detected.
2. **Possession detection:** identifying which team controls the ball during active play.

The outputs of these two processes are then merged into the final frame-level labels.

### 4.1 Pause detection (pauses_analysis.ipynb)

Pause information is obtained from the Smart Tagging data and processed into frame-level labels.

The following distinctions are important:

- `D`: an actual interruption of play. Possession is temporarily suspended.
- `H`: the home goalkeeper holds the ball.
- `A`: the away goalkeeper holds the ball.
- `X`: tracking error or a frame outside the relevant evaluation period.

A goalkeeper holding the ball does **not** mean that play is dead. It is still possession by that goalkeeper's team.

This distinction is essential because pause labels and possession labels describe different aspects of the game.

### 4.2 Possession detection (frame_features.ipynb)

Possession is reconstructed using tracking information and detected touches.

The main principle is to combine several sources of evidence rather than rely on a single indicator.

The pipeline considers:

- Ball position and visibility.
- Ball speed and direction.
- Distance between the ball and the nearest player.
- Team identity of the nearest player.
- Persistence of player proximity.
- Detected touch events.
- The previous possession state.

A detected touch can provide useful evidence, but it is not mandatory. Not every ball touch is detected, so requiring a touch event would cause valid possession changes to be missed.

### 4.3 Restart detection 

A particularly important part of the pipeline is the detection of restarts following a dead-ball interval (`D`).

For each dead-ball interval, the algorithm searches for evidence that play has resumed.

The current tracking-based approach uses:

- A causal ball-speed calculation.
- A minimum ball-speed threshold.
- A short observation window.
- A minimum number of moving frames.
- Player proximity to the ball.
- The team identity of nearby players.

Touch events are also considered as supporting evidence.

**Important timing convention:** the possession restart frame is currently set to the first frame immediately after the dead-ball interval:

`restart_frame = D_end + 1` (we have to control if this assumption is good or not)

The tracking evidence may only become sufficiently clear a few frames later. This distinction is intentional: the first frame after `D` determines when possession resumes, while subsequent observations help identify the team.

The team taking the restart is inferred from nearby players and touch information. If the evidence is weak or conflicting, the team assignment should ideally remain unresolved rather than being forced.

### 4.4 Possession state reconstruction

After identifying touches, restarts, goalkeeper holds and dead-ball intervals, the pipeline reconstructs possession over time.

The main events considered are:

- Dead-ball starts.
- Goalkeeper holds.
- Restarts.
- Detected touches.

These events are ordered chronologically and processed through a state-based procedure.

The current implementation also uses a minimum number of touches (2/3?) and a maximum time gap (5/10 sec?) to distinguish meaningful possession sequences from isolated or potentially unreliable touch detections.

Current parameters include:

| Parameter | Current value |
|---|---:|
| Minimum touches | 2 |
| Maximum gap between touches | 10 seconds |
| Restart ball-speed threshold | 1.0 m/s |
| Restart movement window | 5 frames |
| Minimum moving frames | 3 |
| Player proximity radius | 10 m |

These are working parameters, not necessarily optimal values. They should be tested and validated rather than treated as fixed assumptions.

---

## 5. Merging the Labels

The possession and pause outputs are combined into a single frame-level prediction.

The merge procedure follows these rules:

1. Start from the possession labels.
2. Replace the corresponding frames with pause labels.
3. Explicitly assign `D` to the second-half pre-kickoff interval.
4. Assign `X` to any remaining unlabelled frames.

This ensures that pause labels take precedence over possession labels.

The final output is:

`analysis/final/final_frame_labels.csv`

It contains two columns:

- `video_frame`
- `label`

There should be exactly one row for each of the 199,319 video frames.

---

## 6. Evaluation

The pipeline is evaluated using the challenge's public ground-truth labels and the provided evaluation module.

The evaluation considers both frame-level classification and possession-segment reconstruction.

The main metrics currently monitored are:

- **Macro F1:** frame-level classification performance across the label classes.
- **Spell F1:** quality of the reconstructed possession spells, evaluated using temporal overlap (we are ranked based on this).
- **Number of predicted possessions:** the number of reconstructed H/A possession segments.
- **Mean possession duration:** average duration of the predicted possession segments.


---

## 7. Current Limitations and Potential Improvements

There are several areas where the pipeline can be improved.

### 7.1 Possession changes

The current algorithm sometimes merges consecutive possession spells belonging to different teams.

This is especially problematic when:
- A possession is very short.
- A team regains the ball shortly after losing it.
- A touch is not detected.
- The ball is temporarily invisible.

A possible improvement is to investigate changes in nearest-player team, ball direction, player movement and ball speed together.

### 7.2 Restart team identification

The identification of the team taking a restart is not always reliable.

Potential improvements include:
- Using the nearest player's team at the restart.
- Analysing the direction of ball movement.
- Considering the attacking direction of each team in each half.
- Distinguishing the player taking the restart from players who approach the ball afterwards.
- Avoiding team assignments when the evidence is insufficient.

Absolute ball direction alone is not enough to identify the team. It must be interpreted in relation to each team's attacking direction.

### 7.3 Ball visibility

Since the ball is missing in approximately 43% of frames, the pipeline needs to make better use of player tracking.

Possible improvements:
- Estimating possession during short periods of ball invisibility.
- Using player proximity persistence.
- Analysing player movement and team-level positioning.
- Avoiding unnecessary switches to unknown possession when the available evidence is still sufficient.

### 7.4 Parameter calibration

The current parameters were selected through experimentation.

They should be tested systematically, particularly:
- Minimum number of touches.
- Maximum gap between touches.
- Ball-speed threshold for restarts.
- Restart detection window.
- Player proximity radius.
- Minimum evidence required for team identification.

Parameter changes should be evaluated against both frame-level and spell-level metrics.

### 7.5 Better error analysis

Rather than focusing only on aggregate metrics, it would be useful to inspect individual possession spells.

The most important questions are:
- Which ground-truth possessions are completely missed?
- Which consecutive possessions are incorrectly merged?
- Are possession boundaries detected too early or too late?
- Are teams assigned incorrectly?
- Are errors concentrated around restarts or during normal play?

A frame-by-frame comparison with the ground truth and video is likely to be particularly useful.


