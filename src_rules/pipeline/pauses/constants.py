from games import GAME, GAMES

# --- Match / pitch geometry (metres) -----------------------------------------
FPS = GAMES[GAME]["fps"]
PITCH_LENGTH, PITCH_WIDTH = GAMES[GAME]["pitch"]
N_SLOTS = 46                                   # tracking columns 1_x .. 46_x
# Official dimensions (16.5 metros de profundidad y 40.3 metros de ancho)
PENALTY_BOX_DEPTH = 16.5
PENALTY_BOX_WIDTH = 40.3
GOAL_AREA_DEPTH, GOAL_AREA_HALF_WIDTH = 5.5, 9.16

# --- Feature parameters ------------------------------------------------------
KEEPER_MIN_BOX_SHARE = 0.2   # min share of visible frames in a penalty box to count as keeper
MIN_TRACKED_PLAYERS = 10     # frames with fewer tracked players are skipped for the speed mean
KEEPER_MAX_GOAL_DIST_M = 25  # a keeper pause must restart within this distance of a goal line

# --- Rule thresholds ---------------------------------------------------------
LONG_PAUSE_S = 30.0              # R0: longer pauses are always a stoppage
SHORT_PAUSE_S = 5.0              # R1: shorter pauses are always a tracking dropout
THROW_IN_TOUCHLINE_M = 4.0       # R4: ball disappeared this close to a touchline...
THROW_IN_DISPLACEMENT_M = 10.0   # R4: ...and reappeared this close to where it left
PLAY_ON_SPEED_MS = 2.5           # R5: players this fast mean play never stopped

# --- Labels ------------------------------------------------------------------
DEAD_BALL = "D"
ERROR = "X"
KEEPER = "K"   # K = internal only, resolved to H/A/X
HOME_KEEPER = "H"
AWAY_KEEPER = "A"

# Which goal (LEFT = x near 0, RIGHT = x near PITCH_LENGTH) each team defends,
# keyed by (half, team). Hardcoded for this match.
KEEPER_SIDE = {(1, 0): "LEFT", (1, 1): "RIGHT", (2, 0): "RIGHT", (2, 1): "LEFT"}