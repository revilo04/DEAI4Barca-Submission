from src_model.constants import *

RAW_DIR = ROOT / "data" / "external" / "skillcorner"   # one folder per match, as downloaded
CACHE_DIR = RAW_DIR / "processed"                      # one pickle per converted match

MATCH_IDS = None             # None = every match in the repo (20 matches, about 90 MB each)
SOURCE_FPS = 10              # SkillCorner tracking frame rate
TARGET_FPS = 25              # frame rate matches are resampled to (ours); None keeps 10 fps
N_SLOTS = 46                 # same number of player slots as our tracking

# Our tracking loses the ball on about 41% of graded frames. SkillCorner
# extrapolates the ball when it is not detected; dropping those positions gives
# a similar share of missing ball.
# In SkillCorner data, this is given by "ball_data['is_detected']"
BALL_DETECTED_ONLY = True

# Players off screen are extrapolated too. Kept by default, since our tactical
# camera sees everyone.
PLAYERS_DETECTED_ONLY = True

HOME = "H"
AWAY = "A"
DEAD = "D"
NO_ANSWER = "X"