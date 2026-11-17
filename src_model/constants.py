import sys
from pathlib import Path

# .../initial-challenge
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import games  # noqa: E402

# Video frames of each half, derived from the per-half offsets in games.py.
# Both ends of each range are included.
(_, FIRST_HALF_START, FIRST_HALF_END), (_, SECOND_HALF_START, SECOND_HALF_END) = (
    games.half_video_blocks(games.GAME)
)
SECOND_HALF_KICKOFF = 113742  # second half is graded from SECOND_HALF_START but only kicks off here
