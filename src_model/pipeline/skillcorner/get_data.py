import json
import urllib.request

from src_model.constants import *
from pipeline.skillcorner.utils import match_file
from pipeline.skillcorner.constants import RAW_DIR

REPO_RAW = "https://raw.githubusercontent.com/SkillCorner/opendata/master/data"
REPO_LFS = "https://media.githubusercontent.com/media/SkillCorner/opendata/master/data"  # tracking is in Git LFS

MATCH_FILES = [
    "match.json",
    "phases_of_play.csv",
    "dynamic_events.csv",
    "tracking_extrapolated.jsonl"
]


def list_match_ids() -> list[int]:
    """Ids of every match in the repo."""
    with urllib.request.urlopen(f"{REPO_RAW}/matches.json") as response:
        return [match["id"] for match in json.load(response)]

def download_match(match_id: int) -> None:
    """Download the files of one match that are not on disk yet."""
    for name in MATCH_FILES:
        path = match_file(match_id, name)
        if path.exists():
            continue
        base = REPO_LFS if name.endswith(".jsonl") else REPO_RAW
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".part")   # so an interrupted download is not kept
        urllib.request.urlretrieve(f"{base}/matches/{match_id}/{match_id}_{name}", partial)
        partial.rename(path)


def get_skillcorner_data() -> None:
    """Download the files of the given matches, or all matches if None."""
    match_ids = list_match_ids()
    for match_id in match_ids:
        download_match(match_id)
    print(f"{len(match_ids)} matches in {RAW_DIR}")