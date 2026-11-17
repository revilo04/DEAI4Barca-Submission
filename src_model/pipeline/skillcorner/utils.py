from pathlib import Path
from pipeline.skillcorner.constants import RAW_DIR

def match_file(match_id: int, name: str) -> Path:
    """Local path of one of a match's files."""
    return RAW_DIR / str(match_id) / f"{match_id}_{name}"