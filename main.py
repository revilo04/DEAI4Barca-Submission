import sys
from pathlib import Path

# Ensure both root and src_model are on sys.path so running `python main.py` from root works seamlessly
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src_model"))

from src_model.main import main

if __name__ == "__main__":
    main()
