"""Load the original dataframes, caching each one in data/derived after the first time."""

import os
import pandas as pd

import games

from kloppy import metrica, sportscode

DERIVED_DIR = os.path.join(games.DATA_DIR, "derived")


def load_tracking() -> pd.DataFrame:
    """
    The tactical tracking dataframe, one row per frame. Delete the cache if the
    data changes.
    """
    path = os.path.join(DERIVED_DIR, "BAR-ATH_tactical_tracking.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)

    paths = games.asset_paths(source='tactical')
    meta, raw = paths['atd_meta'], paths['atd_raw']
    print(f"Loading tracking from {meta} and {raw}...")
    df = metrica.load_tracking_epts(meta_data=meta, raw_data=raw).to_df()

    os.makedirs(DERIVED_DIR, exist_ok=True)
    df.to_pickle(path)
    return df


def load_tags():
    """
    The Smart Tagging dataframe, one row per interval. Delete the cache if the
    data changes.
    """
    path = os.path.join(DERIVED_DIR, "BAR-ATH_tactical_smart_tagging.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)

    path = games.asset_paths()['smart_tagging']
    print(f"Loading smart tagging from {path}...")
    df = sportscode.load(path).to_df()

    os.makedirs(DERIVED_DIR, exist_ok=True)
    df.to_pickle(path)
    return df