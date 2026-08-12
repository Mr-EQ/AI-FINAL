"""
recommender_interface.py
=========================
Bridges Member 2's real recommendation_engine.py to the contract app.py
expects. This is the ONLY file that changes when the recommender changes —
app.py never needs to know how recommendations are actually generated.

STATUS: now wired to the REAL collaborative-filtering engine
(recommendation_engine.py — user-based cosine similarity), replacing the
earlier popularity-based placeholder.

IMPORTANT CHANGE from the placeholder version: user_id is now a STRING
hash (e.g. "6beb4699102775dab57aa406c5ea1217c4ff4869"), matching the real
user_ids.npy file — NOT an integer index 0..n_users-1. app.py's UI has
been updated to match (see the "pick a random user" / text-input flow).

CONTRACT (unchanged)
---------------------
    get_recommendations(user_id: str, top_n: int = 10) -> list[dict]

    Each dict has:
        track_id, name, artist, genre, year, tags, spotify_preview_url, score

    Member 2's raw output only has {track_id, song, artist, score} — this
    module renames "song" -> "name" and enriches every result with genre,
    year, tags, and spotify_preview_url by joining back to
    music_info_clean.csv on track_id.
"""

from functools import lru_cache
from pathlib import Path

import pandas as pd

import recommendation_engine as engine

DATA_DIR = Path(__file__).parent

DEFAULT_TOP_USERS = 5  # how many similar users to base recommendations on


@lru_cache(maxsize=1)
def _load_music_info() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "music_info_clean.csv")


@lru_cache(maxsize=1)
def _track_lookup() -> pd.DataFrame:
    """music_info indexed by track_id for fast enrichment lookups."""
    df = _load_music_info()
    return df.set_index("track_id")


def get_user_count() -> int:
    return len(engine.get_all_user_ids())


def get_song_count() -> int:
    return len(_load_music_info())


def get_all_user_ids():
    """Exposed so app.py can offer a 'pick a random real user' button."""
    return engine.get_all_user_ids()


def user_listening_history(user_id: str) -> pd.DataFrame:
    """Return the songs a given user has already listened to, with play counts."""
    d = engine._load_all()
    user_to_idx = d["user_to_idx"]
    idx_to_track = d["idx_to_track"]
    interaction_matrix = d["interaction_matrix"]
    music_info = _load_music_info()

    if user_id not in user_to_idx:
        return pd.DataFrame()

    user_index = user_to_idx[user_id]
    row = interaction_matrix[user_index]
    if row.nnz == 0:
        return pd.DataFrame()

    track_ids = [idx_to_track[int(i)] for i in row.indices]
    history = pd.DataFrame({"track_id": track_ids, "play_count": row.data})
    history = history.merge(music_info, on="track_id", how="left")
    return history.sort_values("play_count", ascending=False).reset_index(drop=True)


def get_recommendations(
    user_id: str, top_n: int = 10, top_users: int = DEFAULT_TOP_USERS
) -> list[dict]:
    """
    See module docstring for the full contract.

    Runs Member 2's real user-based collaborative filtering, then enriches
    each result with genre/year/tags/spotify_preview_url from
    music_info_clean.csv (their function doesn't return those fields).
    """
    if not engine.user_exists(user_id):
        raise ValueError(f"user_id '{user_id}' was not found in user_ids.npy")
    if top_n <= 0:
        raise ValueError("top_n must be a positive integer")
    if top_users <= 0:
        raise ValueError("top_users must be a positive integer")

    raw_results = engine.recommend_for_user(
        user_id, top_users=top_users, top_songs=top_n
    )

    if not raw_results:
        return []

    lookup = _track_lookup()
    enriched = []
    for r in raw_results:
        track_id = r["track_id"]
        extra = lookup.loc[track_id] if track_id in lookup.index else None

        genre = extra["genre"] if extra is not None and pd.notna(extra.get("genre")) else "Unkown"
        year = int(extra["year"]) if extra is not None and pd.notna(extra.get("year")) else 0
        tags = extra["tags"] if extra is not None and pd.notna(extra.get("tags")) else ""
        preview_url = (
            extra["spotify_preview_url"]
            if extra is not None and pd.notna(extra.get("spotify_preview_url"))
            else ""
        )

        enriched.append(
            {
                "track_id": track_id,
                "name": r["song"],
                "artist": r["artist"],
                "genre": genre,
                "year": year,
                "tags": tags,
                "spotify_preview_url": preview_url,
                "score": r["score"],
            }
        )

    return enriched
