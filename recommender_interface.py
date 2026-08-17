from functools import lru_cache
from pathlib import Path

import pandas as pd

import recommendation_engine as engine

DATA_DIR = Path(__file__).parent

DEFAULT_TOP_USERS = 5  # number of similar users to base it on


@lru_cache(maxsize=1)
def _load_music_info() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "music_info_clean.csv")


@lru_cache(maxsize=1)
def _track_lookup() -> pd.DataFrame:
    """music_info indexed by track_id(for better lookups)"""
    df = _load_music_info()
    return df.set_index("track_id")


def get_user_count() -> int:
    return len(engine.get_all_user_ids())


def get_song_count() -> int:
    return len(_load_music_info())


def get_all_user_ids():
    """helps the app.py to be able to pick random user"""
    return engine.get_all_user_ids()


def user_listening_history(user_id: str) -> pd.DataFrame:
    """this will return the songs a given user has already listened to, with play counts."""
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

        genre = (
            extra["genre"]
            if extra is not None and pd.notna(extra.get("genre"))
            else "Unkown"
        )
        year = (
            int(extra["year"])
            if extra is not None and pd.notna(extra.get("year"))
            else 0
        )
        tags = (
            extra["tags"] if extra is not None and pd.notna(extra.get("tags")) else ""
        )
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
