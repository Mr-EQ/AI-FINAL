from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).parent


@lru_cache(maxsize=1)
def _load_all():
    music_info = pd.read_csv(DATA_DIR / "music_info_clean.csv")
    interaction_matrix = load_npz(DATA_DIR / "interaction_matrix.npz")
    user_ids = np.load(DATA_DIR / "user_ids.npy", allow_pickle=True)
    track_ids = np.load(DATA_DIR / "track_ids.npy", allow_pickle=True)

    user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    track_to_idx = {track_id: idx for idx, track_id in enumerate(track_ids)}
    idx_to_user = {idx: user_id for idx, user_id in enumerate(user_ids)}
    idx_to_track = {idx: track_id for idx, track_id in enumerate(track_ids)}

    return {
        "music_info": music_info,
        "interaction_matrix": interaction_matrix,
        "user_ids": user_ids,
        "track_ids": track_ids,
        "user_to_idx": user_to_idx,
        "track_to_idx": track_to_idx,
        "idx_to_user": idx_to_user,
        "idx_to_track": idx_to_track,
    }


def get_all_user_ids() -> np.ndarray:
    return _load_all()["user_ids"]


def user_exists(user_id) -> bool:
    return user_id in _load_all()["user_to_idx"]


def find_similar_users(user_id, top_n=5):
    d = _load_all()
    user_to_idx = d["user_to_idx"]
    idx_to_user = d["idx_to_user"]
    interaction_matrix = d["interaction_matrix"]

    # Checking if the user exists
    if user_id not in user_to_idx:
        print("User not found")
        return []

    # Getting row index of the target user
    user_index = user_to_idx[user_id]

    # Getting the targets listening data
    target_user = interaction_matrix[user_index]

    # Calculating the cosine similarity
    similarities = cosine_similarity(target_user, interaction_matrix).flatten()

    # Excluding the target user
    similarities[user_index] = -1

    # Finding similar users indices
    similar_indices = similarities.argsort()[::-1][:top_n]

    # storing similar users and their similarity scores
    similar_users = []

    for index in similar_indices:
        similar_users.append((idx_to_user[index], similarities[index]))

    return similar_users


def recommend_songs(user_id, top_users=5, top_songs=5):
    d = _load_all()
    user_to_idx = d["user_to_idx"]
    interaction_matrix = d["interaction_matrix"]

    # Checking if the user exists
    if user_id not in user_to_idx:
        print("User not found")
        return []

    # Getting similar users
    similar_users = find_similar_users(user_id, top_n=top_users)

    # Getting the target user's row
    user_index = user_to_idx[user_id]
    target_vector = interaction_matrix[user_index]

    # Songs that the target user has listened to already
    listened_tracks = set(target_vector.indices)

    # Storing the recommended songs
    recommendation_scores = {}

    # Going through each similar user
    for similar_user_id, similarity_score in similar_users:
        similar_user_index = user_to_idx[similar_user_id]
        similar_user_vector = interaction_matrix[similar_user_index]

        # Checking the song each similar user listened to & Skip songs the target user has already listened to
        for track_index, playcount in zip(
            similar_user_vector.indices, similar_user_vector.data):
                 
            if track_index in listened_tracks:
                continue

            # Weighted recommended songs
            score = similarity_score * playcount

            if track_index in recommendation_scores:
                recommendation_scores[track_index] += score
            else:
                recommendation_scores[track_index] = score

    # Sorting the songs & Keeping only the top songs
    sorted_tracks = sorted(
        recommendation_scores.items(), key=lambda x: x[1], reverse=True
    )
    top_tracks = sorted_tracks[:top_songs]
    return top_tracks


def recommend_for_user(user_id, top_users=5, top_songs=5):
    d = _load_all()
    idx_to_track = d["idx_to_track"]
    music_info = d["music_info"]

    # Ask for more candidates than we actually need
    recommendations = recommend_songs(
        user_id, top_users=top_users, top_songs=top_songs * 10
    )

    song_details = []

    for track_index, score in recommendations:
        track_id = idx_to_track[int(track_index)]

        song = music_info[music_info["track_id"] == track_id]

        if not song.empty:
            song_details.append(
                {
                    "track_id": track_id,
                    "song": song.iloc[0]["name"],
                    "artist": song.iloc[0]["artist"],
                    "score": float(score),
                }
            )

        # Stop once we have enough valid songs
        if len(song_details) == top_songs:
            break

    return song_details
