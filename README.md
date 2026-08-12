# Ethical Music Recommendation Engine — Interface

Streamlit interface for the group project. This folder is Member 4's
(Interface & Integration) deliverable.

## Setup

```bash
pip install -r requirements.txt
```

Make sure these four files are in this same folder:
- `music_info_clean.csv`
- `interaction_matrix.npz`
- `user_ids.npy`
- `track_ids.npy`

## Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

## What's here

- **`app.py`** — the UI. Two tabs:
  - **Recommendations**: pick a user ID, get top-N songs, optionally view
    their listening history, play 30s previews inline.
  - **Browse Library**: search/filter the full song catalog by name,
    artist, genre, year.
- **`recommendation_engine.py`** — Member 2's real user-based collaborative
  filtering (cosine similarity over the interaction matrix), extracted
  from their notebook into an importable module.
- **`recommender_interface.py`** — glue layer between Member 2's raw output
  and what `app.py` expects. Enriches each recommendation with
  genre/year/tags/preview URL by joining back to `music_info_clean.csv`.
- **`test_recommender_contract.py`** — run this any time the recommender
  changes, to check it still returns what the UI expects before you demo:
  ```
  python test_recommender_contract.py
  ```

## Resolved: song index mapping

Earlier drafts of this app used a fake column→track_id mapping since that
file didn't exist yet. Member 2 has since provided `user_ids.npy` and
`track_ids.npy`, which give the real mapping — the interface now uses
those directly (29,915 of 29,922 track_ids matched `music_info_clean.csv`
on first check; the ~7 unmatched are skipped gracefully, same as Member
2's own code does).

## Note on User IDs

User IDs are string hashes (e.g.
`6beb4699102775dab57aa406c5ea1217c4ff4869`), not simple integers — this
matches how Member 2's model identifies users. The UI has a "pick a
random user" button plus a text field for pasting a specific ID.

## Team contract (for Member 2)

Your function should match this signature:

```python
def get_recommendations(user_id: int, top_n: int = 10) -> list[dict]:
    """
    Returns a list of dicts, each with at least:
    track_id, name, artist, genre, year, tags, spotify_preview_url, score
    """
```

Drop your real implementation into `recommender_interface.py` in place of
the current placeholder — nothing in `app.py` needs to change.
