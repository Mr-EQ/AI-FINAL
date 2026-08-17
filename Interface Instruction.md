## Music Recommender interface
## To set up make sure that 
1. You have installed the -r requirements.txt
   `pip install -r requirements.txt`
2. these files are in this same folder:
- `music_info_clean.csv`
- `interaction_matrix.npz`
- `user_ids.npy`
- `track_ids.npy`
- `app.py`
3.Have a .streamlit file where `config.toml` should be
4.Run by using
`streamlit run app.py`
Apart from app.py which has two tabs;
a. Recommendations: which picks a user ID,gets top-N songs,optionally views their listening history,
plays 30s preview inline.
b.Browse Library: search/filter the full song catalog by name, artist, genre, year.
We have the `recommendation_engine.py` which works as a user-based  collaborative
  filter (cosine similarity over the interaction matrix),the `recommender_interface.py` which acts as an in between for the `app.py` and the raw ouput such as `music_info_clean.csv` and also the `test_recommender_contract.py` which is run any time the recommender changes, to check it still returns what the UI expects before you demo
## Note on User IDs
User IDs are string hashes (e.g.
`6beb4699102775dab57aa406c5ea1217c4ff4869`), not simple integers 
