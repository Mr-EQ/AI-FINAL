"""
app.py — Ethical Music Recommendation Engine (Streamlit UI)
=============================================================
Member 4 (Interface & Integration) deliverable.

Run with:
    streamlit run app.py

Expects these files in the SAME folder as this script:
    - music_info_clean.csv
    - interaction_matrix.npz
    - .streamlit/config.toml  (theme colors — keep this folder alongside app.py)

Integration point with Member 2:
    See recommender_interface.py — that's the only file that needs to
    change once the real collaborative-filtering model is ready.
"""

import random

import streamlit as st
import pandas as pd

from recommender_interface import (
    get_recommendations,
    user_listening_history,
    get_user_count,
    get_song_count,
    get_all_user_ids,
    _load_music_info,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Music Recommender",
    page_icon="🎧",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Design tokens (black / green / blue / white palette, flat — no gradients)
# ---------------------------------------------------------------------------
ACCENTS = ["#00C853", "#1B7A43", "#66BB6A", "#2E7D32", "#A5D6A7"]  # all shades of green

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@500;700&family=Quicksand:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Quicksand', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Fredoka', sans-serif !important;
        letter-spacing: 0.2px;
    }

    .hero-title {
        font-family: 'Fredoka', sans-serif;
        font-size: 2.1rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0;
        color: #FFFFFF;
    }
    .hero-title .note {
        color: #00C853;
    }

    .subtitle {
        color: #3B82F6;
        font-size: 0.95rem;
        margin-top: 0.15rem;
        margin-bottom: 1.2rem;
    }

    /* song cards — flat black/dark-green, no gradients */
    .song-card {
        padding: 1rem 1.1rem;
        border-radius: 14px;
        background-color: #0F2E1A;
        border: 1px solid #1B7A43;
        margin-bottom: 0.7rem;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .song-card:hover {
        border-color: #00C853;
        box-shadow: 0 0 14px rgba(0,200,83,0.25);
    }
    .song-title {
        font-family: 'Fredoka', sans-serif;
        font-size: 1.08rem;
        font-weight: 600;
        margin-bottom: 0.15rem;
        color: #FFFFFF;
    }
    .song-meta {
        font-size: 0.85rem;
        color: #3B82F6;
        margin-bottom: 0.5rem;
    }
    .tag-pill {
        display: inline-block;
        padding: 3px 10px;
        margin: 2px 4px 0 0;
        border-radius: 999px;
        font-size: 0.7rem;
        font-weight: 600;
        color: #FFFFFF;
    }

    /* mini equalizer, purely decorative */
    .eq {
        display: inline-flex;
        align-items: flex-end;
        gap: 3px;
        height: 16px;
        margin-right: 6px;
        vertical-align: middle;
    }
    .eq span {
        display: block;
        width: 3px;
        border-radius: 2px;
        background: #00C853;
        animation: eq-bounce 1s ease-in-out infinite;
    }
    .eq span:nth-child(1) { height: 40%; animation-delay: 0s; }
    .eq span:nth-child(2) { height: 100%; animation-delay: 0.2s; }
    .eq span:nth-child(3) { height: 65%; animation-delay: 0.4s; }
    .eq span:nth-child(4) { height: 85%; animation-delay: 0.1s; }
    @keyframes eq-bounce {
        0%, 100% { transform: scaleY(0.4); }
        50% { transform: scaleY(1); }
    }

    div[data-testid="stButton"] button {
        border-radius: 8px !important;
        font-family: 'Quicksand', sans-serif;
        font-weight: 700;
        border: 1px solid #00C853 !important;
    }

    .surprise-box {
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        background-color: #0F2E1A;
        border: 1px solid #3B82F6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-title"><span class="note">🎵</span> Music Recommender</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Good music and vibes based off what you love.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data load
# ---------------------------------------------------------------------------
try:
    music_df = _load_music_info()
    n_users = get_user_count()
    n_songs = get_song_count()
    data_loaded = True
except FileNotFoundError as e:
    data_loaded = False
    st.error(
        "⚠️ Couldn't find the data files. Make sure `music_info_clean.csv` and "
        "`interaction_matrix.npz` are in the same folder as `app.py`.\n\n"
        f"Details: {e}"
    )

if not data_loaded:
    st.stop()

# ---------------------------------------------------------------------------
# Mood definitions — tied to REAL audio-feature columns, not just for show
# ---------------------------------------------------------------------------
MOODS = {
    "🎉 Hype": lambda df: (df["energy"] > 0.7) & (df["danceability"] > 0.6),
    "😌 Chill": lambda df: (df["energy"] < 0.4) & (df["acousticness"] > 0.3),
    "💛 Happy": lambda df: df["valence"] > 0.6,
    "🌧️ Moody": lambda df: df["valence"] < 0.4,
}


def render_song_card(song: dict, show_score: bool = False, key_prefix: str = ""):
    tags = str(song.get("tags", "") or "")
    tag_list = [t.strip() for t in tags.split(",") if t.strip()][:5]
    tag_html = "".join(
        f'<span class="tag-pill" style="background-color:{ACCENTS[i % len(ACCENTS)]}">{t}</span>'
        for i, t in enumerate(tag_list)
    )

    year = song.get("year", "")
    genre = song.get("genre", "Unknown")
    if genre == "Unkown":
        genre = "Unlabeled"

    col1, col2 = st.columns([4, 1.3])
    with col1:
        st.markdown(
            f"""
            <div class="song-card">
                <div class="song-title"><span class="eq"><span></span><span></span><span></span><span></span></span>{song.get('name', 'Unknown Title')}</div>
                <div class="song-meta">{song.get('artist', 'Unknown Artist')} · {genre} · {year}</div>
                {tag_html}
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        preview_url = song.get("spotify_preview_url", "")
        if preview_url and isinstance(preview_url, str) and preview_url.startswith("http"):
            st.audio(preview_url)
        else:
            st.caption("No preview available")
    if show_score and "score" in song:
        st.caption(f"Match score: {song['score']:.2f} (based on similar listeners)")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_recs, tab_browse, tab_surprise = st.tabs(
    ["🎯 Recommendations", "📚 Browse Library", "🎲 Surprise Me"]
)

# --- Tab 1: Recommendations ------------------------------------------------
with tab_recs:
    st.write(
        f"Dataset: **{n_users:,}** users · **{n_songs:,}** songs in the interaction matrix."
    )

    all_user_ids = get_all_user_ids()

    if "current_user_id" not in st.session_state:
        st.session_state.current_user_id = str(all_user_ids[100])

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("🎲 Pick a random user"):
            st.session_state.current_user_id = str(random.choice(all_user_ids))
    with col_b:
        typed_id = st.text_input(
            "...or paste a specific user ID",
            value="",
            placeholder="e.g. 6beb4699102775dab57aa406c5ea1217c4ff4869",
        )
        if typed_id.strip():
            st.session_state.current_user_id = typed_id.strip()

    user_id = st.session_state.current_user_id
    st.caption(f"Current user: `{user_id}`")

    col_c, col_d = st.columns([2, 1])
    with col_c:
        top_n = st.slider("Number of recommendations", min_value=3, max_value=25, value=10)
    with col_d:
        top_users = st.slider(
            "Similar users to consider",
            min_value=1,
            max_value=20,
            value=5,
            help="How many similar listeners to base recommendations on.",
        )

    show_history = st.checkbox("Show this user's listening history", value=False)

    if show_history:
        history = user_listening_history(user_id)
        if history.empty:
            st.info("No listening history found for this user ID.")
        else:
            st.write(f"**{len(history)} songs previously played:**")
            st.dataframe(
                history[["name", "artist", "genre", "play_count"]].head(20),
                use_container_width=True,
                hide_index=True,
            )

    st.divider()

    if st.button("🎵 Get Recommendations", type="primary"):
        try:
            with st.spinner("Finding listeners with similar taste..."):
                recs = get_recommendations(user_id, int(top_n), top_users=int(top_users))
        except ValueError as e:
            st.error(f"Invalid input: {e}")
        else:
            if not recs:
                st.warning(
                    "No recommendations could be generated for this user "
                    "(they may have already listened to everything their similar users have)."
                )
            else:
                st.success(f"Top {len(recs)} recommendations for this user! 🎉")
                st.balloons()
                for song in recs:
                    render_song_card(song, show_score=True)

# --- Tab 2: Browse Library --------------------------------------------------
with tab_browse:
    st.write("Search and explore the full song catalog.")

    st.write("**Pick a vibe** (filters using real audio features — energy, valence, danceability, acousticness):")
    mood_cols = st.columns(len(MOODS) + 1)
    if "mood" not in st.session_state:
        st.session_state.mood = "All"
    for i, mood_name in enumerate(["All"] + list(MOODS.keys())):
        with mood_cols[i]:
            if st.button(mood_name, key=f"mood_{mood_name}", use_container_width=True):
                st.session_state.mood = mood_name

    col1, col2, col3 = st.columns(3)
    with col1:
        search_term = st.text_input("Search by song or artist", "")
    with col2:
        genres = sorted(music_df["genre"].dropna().unique().tolist())
        genre_filter = st.selectbox("Genre", ["All"] + genres)
    with col3:
        min_year, max_year = int(music_df["year"].min()), int(music_df["year"].max())
        year_range = st.slider("Year range", min_year, max_year, (min_year, max_year))

    filtered = music_df.copy()
    if search_term:
        mask = filtered["name"].str.contains(search_term, case=False, na=False) | filtered[
            "artist"
        ].str.contains(search_term, case=False, na=False)
        filtered = filtered[mask]
    if genre_filter != "All":
        filtered = filtered[filtered["genre"] == genre_filter]
    if st.session_state.mood != "All":
        filtered = filtered[MOODS[st.session_state.mood](filtered)]
    filtered = filtered[
        (filtered["year"] >= year_range[0]) & (filtered["year"] <= year_range[1])
    ]

    mood_label = f" · vibe: {st.session_state.mood}" if st.session_state.mood != "All" else ""
    st.caption(f"{len(filtered):,} songs match your filters{mood_label}.")

    if filtered.empty:
        st.warning("No songs match those filters — try widening your search or picking a different vibe.")
    else:
        for _, row in filtered.head(30).iterrows():
            render_song_card(row.to_dict())
        if len(filtered) > 30:
            st.caption(f"Showing first 30 of {len(filtered):,} results. Narrow your search to see more.")

# --- Tab 3: Surprise Me -----------------------------------------------------
with tab_surprise:
    st.markdown(
        """
        <div class="surprise-box">
        Feeling indecisive? Hit the button and let fate pick your next song. 🎲✨
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    if st.button("🎲 Surprise me!", type="primary"):
        pick = music_df.sample(1).iloc[0].to_dict()
        render_song_card(pick)
