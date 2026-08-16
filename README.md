# Ethical Music Recommendation Engine 

This project is a music recommendation system that recommends songs to users based on the listening behaviour of similar users. The system uses user-based collaborative filtering and cosine similarity to identify users with similar listening patterns.

The system uses a user-song interaction matrix containing users' play counts. After identifying users similar to a selected target user, it recommends songs listened to by those users that the target user has not previously listened to. Candidate songs are ranked using their recommendation scores, and the highest-ranked songs are returned to the user.

Setup and Installation:
1. Clone the repository
git clone <repository-url>
cd AI-FINAL

3. Create a virtual environment
python -m venv .venv

5. Activate the virtual environment
Windows:.venv\Scripts\activate

7. Install the required libraries
pip install pandas numpy scipy scikit-learn


How to Run the System:
Open the project folder in VS Code.
Activate the .venv virtual environment.
Open the recommendation system notebook in the notebooks folder.
Select the .venv Python kernel.
Run the notebook cells in order.
Select a target user and generate recommendations.

Usage Example:

A target user can be selected from the available user IDs:

target_user = user_ids[0]

recommendations = recommend_for_user(
    target_user,
    top_users=5,
    top_songs=5
)

for i, song in enumerate(recommendations, start=1):
    print(f"{i}. {song['song']} - {song['artist']}")

Example output:

1. Quiet Little Voices - We Were Promised Jetpacks
2. Ice Monster - Minus the Bear
3. That Should Be Me - Justin Bieber
4. The Killing Hand - Dream Theater
5. Don't Get Cute - Kurt Vile

The system identifies users with similar listening patterns using cosine similarity and recommends songs from those users that the target user has not previously listened to.
