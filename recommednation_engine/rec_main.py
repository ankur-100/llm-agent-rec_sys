# recommender.py
import pandas as pd
from sqlalchemy import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from open_connect import get_session

def load_movies_from_db():
    """Load movie data from the Items table along with genre info."""
    session = get_session()
    query = session.execute(text("""
        SELECT i.item_id, i.title, i.description, g.genre_name
        FROM Items i
        LEFT JOIN Genres g ON i.genre_id = g.genre_id
    """))
    rows = query.fetchall()
    session.close()
    # Convert rows to a DataFrame.
    movies_df = pd.DataFrame(rows, columns=["item_id", "title", "description", "genre_name"])
    return movies_df

def content_based_scores(movies_df):
    """Compute cosine similarity based on movie descriptions using TF-IDF."""
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies_df['description'].fillna(''))
    sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return sim_matrix

def dummy_collaborative_scores(movies_df):
    """
    For demonstration, create a dummy collaborative score.
    In a real system, this might come from a model (e.g. LightFM or MF).
    Here, we'll use a random vector for each movie.
    """
    np.random.seed(0)
    return np.random.rand(len(movies_df))

def hybrid_recommend(user_id, liked_movie_title=None, top_n=5):
    """
    Generate recommendations by blending content similarity and dummy collaborative scores.
    Optionally, if a liked movie is provided, use its content similarity.
    """
    movies_df = load_movies_from_db()
    if movies_df.empty:
        return pd.DataFrame()
    
    content_sim = content_based_scores(movies_df)
    collab_scores = dummy_collaborative_scores(movies_df)
    
    # If a liked movie is provided, base content score on its similarity row.
    if liked_movie_title:
        idx = movies_df.index[movies_df['title'].str.lower() == liked_movie_title.lower()].tolist()
        if idx:
            idx = idx[0]
            content_scores = content_sim[idx]
        else:
            content_scores = content_sim.mean(axis=0)  # fallback average
    else:
        content_scores = content_sim.mean(axis=0)
    
    # Compute the hybrid score as a weighted average. (Weights here are 0.5/0.5.)
    hybrid_scores = 0.5 * collab_scores + 0.5 * content_scores
    top_indices = hybrid_scores.argsort()[::-1][:top_n]
    return movies_df.iloc[top_indices][['title', 'description']]

if __name__ == "__main__":
    # For a test: recommend based on "Inception"
    recs = hybrid_recommend(user_id=1, liked_movie_title="Inception", top_n=3)
    print("Hybrid Recommendations:\n", recs)
