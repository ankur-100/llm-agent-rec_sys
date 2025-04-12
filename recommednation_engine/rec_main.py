# recommender.py
from open_connect import get_session

def recommend_from_db(user_id):
    """
    A simple function that returns items from the user's top preferred genre.
    """
    session = get_session()
    
    # Get the top genre for the user based on preference_score
    query = session.execute("""
        SELECT g.genre_name
        FROM UserPreferences up
        JOIN Genres g ON up.genre_id = g.genre_id
        WHERE up.user_id = :uid
        ORDER BY up.preference_score DESC
        LIMIT 1
    """, {"uid": user_id})
    row = query.fetchone()
    if not row:
        session.close()
        return []
    top_genre = row[0]
    
    # Retrieve up to 5 items that match the top genre
    items_query = session.execute("""
        SELECT i.title, i.description
        FROM Items i
        JOIN Genres g ON i.genre_id = g.genre_id
        WHERE g.genre_name = :gname
        LIMIT 5
    """, {"gname": top_genre})
    recommendations = [{"title": r[0], "description": r[1]} for r in items_query]
    session.close()
    return recommendations

def update_user_preference(user_id, genre_name, delta):
    """
    Update the user's preference score for a given genre by delta.
    """
    session = get_session()
    # Ensure the genre exists.
    session.execute("""
        INSERT OR IGNORE INTO Genres (genre_name)
        VALUES (:gname)
    """, {"gname": genre_name})
    session.commit()
    # Upsert user preference.
    session.execute("""
        INSERT INTO UserPreferences (user_id, genre_id, preference_score)
        VALUES (:uid, (SELECT genre_id FROM Genres WHERE genre_name = :gname), :delta)
        ON CONFLICT(user_id, genre_id) DO UPDATE SET
            preference_score = UserPreferences.preference_score + :delta
    """, {"uid": user_id, "gname": genre_name, "delta": delta})
    session.commit()
    session.close()
