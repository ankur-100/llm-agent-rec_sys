import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# -------------------------------
# Configuration: Update these values
# -------------------------------
# Path to the MovieLens 32M extracted folder
DATA_DIR = r"E:\Downloads- 24.1.25\ml-32m\ml-32m"

# MySQL connection parameters
MYSQL_USER = "root"
MYSQL_PASS = "ankur999"
MYSQL_HOST = "localhost"      # or the host of your DB server
MYSQL_PORT = "3306"
MYSQL_DB = "movielens"

# Create SQLAlchemy engine (update connection string accordingly)
engine = create_engine(f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASS}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}")

# -------------------------------
# 1. Load Movies Data
# -------------------------------
def load_movies():
    movies_file = os.path.join(DATA_DIR, "movies.csv")
    try:
        movies_df = pd.read_csv(movies_file)
        print(f"Loaded movies.csv: {movies_df.shape[0]} records")
    except Exception as e:
        print("Error loading movies.csv:", str(e))
        return None

    # Preprocess genres: Movies have genres as pipe-separated values.
    # For simplicity, we use the first genre.
    movies_df["primary_genre"] = movies_df["genres"].apply(lambda s: s.split("|")[0] if pd.notnull(s) else None)
    
    # Add a placeholder description (to be enriched later via RAG) and release_year (if available in title)
    # You can extend this by parsing the title for the year.
    movies_df["description"] = "Description pending enrichment."
    
    # Create a column "release_year" by extracting a 4-digit number within parentheses from the title.
    import re
    def extract_year(title):
        match = re.search(r'\((\d{4})\)', title)
        return int(match.group(1)) if match else None
    movies_df["release_year"] = movies_df["title"].apply(lambda x: extract_year(x))
    
    # Rename columns to match your Items table schema if needed.
    # For instance, we assume your Items table has: title, description, genre_id, release_year.
    # We'll handle linking genres later.
    return movies_df

# -------------------------------
# 2. Load Ratings Data
# -------------------------------
def load_ratings():
    ratings_file = os.path.join(DATA_DIR, "ratings.csv")
    try:
        ratings_df = pd.read_csv(ratings_file)
        print(f"Loaded ratings.csv: {ratings_df.shape[0]} records")
    except Exception as e:
        print("Error loading ratings.csv:", str(e))
        return None
    return ratings_df

# -------------------------------
# 3. Load Tags Data (Optional)
# -------------------------------
def load_tags():
    tags_file = os.path.join(DATA_DIR, "tags.csv")
    try:
        tags_df = pd.read_csv(tags_file)
        print(f"Loaded tags.csv: {tags_df.shape[0]} records")
    except Exception as e:
        print("Error loading tags.csv:", str(e))
        return None
    return tags_df

# -------------------------------
# 4. Insert Data into MySQL
# -------------------------------
def insert_movies(movies_df):
    try:
        # Load the current genres from the DB into a mapping dict: genre_name -> genre_id
        with engine.connect() as conn:
            result = conn.execute(text("SELECT genre_id, genre_name FROM genres"))
            # Using tuple indexing: row[0] -> genre_id, row[1] -> genre_name
            genre_map = {str(row[1]).strip().lower(): row[0] for row in result}
        
        # Map the primary genre (already normalized to lowercase) to genre_id
        movies_df["genre_id"] = movies_df["primary_genre"].map(lambda g: genre_map.get(g))
        
        # Filter out rows with no matching genre_id
        valid_movies = movies_df[movies_df["genre_id"].notna()].copy()
        print(f"Valid movies to insert: {len(valid_movies)}. Skipped: {len(movies_df) - len(valid_movies)}")
        
        # Select final fields for the Items table
        items_df = valid_movies[["title", "description", "genre_id", "release_year"]]
        
        # Insert into the 'items' table (use consistent casing for table name)
        items_df.to_sql("items", con=engine, if_exists="append", index=False, method='multi')
        print(f"✅ Inserted {len(items_df)} movies into Items table.")
            
    except SQLAlchemyError as e:
        print("❌ Error inserting movies:", e)


def insert_ratings(ratings_df):
    # We'll create a new table "Ratings" for ratings if not already created.
    # For demonstration, we will insert data into a Ratings table.
    try:
        ratings_df.to_sql("Ratings", con=engine, if_exists="append", index=False, method='multi')
        print("Ratings data inserted into Ratings table.")
    except SQLAlchemyError as e:
        print("Error inserting ratings:", e)

def insert_tags(tags_df):
    # Similarly, insert tags into a table "Tags"
    try:
        tags_df.to_sql("Tags", con=engine, if_exists="append", index=False, method='multi')
        print("Tags data inserted into Tags table.")
    except SQLAlchemyError as e:
        print("Error inserting tags:", e)

def main():
    # Load datasets
    movies_df = load_movies()
    ratings_df = load_ratings()
    tags_df = load_tags()  # Optional
    
    if movies_df is not None:
        insert_movies(movies_df)
        
    if ratings_df is not None:
        insert_ratings(ratings_df)
        
    if tags_df is not None:
        insert_tags(tags_df)

if __name__ == "__main__":
    main()
