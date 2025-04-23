from rag import retrieve_external_info, summarize_documents
from open_connect import get_session, Item

def enrich_movie_metadata(movie_title, item_id):
    # Step 1: Retrieve from Google Search / Web scraping (future: implement real retriever)
    retrieved_docs = retrieve_external_info(f"Movie {movie_title} plot, actors, director")
    
    # Step 2: Summarize
    summary = summarize_documents(retrieved_docs)

    # Step 3: Extract info (simple parsing, or use LLM to extract structured data)
    actors = extract_actors_from_summary(summary)
    director = extract_director_from_summary(summary)

    # Step 4: Store in DB
    session = get_session()
    item = session.query(Item).filter(Item.item_id == item_id).first()
    if item:
        item.summary = summary
        item.actors = actors
        item.director = director
        item.enriched = True
        session.commit()
    session.close()
