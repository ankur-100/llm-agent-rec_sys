# main.py
from agent_orchestration import run_agent
from rag import retrieve_external_info, summarize_documents
from recommender import recommend_from_db, update_user_preference
from open_connect import init_db, get_session, User
import sys

# Initialize database if not already done
init_db()

def create_test_user():
    """Create a test user if one doesn't exist."""
    session = get_session()
    user = session.query(User).filter_by(user_name="TestUser").first()
    if not user:
        user = User(user_name="TestUser")
        session.add(user)
        session.commit()
    session.close()
    return user.user_id

def interactive_session(user_id):
    print("Assistant: Hello! What kind of recommendation do you need? (e.g., movies, songs, books)")
    
    conversation_history = []
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            print("Assistant: Goodbye!")
            sys.exit(0)
        
        # Run the agent (chain-of-thought interactive component)
        agent_response = run_agent(user_input)
        print("Assistant:", agent_response)
        conversation_history.append(("User", user_input))
        conversation_history.append(("Assistant", agent_response))
        
        # If the agent's output ends with a question, assume it's asking for clarification.
        if agent_response.strip().endswith('?'):
            clarifying_input = input("User (clarification): ").strip()
            # Optionally update DB based on clarification; e.g., if the user mentions a genre.
            if "sci-fi" in clarifying_input.lower():
                update_user_preference(user_id, "Sci-Fi", 1)
            # Run agent again with updated context
            combined_input = clarifying_input  # In practice, append to conversation history.
            agent_response = run_agent(combined_input)
            print("Assistant:", agent_response)
            conversation_history.append(("User", clarifying_input))
            conversation_history.append(("Assistant", agent_response))
        
        # Let's assume the conversation is finished when the agent response contains "recommendation" keyword.
        if "recommend" in agent_response.lower():
            break

    # Retrieve external info using RAG (simulate retrieval + summarization)
    query_for_rag = user_input  # can be refined by conversation_history if needed.
    retrieved_docs = retrieve_external_info(query_for_rag)
    summary = summarize_documents(retrieved_docs)
    print("Retrieved Summary from External Sources:", summary)
    
    # Optionally update DB with new item info from external summary.
    # For example, if the summary mentions "Movie XYZ" for movies, you could add it to the Items table.
    # (Skipping detailed insertion for brevity.)
    
    # Call the hybrid recommender to get final recommendations from our SQL DB.
    recommendations = recommend_from_db(user_id)
    if recommendations:
        print("Final Recommendations based on your updated preferences:")
        for rec in recommendations:
            print(f" - {rec['title']}: {rec['description']}")
    else:
        print("Sorry, I couldn't find any recommendations from your preferences.")

if __name__ == "__main__":
    user_id = create_test_user()
    interactive_session(user_id)
