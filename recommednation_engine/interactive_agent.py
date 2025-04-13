# interactive_agent.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent_orchestration import run_agent  # our LangChain agent call from agent.py
from rag import retrieve_external_info, summarize_documents
from recommender import hybrid_recommend
from open_connect import get_session, User
import uvicorn

app = FastAPI(title="Interactive Recommendation System")

# For simplicity, we'll assume a single user (user_id=1) for now.
CURRENT_USER_ID = 1

class ChatInput(BaseModel):
    message: str

# Endpoint for chat conversation.
@app.post("/chat")
def chat_endpoint(chat: ChatInput):
    try:
        # Here, we simply pass the message to the agent.
        response_text = run_agent(chat.message)
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to get final recommendations.
@app.get("/recommendations")
def recommendations_endpoint(liked_movie: str = None):
    try:
        # Optionally, check the DB for existing user preferences here.
        # For this example, if user provided a liked movie, use it for hybrid recommendation.
        # Also, run RAG retrieval to update items.
        # Simulate RAG:
        docs = retrieve_external_info("latest movies info")
        summary = summarize_documents(docs)
        
        # (In production, parse 'summary' to update your Items DB if needed.)
        # Then, generate recommendations:
        recs = hybrid_recommend(CURRENT_USER_ID, liked_movie_title=liked_movie, top_n=5)
        return {"recommendations": recs.to_dict(orient="records"), "rag_summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health-check endpoint.
@app.get("/")
def read_root():
    return {"message": "Interactive Recommendation System is running."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)