# rag.py
from api_connection import call_deepseek_chat

def retrieve_external_info(query):
    """
    Simulated RAG retrieval: Given a query, returns a list of document snippets.
    In production, use a search API or vector retrieval tool.
    """
    # For demonstration, we return static example snippets.
    docs = [
        "Movie XYZ (2025) is a critically acclaimed sci-fi movie directed by Director A. It features mind-bending visuals and a deep plot.",
        "Movie ABC (2024) is a new action release with dramatic elements and has received good reviews for its innovative storytelling."
    ]
    return docs

def summarize_documents(docs):
    """
    Summarize the retrieved documents using DeepSeek as a summarizer.
    """
    combined_text = "\n\n".join(docs)
    summary_prompt = f"Summarize the following movie information into 3 key points:\n\n{combined_text}"
    messages = [
        {"role": "system", "content": "You are an expert summarizer."},
        {"role": "user", "content": summary_prompt}
    ]
    summary = call_deepseek_chat(messages, temperature=0.5)
    return summary
