# deepseek_api.py
import requests

DEEPSEEK_API_KEY = "sk-28ee1f02c07e48e8b3829b75fc0c6927"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    "Content-Type": "application/json"
}

def call_deepseek_chat(messages, temperature=0.7, model="deepseek-chat"):
    """
    Call the DeepSeek API for chat completions.
    
    Args:
        messages (list): List of message dicts with keys "role" and "content".
        temperature (float): Sampling temperature.
        model (str): Model name to use.
    
    Returns:
        str: The text output from the DeepSeek API.
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature
    }
    response = requests.post(DEEPSEEK_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content'].strip()
    else:
        raise Exception(f"DeepSeek API error {response.status_code}: {response.text}")
