# agent.py
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from api_connection import call_deepseek_chat

class DeepSeekLLM(ChatOpenAI):
    """A simple wrapper for DeepSeek API to be used in LangChain."""
    def __init__(self, model_name="deepseek-chat", temperature=0.7):
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, prompts):
        # prompts is a list of strings (assumes a single prompt for simplicity)
        # Build messages with a system instruction
        messages = [
            {"role": "system", "content": (
                "You are a helpful recommendation assistant. "
                "When a request is ambiguous, ask clarifying questions "
                "before recommending.")}
        ]
        messages.append({"role": "user", "content": prompts[0]})
        answer = call_deepseek_chat(messages, temperature=self.temperature, model=self.model_name)
        # Wrap the answer in a result-like object
        class Result:
            def __init__(self, text):
                self.text = text
        return [Result(answer)]

# Initialize memory and the agent.
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = DeepSeekLLM(model_name="deepseek-chat", temperature=0.7)

# For now, we don't add extra tools – you can add later (e.g., for web search or DB queries)
agent = initialize_agent(
    tools=[],  
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# Expose a simple run function.
def run_agent(prompt_text):
    return agent.run(prompt_text)
