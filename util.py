from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import Runnable
from langchain_core.chat_history import InMemoryChatMessageHistory
from nodes.duckduckgo_loader import agentic_rag_pipeline
def get_session_history(session_id: str):
    return InMemoryChatMessageHistory()

# Wrap just the `llm` or entire pipeline if you want history across pipeline
chat_with_history: Runnable = RunnableWithMessageHistory(
    agentic_rag_pipeline,
    get_session_history,
    input_messages_key="query",  # This is what the user types
    history_messages_key="messages",  # This needs to be passed if you want history
)
