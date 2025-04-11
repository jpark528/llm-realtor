from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from model import vector_store
from state import PipelineState
from search import duckduckgo_docs
from langgraph.graph import StateGraph
from retriever import retriever_node
from generator import generator_node

def duckduckgo_loader_node(state: PipelineState) -> PipelineState:
    """
    Load DuckDuckGo search results into a vector store after chunking text.

    Args:
        state (PipelineState): A pipeline state dictionary containing:
            - 'query': str, the search query.
            - 'websites': List[str], the domains to search.

    Returns:
        PipelineState: The updated pipeline state.
    """
    docs = duckduckgo_docs(state['query'], state['websites'])

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    all_splits = text_splitter.split_documents(docs)

    vector_store.add_documents(documents=all_splits)

    return state

# Graph definition
graph = StateGraph(PipelineState)
graph.add_node("duckduckgo_loader", duckduckgo_loader_node)
graph.add_node("retriever", retriever_node)
graph.add_node("generator", generator_node)

graph.set_entry_point("duckduckgo_loader")
graph.add_edge("duckduckgo_loader", "retriever")
graph.add_edge("retriever", "generator")
graph.set_finish_point("generator")

# Compile Graph
agentic_rag_pipeline = graph.compile()
