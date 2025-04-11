import os
from decouple import Config, RepositoryEnv
from pathlib import Path
from typing_extensions import List, TypedDict

import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import Runnable
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph
import re

# ------------------------------------------------------------------------------
# Environment & Global Setup
# ------------------------------------------------------------------------------

root_dir = Path().resolve()
config = Config(RepositoryEnv(root_dir / '.env'))  # Explicitly load .env

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = config('LANGSMITH_API_KEY')

llm = ChatOllama(model="llama3.2", temperature=0)
embeddings = OllamaEmbeddings(model="llama3.2")
vector_store = InMemoryVectorStore(embeddings)

class PipelineState(TypedDict):
    query: str
    websites: List[str]
    context: List[Document]
    top_docs: List[Document]
    messages: str
    answer: str

# ------------------------------------------------------------------------------
# DuckDuckGo Loader
# ------------------------------------------------------------------------------

def duckduckgo_docs(query: str, websites: List[str]) -> List[Document]:
    search = DuckDuckGoSearchResults(output_format="list")
    results = [
        item
        for website in websites
        for item in search.invoke(f"{query} site:{website}")
    ]

    # 🔍 Print raw search results for inspection
    print("\n===== RAW DUCKDUCKGO RESULTS =====")
    for r in results:
        print(f"- Title: {r.get('title')}")
        print(f"  Link: {r.get('link')}")
        print(f"  Snippet: {r.get('snippet')}\n")

    return [
        Document(
            page_content=f"{entry['snippet']}\nLink: {entry['link']}",
            metadata={"source": entry["link"]}
        )
        for entry in results
    ]

def duckduckgo_loader_node(state: PipelineState) -> PipelineState:
    docs = duckduckgo_docs(state['query'], state['websites'])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    all_splits = text_splitter.split_documents(docs)
    vector_store.add_documents(documents=all_splits)
    return state

# ------------------------------------------------------------------------------
# Retriever with Bed/Bath Filter
# ------------------------------------------------------------------------------

def parse_unit_filter(query: str):
    filters = {}
    if match := re.search(r"(\d+)\s*bed", query.lower()):
        filters["bed"] = int(match.group(1))
    if match := re.search(r"(\d+)\s*bath", query.lower()):
        filters["bath"] = int(match.group(1))
    return filters

def retriever_node(state: PipelineState):
    filters = parse_unit_filter(state["query"])
    docs = vector_store.similarity_search(state["query"])

    def matches_unit(doc):
        text = doc.page_content.lower()
        if "bed" in filters and f"{filters['bed']} bed" not in text:
            return False
        if "bath" in filters and f"{filters['bath']} bath" not in text:
            return False
        return True

    filtered_docs = [doc for doc in docs if matches_unit(doc)]
    state["context"] = filtered_docs or docs  # fallback to original if all filtered out

    # Optional debug logs
    print("Parsed filters:", filters)
    print("Retrieved docs:", len(docs))
    print("Filtered docs:", len(filtered_docs))

    return state

# ------------------------------------------------------------------------------
# Generator Node
# ------------------------------------------------------------------------------

def generator_node(state: PipelineState):
    if not state["context"]:
        state["answer"] = (
            "Sorry, I couldn’t find any listings that match all your criteria exactly. "
            "Try modifying your query or removing some filters."
        )
        return state

    docs_content = "\n\n".join(
        f"{doc.page_content}\n[Source]({doc.metadata.get('source', '')})"
        for doc in state["context"]
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise and strict real estate recommendation engine. "
                "You must select **only one** apartment listing that matches the user’s search query **exactly** and use information strictly from the provided crawled context. "
                "You are not allowed to guess, infer, or assume anything. "
                "Only recommend listings that meet all parts of the user's query as explicitly stated in the context. "
                "Do NOT relax the match. For example, if the query asks for '2 bed, 2 bath', you must find a listing that explicitly includes both '2 bed' and '2 bath' in the same description. "
                "Do not recommend listings that say '1 bath' or are missing any required detail. "
                "Do NOT state assumptions like 'likely a typo' or make excuses for mismatched data. "
                "The output must include exactly and only what's in the crawled context: apartment name, address, unit type (must match), and the listing URL (must come from the same listing block). "
                "Never invent or adjust any details. Omit any field that is not present. Do not hallucinate or rationalize missing or conflicting data."
            )
        },
        {
            "role": "user",
            "content": (
                f"{state['query']}\n\n"
                "Below is the crawled context from apartments.com:\n\n"
                f"{docs_content}\n\n"
                "Based **only** on this context, recommend **one** specific apartment listing that matches the query **exactly** (e.g., 2 bed, 2 bath, located in LA, CA).\n\n"
                "If such listing exists, output the result clearly in the following format:\n\n"
                "🏢 Apartment Name: <name>\n"
                "📍 Address: <address or 'Not stated'>\n"
                "🛏️ Unit Type: <exact unit type as found in text>\n"
                "✨ Amenities: <if any>\n"
                "🔗 URL: <valid link>\n\n"
                "If no listing matches the request exactly, say:\n"
                "\"⚠️ No exact match for '2 bed, 2 bath' was found in the context.\""
            )
        }
    ]
    response = llm.invoke(messages)
    state["answer"] = response.content
    return state

# ------------------------------------------------------------------------------
# LangGraph: Define and Compile
# ------------------------------------------------------------------------------

graph = StateGraph(PipelineState)
graph.add_node("duckduckgo_loader", duckduckgo_loader_node)
graph.add_node("retriever", retriever_node)
graph.add_node("generator", generator_node)

graph.set_entry_point("duckduckgo_loader")
graph.add_edge("duckduckgo_loader", "retriever")
graph.add_edge("retriever", "generator")
graph.set_finish_point("generator")

agentic_rag_pipeline = graph.compile()

def get_session_history(session_id: str):
    return InMemoryChatMessageHistory()

chat_with_history: Runnable = RunnableWithMessageHistory(
    agentic_rag_pipeline,
    get_session_history,
    input_messages_key="query",
    history_messages_key="messages",
)

# ------------------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------------------

st.set_page_config(page_title="LLM-REALTOR")
st.title("🏠 LLM REALTOR")

user_input = st.chat_input("Ask me anything!")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if user_input:
    st.session_state['messages'].append({'role': 'user', 'content': user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    websites = ["apartments.com"]
    graph_input = {"query": user_input, "websites": websites}
    response = agentic_rag_pipeline.invoke(graph_input)['answer']

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state['messages'].append({'role': 'assistant', 'content': response})
