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
    num_docs: int


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
    docs = vector_store.similarity_search(state["query"], k=state["num_docs"])

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
st.title("🏠 Apartment Search")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

location = st.text_input("📍 Enter a location (e.g., Los Angeles, CA):")

with st.expander("🔎 Advanced Filters"):
    beds = st.selectbox("Beds", ["Any", 1, 2, 3, 4, 5])
    baths = st.selectbox("Baths", ["Any", 1, 2, 3, 4, 5])
    default_price_range: tuple = (1000, 3000)
    min_price, max_price = st.slider("Price Range ($)", 500, 5000, default_price_range)
    pets_allowed = st.checkbox("🐶 Pets Allowed")
    parking_included = st.checkbox("🚗 Parking Included")
    amenities = st.multiselect(
        "✨ Amenities",
        ["Gym", "Pool", "Laundry", "Elevator", "Air Conditioning", "Dishwasher"]
    )

num_docs = st.slider("📚 Number of documents to search", min_value=3, max_value=100, value=20)
query_parts = [f"{location} apartments"]

if beds != "Any":
    query_parts.append(f"{beds} bedrooms")

if baths != "Any":
    query_parts.append(f"{baths} bathrooms")

query_parts.append(f"rent from \${min_price} to \${max_price}")

if pets_allowed:
    query_parts.append("pet-friendly")

if parking_included:
    query_parts.append("parking available")

if amenities:
    query_parts.append("Amenities: " + ", ".join(amenities))

final_query = " ".join(query_parts)

st.write(f"🔍 **Search query:** {final_query}")

if st.button("🔍 Search Apartments"):
    if not location.strip():
        st.error("🚨 Please enter a valid location.")
    else:
        with st.spinner("Searching for apartments..."):
            websites = ["apartments.com"]
            graph_input = {
                "query": final_query,
                "websites": websites,
                "num_docs": num_docs
            }
            response = agentic_rag_pipeline.invoke(graph_input)['answer']

        # 정확한 결과가 없을 때
        if "No exact match" in response or "couldn’t find any listings" in response:
            st.warning("⚠️ No exact matches found for your criteria.")

            choice = st.radio(
                "Would you like to see similar listings or adjust your search?",
                ["✨ See similar listings", "🔄 Adjust search criteria"]
            )

            if choice == "✨ See similar listings":
                with st.spinner("Looking for similar listings..."):
                    relaxed_query = f"similar apartments near {location}, {beds} bed, {baths} bath, \${min_price}-\${max_price}"
                    relaxed_graph_input = {
                        "query": relaxed_query,
                        "websites": websites,
                        "num_docs": num_docs + 5
                    }
                    relaxed_response = agentic_rag_pipeline.invoke(relaxed_graph_input)['answer']
                st.markdown(relaxed_response)
                st.session_state['messages'].append({'role': 'user', 'content': relaxed_query})
                st.session_state['messages'].append({'role': 'assistant', 'content': relaxed_response})

            elif choice == "🔄 Adjust search criteria":
                st.info("Please adjust your criteria and search again.")

        else:
            st.markdown(response)
            st.session_state['messages'].append({'role': 'user', 'content': final_query})
            st.session_state['messages'].append({'role': 'assistant', 'content': response})
