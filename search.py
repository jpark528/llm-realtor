from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from state import PipelineState
from typing_extensions import List, TypedDict

def duckduckgo_docs(query: str, websites: List[str]) -> List[Document]:
    """
    Search DuckDuckGo for a query restricted to specific websites.

    Args:
        query (str): The search query string.
        websites (List[str]): A list of website domains to search within.

    Returns:
        List[Document]: A list of LangChain Document objects created from search result snippets.
    """
    search = DuckDuckGoSearchResults(output_format="list")
    results = [
        item
        for website in websites
        for item in search.invoke(f"{query} site:{website}")
    ]
    return [
        Document(
            page_content=entry["snippet"],
            metadata={"source": entry["link"]}
        )
        for entry in results
    ]
