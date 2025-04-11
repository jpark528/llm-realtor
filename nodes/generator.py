from state import PipelineState
from model import llm
def generator_node(state: PipelineState):
    # Placeholder for actual generation logic
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
                "Based **only** on this context, recommend **one** specific apartment listing that matches the query **exactly** (2 bed, 2 bath, located in LA, CA). "
                "Include only the following fields, exactly as found in the same listing: Apartment Name, Address, Apartment Unit Type, Amenities (if any), and URL Link."
            )
        }
    ]
    response = llm.invoke(messages)
    state["answer"] = response.content
    return state
