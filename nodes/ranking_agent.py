from state import PipelineState
from model import llm

def ranking_agent_node(state: PipelineState) -> PipelineState:
    doc_list = "\n\n".join(
        f"[{i}] {doc.page_content}\nURL: {doc.metadata.get('source', '')}"
        for i, doc in enumerate(state["context"])
    )
    messages = [
        {"role": "system", "content": "You are a real estate ranking assistant."},
        {"role": "user", "content": f"Query: {state['query']}\n\n{doc_list}"}
    ]
    llm.invoke(messages)
    state["top_docs"] = state["context"][:3]
    return state
