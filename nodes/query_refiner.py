from state import PipelineState
from model import llm

def query_refiner_node(state: PipelineState) -> PipelineState:
    messages = [
        {"role": "system", "content": "You rewrite vague or incomplete real estate queries into short, specific search queries. Output ONLY the improved query."},
        {"role": "user", "content": f"Original query: {state['query']}"}
    ]
    refined = llm.invoke(messages)
    state["query"] = refined.content.strip().strip('"')
    return state
