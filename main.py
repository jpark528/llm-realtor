import streamlit as st
from nodes.duckduckgo_loader import agentic_rag_pipeline
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
    websites=["apartments.com"]
    graph_input = {"query": user_input, "websites":websites}
    response = agentic_rag_pipeline.invoke(graph_input)['answer']

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state['messages'].append({'role': 'assistant', 'content': response})
