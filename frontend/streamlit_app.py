import streamlit as st
import sys
import os

# Add PROJECT ROOT to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from main import RAGService   # ✅ NOW THIS WILL WORK

st.set_page_config(
    page_title="Employee Handbook RAG",
    page_icon="📘",
    layout="centered"
)

st.title("📘 Upgrad Employee Assistant")
st.caption("Ask questions from company handbook using RAG")

@st.cache_resource
def load_rag():
    return RAGService()

rag = load_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask a question")

if query:
    st.session_state.messages.append(
        {"role": "user", "content": query}
    )
    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("Thinking..."):
        answer = rag.ask(query)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
    with st.chat_message("assistant"):
        st.markdown(answer)
