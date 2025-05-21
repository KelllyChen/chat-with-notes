# app.py

import streamlit as st
import tempfile
import os
from rag import load_document, split_into_chunks, build_vector_store, create_qa_chain, answer_query

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


st.set_page_config(page_title="Chat with Your Notes", layout="centered")
st.title("📄 Chat with Your Notes")

uploaded_file = st.file_uploader("Upload a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])
query = st.text_input("Ask a question about your notes")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if uploaded_file:
    with st.spinner("Processing your document..."):
        suffix = os.path.splitext(uploaded_file.name)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        docs = load_document(tmp_path, suffix)
        chunks = split_into_chunks(docs)
        vectorstore = build_vector_store(chunks)
        st.session_state.qa_chain = create_qa_chain(vectorstore)

        st.success("Document processed! Ask any question above 👆")

if query:
    st.session_state["raw_query"] = query

if query and st.session_state.qa_chain:
    if len(st.session_state.chat_history) >= 1:
        last_user_question = st.session_state.chat_history[-1][0]
        query = f"In the context of this earlier question: '{last_user_question}'\nFollow-up: {query}"

    with st.spinner("Thinking..."):
        answer = answer_query(st.session_state.qa_chain, query)

        # Save the original question (not the rephrased one) for cleaner chat display
        st.session_state.chat_history.append((st.session_state["raw_query"], answer))

for q, a in st.session_state.chat_history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**AI:** {a}")

