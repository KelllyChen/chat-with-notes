import streamlit as st
import tempfile
import os
from rag import load_document, split_into_chunks, build_vector_store, create_qa_chain, answer_query, generate_quiz_from_docs, judge_answer

# === Session Initialization ===
st.set_page_config(page_title="Chat with Your Notes", layout="centered")
st.title("📄 Chat with Your Notes")

for key in [
    "chat_history", "qa_chain", "quiz", "quiz_questions", "quiz_answers",
    "user_answers", "question_index", "docs", "query_submitted", "last_query"
]:
    if key not in st.session_state:
        st.session_state[key] = [] if "history" in key or "answers" in key or "questions" in key else None

# === File Upload ===
uploaded_file = st.file_uploader("Upload a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])

# === Chat Input ===
new_query = st.text_input("Ask a question about your notes", key="query_box")

if new_query and (
    "last_query" not in st.session_state or new_query != st.session_state["last_query"]
):
    st.session_state["raw_query"] = new_query
    st.session_state["last_query"] = new_query
    st.session_state.query_submitted = False

# === Document Processing ===
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
        st.session_state.docs = docs

        st.success("Document processed! Ask any question above 👆")

# === RAG Q&A ===
if "raw_query" in st.session_state and not st.session_state.query_submitted and st.session_state.qa_chain:
    query = st.session_state["raw_query"]

    if len(st.session_state.chat_history) >= 1:
        last_user_question = st.session_state.chat_history[-1][0]
        full_query = f"In the context of this earlier question: '{last_user_question}'\nFollow-up: {query}"
    else:
        full_query = query

    with st.spinner("Thinking..."):
        answer = answer_query(st.session_state.qa_chain, full_query)
        st.session_state.chat_history.append((query, answer))
        st.session_state.query_submitted = True

# === Display Chat History ===
st.markdown("### 💬 Chat")

for q, a in st.session_state.chat_history:
    with st.container():
        st.markdown(f"""
<div style='display: flex; align-items: flex-start; margin-bottom: 10px;'>
    <div style='background-color: #e0f7fa; padding: 10px 15px; border-radius: 10px; max-width: 80%;'>
        <b>You:</b><br>{q}
    </div>
</div>
""", unsafe_allow_html=True)

        st.markdown(f"""
<div style='display: flex; justify-content: flex-end; margin-bottom: 20px;'>
    <div style='background-color: #f1f8e9; padding: 10px 15px; border-radius: 10px; max-width: 80%;'>
        <b>AI:</b><br>{a}
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# === Start One-by-One Quiz ===
if st.session_state.qa_chain and st.session_state.docs:
    if st.button("📝 Start Quiz (One Question at a Time)"):
        with st.spinner("Generating quiz..."):
            quiz_raw = generate_quiz_from_docs(st.session_state.docs)
            lines = quiz_raw.strip().split("\n")
            questions, answers = [], []
            buffer = ""

            for line in lines:
                line = line.strip()
                if line.lower().startswith("answer:"):
                    questions.append(buffer.strip())
                    answers.append(line.replace("Answer:", "").strip())
                    buffer = ""
                else:
                    buffer += " " + line

            st.session_state.quiz_questions = questions
            st.session_state.quiz_answers = answers
            st.session_state.user_answers = []
            st.session_state.question_index = 0

# === Quiz Interaction ===
if (
    st.session_state.quiz_questions and
    st.session_state.question_index < len(st.session_state.quiz_questions)
):
    i = st.session_state.question_index
    st.markdown(f"### ✏️ Quiz Practice")
    st.markdown(f"**{st.session_state.quiz_questions[i]}**")
    user_input = st.text_input("Your Answer:", key=f"user_input_{i}")

    if st.button("✅ Submit Answer"):
        st.session_state.user_answers.append(user_input)
        st.session_state.question_index += 1
        st.rerun()

# === Show Quiz Results ===
if (
    st.session_state.quiz_questions and
    st.session_state.question_index == len(st.session_state.quiz_questions)
):
    st.markdown("### ✅ Quiz Complete!")
    score = 0
    for i, (q, user, correct) in enumerate(zip(
        st.session_state.quiz_questions,
        st.session_state.user_answers,
        st.session_state.quiz_answers
    )):
        st.markdown(f"**{q}**")
        st.markdown(f"Your Answer: {user}")
        st.markdown(f"✅ Correct Answer: {correct}")
        

        result = judge_answer(user, correct, q)
        if result:
            score += 1
            st.markdown("🟢 **Correct!**")
        else:
            st.markdown("🔴 **Incorrect**")

        st.markdown("---")

    st.markdown(f"🎯 Score: **{score} / {len(st.session_state.quiz_questions)}**")

    if st.button("🔁 Restart Quiz"):
        st.session_state.quiz_questions = []
        st.session_state.quiz_answers = []
        st.session_state.user_answers = []
        st.session_state.question_index = 0
        st.rerun()


