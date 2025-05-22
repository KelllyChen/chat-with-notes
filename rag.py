import os
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

def load_document(file_path, file_type):
    if file_type == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_type == ".docx":
        loader = Docx2txtLoader(file_path)
    elif file_type == ".txt":
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsopported file type")
    return loader.load()

def split_into_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

def build_vector_store(docs):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(docs, embeddings)

def create_qa_chain(vectorstore):
    llm = ChatOpenAI(temperature=0, max_tokens=500)

    # Create memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    
    prompt_template = PromptTemplate(
    input_variables=["chat_history", "question", "context"],
    template="""
    You are an AI assistant helping the user understand their uploaded document.
    Use the following retrieved context to answer the user's question.

    Conversation so far:
    {chat_history}

    Context:
    {context}

    Question: {question}
    Answer:"""
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template}
    )

def answer_query(chain, query):
    return chain.run({"question": query})


def generate_quiz_from_docs(docs):
    text = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""
You are a tutor. Based on the following document, generate exactly 5 short-answer quiz questions.
After each question, also provide the correct answer.

Document:
{text}

Format your output like:
1. [Question text]  
Answer: [Short correct answer]

2. ...
    """
    llm = ChatOpenAI(temperature=0.3)
    response = llm.predict(prompt)
    return response


def judge_answer(user_answer: str, correct_answer: str, question: str) -> bool:
    prompt = f"""
You are an AI quiz grader.

Question: {question}
Correct Answer: {correct_answer}
Student's Answer: {user_answer}

Decide if the student's answer demonstrates the same core idea or key facts as the correct answer.

Only respond with one word: Correct or Incorrect.
Be strict: If the student’s answer is vague, off-topic, incomplete, or uses the wrong concept, say Incorrect.
"""
    llm = ChatOpenAI(temperature=0, max_tokens=5)
    result = llm.predict(prompt).strip().lower()
    return result == "correct"
