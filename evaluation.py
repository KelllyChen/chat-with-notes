import os
from rag import load_document, split_into_chunks, build_vector_store, create_qa_chain
from langchain.chat_models import ChatOpenAI
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall


file_path = "Week 1 Slides.pdf"  
file_type = ".pdf"

docs = load_document(file_path, file_type)
chunks = split_into_chunks(docs)
vectorstore = build_vector_store(chunks)

# Set up chain 
qa_chain = create_qa_chain(vectorstore)

# Prepare evaluation questions
evaluation_questions = [
    {
        "question": "What are the two roles each company need to have",
        "ground_truth": "deep-and-narrow engineers and full-cycle engineers",
    },
    {
        "question": "What does every business need to thrive?",
        "ground_truth": "Differentiation",
    },
    {
        "question": "What is WIDGET?",
        "ground_truth": "a model for how works gets done, and who does it. It isalso a way to understand your gifts",
    },
    {
        "question": "What does WIDGET stand for?",
        "ground_truth": "Wonder, Invention, Discrement, Galvanizing, Enablement, Tenacity",
    },
    {
        "question": "What motivate a company to hire a person?",
        "ground_truth": "Person who can help increase company's bottom-line profits, equity",
    },
]

# Evaluate each question
results = []

for sample in evaluation_questions:
    query = sample["question"]
    # Run through the chain
    response = qa_chain({"question": query})
    answer = response["answer"] if isinstance(response, dict) else response

    # Retrieve context manually
    retriever = vectorstore.as_retriever()
    retrieved_docs = retriever.get_relevant_documents(query)
    contexts = [doc.page_content for doc in retrieved_docs]

    # Collect data
    results.append({
        "question": query,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": sample["ground_truth"]
    })

data = Dataset.from_list(results)

# Run RAGAS Evaluation 
metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
evaluation_result = evaluate(data, metrics=metrics)

# Save Results 
df = evaluation_result.to_pandas()
df.to_csv("evaluation_result.csv", index=False)
