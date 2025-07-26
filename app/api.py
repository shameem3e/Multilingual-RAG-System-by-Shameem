from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import pipeline

from app.utils import extract_text_from_pdf, chunk_text
from app.rag_pipeline import load_embedding_model, get_embeddings
from app.vector_store import store_vectors, search_vectors

import os

app = FastAPI()

# === Load models on startup ===
embedding_model = load_embedding_model()
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# === Load and index document ===
pdf_path = "data/hsc26_bangla.pdf"
text = extract_text_from_pdf(pdf_path)
chunks = chunk_text(text, max_len=500)
embeddings = get_embeddings(embedding_model, chunks)
store_vectors(embeddings, chunks)

# === Request body ===
class QueryRequest(BaseModel):
    question: str

@app.post("/query")
def query_rag(request: QueryRequest):
    query = request.question
    retrieved_chunks = search_vectors(query, embedding_model, k=3)
    context = " ".join(retrieved_chunks)
    result = qa_pipeline(question=query, context=context)
    return {
        "question": query,
        "answer": result["answer"]
    }
