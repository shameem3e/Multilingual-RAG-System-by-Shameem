# app/rag_pipeline.py

from sentence_transformers import SentenceTransformer
import openai


def load_embedding_model():
    """
    Loads the multilingual embedding model.
    """
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return model


def get_embeddings(model, chunks):
    """
    Generates dense embeddings for the list of chunks using the given model.
    """
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings


from transformers import pipeline

# Load once globally or in your main.py
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

def generate_answer(query, retrieved_chunks, chat_history=None):
    """
    Uses HuggingFace QA model to answer based on retrieved chunks.
    Does not require OpenAI.
    """
    context = " ".join(retrieved_chunks)
    result = qa_pipeline(question=query, context=context)
    return result['answer']
