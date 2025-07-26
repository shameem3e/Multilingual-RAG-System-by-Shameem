from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

def evaluate_groundedness(answer, retrieved_chunks, model=None):
    """
    Checks whether the generated answer is semantically present in the retrieved context.
    Returns cosine similarity score between the answer and context.
    """
    if model is None:
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    context = " ".join(retrieved_chunks)
    embeddings = model.encode([answer, context])
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return round(score, 4)

def evaluate_relevance(query, retrieved_chunks, model=None):
    """
    Measures semantic similarity between query and each retrieved chunk.
    Returns average cosine similarity across top-k chunks.
    """
    if model is None:
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    query_emb = model.encode([query])
    chunk_embs = model.encode(retrieved_chunks)

    similarities = cosine_similarity(query_emb, chunk_embs)[0]
    avg_score = np.mean(similarities)
    return round(avg_score, 4)
