import faiss
import numpy as np
import pickle
import os

def store_vectors(embeddings, chunks, save_dir="vector_store"):
    """
    Stores the FAISS index and chunks to disk.
    """
    os.makedirs(save_dir, exist_ok=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    # Save index
    index_path = os.path.join(save_dir, "vector.index")
    faiss.write_index(index, index_path)

    # Save chunks
    chunk_path = os.path.join(save_dir, "chunks.pkl")
    with open(chunk_path, "wb") as f:
        pickle.dump(chunks, f)

def search_vectors(query, model, save_dir="vector_store", k=3):
    """
    Searches for top-k relevant chunks based on the query.
    """
    index_path = os.path.join(save_dir, "vector.index")
    chunk_path = os.path.join(save_dir, "chunks.pkl")

    if not os.path.exists(index_path) or not os.path.exists(chunk_path):
        raise FileNotFoundError("Vector index or chunks file not found. Run the vector store step first.")

    # Load saved data
    index = faiss.read_index(index_path)
    with open(chunk_path, "rb") as f:
        chunks = pickle.load(f)

    # Encode query
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), k)
    return [chunks[i] for i in I[0]]