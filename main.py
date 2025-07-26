from app.utils import extract_text_from_pdf, chunk_text
from app.rag_pipeline import load_embedding_model, get_embeddings, generate_answer
from app.vector_store import store_vectors, search_vectors
from transformers import pipeline
from app.evaluation import evaluate_groundedness, evaluate_relevance

# === Step 1: Extract and Chunk Text ===
pdf_path = "data/hsc26_bangla.pdf"  # Make sure this exists
text = extract_text_from_pdf(pdf_path)
chunks = chunk_text(text, max_len=500)

print(f"✅ Extracted text and created {len(chunks)} chunks.")

# === Step 2: Load Embedding Model and Encode Chunks ===
embedding_model = load_embedding_model()
embeddings = get_embeddings(embedding_model, chunks)

# === Step 3: Store Vectors ===
store_vectors(embeddings, chunks)
print("✅ Stored vector index and chunks.")

# === Step 4: Load QA Model ===
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
print("✅ QA model loaded.")

# === Step 5: Ask a Question ===
query = "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
retrieved_chunks = search_vectors(query, embedding_model, k=3)

# Join retrieved chunks as context
context = " ".join(retrieved_chunks)
print("✅ Retrieved relevant chunks.")

# === Step 6: Generate Answer ===
result = qa_pipeline(question=query, context=context)
answer = result["answer"]

print("\nQuestion:", query)
print("Answer:", answer)


# Your existing vars: query, retrieved_chunks, answer

ground_score = evaluate_groundedness(answer, retrieved_chunks)
relevance_score = evaluate_relevance(query, retrieved_chunks)

print("\nEvaluation Metrics:")
print("✅ Groundedness Score:", ground_score)
print("✅ Relevance Score:", relevance_score)
