# 📚 Multilingual RAG System

This is a lightweight Retrieval-Augmented Generation (RAG) system that supports Bangla and English. It extracts content from PDFs, generates dense embeddings, retrieves relevant chunks, and uses a question-answering model to respond to user queries. Includes an interactive FastAPI endpoint and evaluation module.

## ⚙️ Setup Guide

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/shameem3e/Multilingual-RAG-System-by-Shameem.git
cd Multilingual-RAG-System-by-Shameem

```
### **2️⃣ Create Virtual Environment(optional but recommended)**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

```
### **3️⃣ Install Requirements**
```bash
pip install -r requirements.txt

```
### **4️⃣  Add your PDF to the data/ directory**
   
   data/
   
   └── hsc26_bangla.pdf
### **5️⃣ Run the API server**
```bash
uvicorn app.api:app --reload
Access the interactive UI at http://127.0.0.1:8000/docs
```

## Tools & Libraries Used
| Tool / Library            | Purpose                                  |
| ------------------------- | ---------------------------------------- |
| FastAPI                   | API interface                            |
| pdfplumber                | Extract text from PDF                    |
| sentence-transformers     | Generate dense multilingual embeddings   |
| FAISS                     | Store and retrieve vector indexes        |
| Hugging Face Transformers | Question-answering model (RoBERTa)       |
| scikit-learn              | Cosine similarity metrics for evaluation |

## Sample Queries & Output
✓ Bangla Query

Input: অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?

Output: শম্ভুনাথ

## API Documentation
POST /query

Request:
{
  "question": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
}
Response:

{
  "question": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
  "answer": "শম্ভুনাথ"
}

Interactive Swagger UI available at: `http://127.0.0.1:8000/docs`

## Evaluation Metrics
| Metric       | Description                                                      |
| ------------ | ---------------------------------------------------------------- |
| Groundedness | Cosine similarity between generated answer and retrieved context |
| Relevance    | Average similarity between query and top-k retrieved chunks      |

Example:

* Groundedness: 0.91
* Relevance: 0.84

## ❓ Reflection Questions

Q: What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?

A: I used pdfplumber because it preserves layout and handles Unicode text (important for Bangla). Some pages had formatting noise like misaligned text and missing line breaks, but overall, it worked well for text extraction.

Q: What chunking strategy did you choose (e.g., paragraph-based, sentence-based, character limit)? Why do you think it works well for semantic retrieval?

A: I used a sentence-based chunking strategy with a max length of 500 characters. This provides enough context for semantic retrieval without losing the meaning of the passage. Smaller chunks help with fine-grained retrieval.

Q: What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?

A: I used paraphrase-multilingual-MiniLM-L12-v2 from SentenceTransformers. It supports Bangla and English, is lightweight, and optimised for semantic similarity tasks. It captures the meaning of text at the sentence level in a shared multilingual embedding space.

Q: How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?

A: I use cosine similarity between the embedding of the query and stored chunk embeddings. This is efficient and interpretable. FAISS allows fast retrieval from large datasets.

Q: How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague or missing context?

A: The use of multilingual sentence-level embeddings ensures a consistent vector space. If the query is vague, the model may retrieve less relevant chunks or produce a generic answer. This could be improved with better query rewriting or prompt engineering.

Q: Do the results seem relevant? If not, what might improve them (e.g., better chunking, better embedding model, larger document)?

A: Yes, the results are mostly relevant. However, improvements could include better OCR for low-quality PDFs, refining the chunking to preserve semantic boundaries more carefully, or switching to a stronger embedding model like LaBSE or multilingual-e5.

## 🛠 Tech Stack
* Python 3.8+
* Hugging Face Transformers
* FAISS (Facebook AI Similarity Search)

## 🚀 Future Improvements
* Add summarization of retrieved results
* Integrate speech-to-text & text-to-speech
* Add web interface (Flask/Streamlit)

## 👨‍💻 Author
[MD. Shameem Ahammed](https://sites.google.com/view/shameem3e)
Graduate Student, AI & ML Enthusiast

---

If you want, I can also add a **"Preview" section** with a screenshot of your terminal chatbot in action so the GitHub page looks more engaging. Would you like me to make that?
