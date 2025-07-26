import pdfplumber
import re


def extract_text_from_pdf(pdf_path):
    """
    Extracts text from all pages of a given PDF file.
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def chunk_text(text, max_len=500):
    """
    Splits text into semantically meaningful chunks of up to `max_len` characters.
    """
    sentences = re.split(r'[।?!]', text)
    chunks = []
    chunk = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(chunk) + len(sentence) <= max_len:
            chunk += sentence + "।"
        else:
            chunks.append(chunk.strip())
            chunk = sentence + "।"
    if chunk:
        chunks.append(chunk.strip())
    return chunks
