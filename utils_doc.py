# doc_utils.py
from typing import List, Tuple
import fitz  # pymupdf
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    texts = []
    for page in doc:
        texts.append(page.get_text("text"))
    return "\n".join(texts)

def clean_text(t: str) -> str:
    t = t.replace("\r"," ")
    t = re.sub(r'\s+', ' ', t)
    return t.strip()

def chunk_text(text: str, chunk_size:int=800, overlap:int=100) -> List[str]:
    text = clean_text(text)
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

class Retriever:
    def __init__(self, docs: List[str]):
        # docs: list of strings (chunks)
        self.chunks = docs
        # fall back vectorizer config
        self.vectorizer = TfidfVectorizer().fit(self.chunks)
        self.doc_vectors = self.vectorizer.transform(self.chunks)

    def query(self, q: str, top_k:int=3):
        qv = self.vectorizer.transform([q])
        sims = cosine_similarity(qv, self.doc_vectors).flatten()
        ids = sims.argsort()[::-1][:top_k]
        results = [(int(i), float(sims[i]), self.chunks[i]) for i in ids]
        return results
