"""Ingestion : extraction PDF, chunking, embeddings, index FAISS, recherche."""
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

EMB_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # FR + EN
_model = None

def _emb():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMB_MODEL)
    return _model

def extract_text(file) -> str:
    """Extrait le texte d'un PDF (chemin ou fichier uploadé)."""
    reader = PdfReader(file)
    return "\n".join((p.extract_text() or "") for p in reader.pages)

def chunk_text(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    """Découpe en chunks de `size` caractères avec chevauchement."""
    text = " ".join(text.split())
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i + size])
        i += size - overlap
    return chunks

class Index:
    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        vecs = _emb().encode(chunks, normalize_embeddings=True, show_progress_bar=False)
        self.index = faiss.IndexFlatIP(vecs.shape[1])  # produit scalaire = cosinus
        self.index.add(np.asarray(vecs, dtype="float32"))

    def search(self, query: str, k: int = 4):
        q = _emb().encode([query], normalize_embeddings=True)
        scores, ids = self.index.search(np.asarray(q, dtype="float32"), k)
        return [(int(i), float(s), self.chunks[i]) for i, s in zip(ids[0], scores[0]) if i >= 0]
