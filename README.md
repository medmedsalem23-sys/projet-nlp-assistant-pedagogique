# Assistant pédagogique (résumés, quiz, chatbot RAG)

Projet NLP – Master DSSD. Téléversez un cours PDF, puis obtenez des résumés (court/moyen/détaillé),
des QCM avec correction et un chatbot qui répond en citant les passages du cours.

## Prérequis
- Python 3.10+
- Une clé d'API pour un LLM compatible OpenAI (Groq offre un accès gratuit)

## Installation
```bash
pip install -r requirements.txt
```

## Configuration (exemple avec Groq)
```bash
export LLM_API_KEY="votre_cle"
export LLM_BASE_URL="https://api.groq.com/openai/v1"
export LLM_MODEL="openai/gpt-oss-20b"
```
(Sous Windows PowerShell : `$env:LLM_API_KEY="votre_cle"`.)

## Exécution
```bash
streamlit run app.py
```

## Pipeline
PDF → extraction (pypdf) → chunking (taille/chevauchement réglables) → embeddings
(paraphrase-multilingual-MiniLM-L12-v2) → index FAISS → retrieval top-k → LLM.
- Résumés : map-reduce (résumé par section puis résumé des résumés)
- Quiz : sortie JSON (question, 4 choix, réponse, explication)
- Chat : RAG avec citations [n] et réponse « je ne sais pas » hors document

## Structure
- `app.py` : interface Streamlit
- `src/rag.py` : extraction, chunking, embeddings, FAISS
- `src/features.py` : résumés, quiz, chatbot
- `src/llm.py` : appel du LLM
