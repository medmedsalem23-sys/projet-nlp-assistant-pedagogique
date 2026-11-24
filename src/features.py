"""Résumés, quiz et chatbot RAG."""
import json
import re
from .llm import ask

LEVELS = {
    "court": "un résumé très court : 3 phrases maximum et 50 mots maximum",
    "moyen": "un résumé moyen : un paragraphe d'environ 150 mots",
    "détaillé": "un résumé détaillé avec titres et puces (environ 400 mots)",
}

NO_LATEX = "N'utilise pas de LaTeX : écris les formules en texte simple (ex. w = w + eta*(c-o)*x)."

def summarize(chunks: list[str], level: str = "moyen", section_chars: int = 6000) -> str:
    """Map-reduce : on résume chaque section, puis on résume les résumés."""
    text = " ".join(chunks)
    sections = [text[i:i + section_chars] for i in range(0, len(text), section_chars)]
    partials = [ask(f"Résume fidèlement ce passage de cours en gardant les idées clés. {NO_LATEX}\n\n{s}")
                for s in sections]
    joined = "\n\n".join(partials)
    return ask(f"À partir de ces résumés partiels d'un cours, rédige {LEVELS[level]}. "
               f"N'invente rien, reste fidèle au contenu. {NO_LATEX}\n\n{joined}")

def generate_quiz(chunks: list[str], n: int = 5, difficulty: str = "moyen") -> list[dict]:
    """Génère n QCM au format JSON."""
    step = max(1, len(chunks) // n)
    context = "\n\n".join(chunks[::step][:n * 2])[:8000]
    prompt = f"""À partir du cours ci-dessous, génère {n} QCM de difficulté "{difficulty}".
Réponds UNIQUEMENT avec un tableau JSON, sans texte autour, de la forme :
[{{"question": "...", "choices": ["A", "B", "C", "D"], "answer": 0, "explanation": "..."}}]
("answer" = indice de la bonne réponse, 0 à 3). Les questions doivent porter uniquement sur le cours.

COURS :
{context}"""
    raw = ask(prompt, temperature=0.4)
    match = re.search(r"\[.*\]", raw, re.S)
    return json.loads(match.group(0))

def chat_answer(index, question: str, k: int = 4):
    """RAG : récupère les passages, répond en citant [1], [2]... ou dit qu'il ne sait pas."""
    hits = index.search(question, k)
    context = "\n\n".join(f"[{n + 1}] {t}" for n, (_, _, t) in enumerate(hits))
    prompt = f"""Réponds à la question en t'appuyant UNIQUEMENT sur les passages ci-dessous.
Cite les passages utilisés entre crochets, par exemple [1]. Si la réponse n'est pas dans les passages,
réponds exactement : "Je ne trouve pas la réponse dans le document."

PASSAGES :
{context}

QUESTION : {question}"""
    return ask(prompt), hits
