"""Appel du LLM via une API compatible OpenAI (Groq, OpenAI, Mistral, Ollama...)."""
import os
from openai import OpenAI

# Variables d'environnement :
#   LLM_API_KEY, LLM_BASE_URL (ex: https://api.groq.com/openai/v1), LLM_MODEL
_client = None

def ask(prompt: str, system: str = "Tu es un assistant pédagogique précis.", temperature: float = 0.2) -> str:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["LLM_API_KEY"],
                         base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"))
    r = _client.chat.completions.create(
        model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        temperature=temperature,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}])
    return r.choices[0].message.content
