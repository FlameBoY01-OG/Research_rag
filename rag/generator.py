import requests

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2"


def generate_answer(context: str, question: str) -> str:
    prompt = f"""
You are an AI assistant answering questions ONLY using the provided context.

If the answer is not present in the context, say:
"I could not find the answer in the provided document."

Context:
{context}

Question:
{question}

Answer:
"""

    response = requests.post(
        OLLAMA_GENERATE_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    response.raise_for_status()
    return response.json()["response"].strip()
