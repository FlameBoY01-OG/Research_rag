import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2"


def generate_answer(context: str, question: str, mode: str = "fact") -> str:
    if mode == "summary":
        system_prompt = (
            "You are a research assistant. "
            "Summarize the following context clearly and concisely. "
            "Do not add information that is not present."
        )
    elif mode == "explanation":
        system_prompt = (
            "You are a research assistant. "
            "Explain the answer clearly using the provided context only."
        )
    else:
        system_prompt = (
            "You are a research assistant. "
            "Answer the question using only the provided context. "
            "If the answer is not present, say you could not find it."
        )

    prompt = f"""
    {system_prompt}

    Context:
    {context}

    Question:
    {question}
    """

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]
