import os
import requests
from typing import Optional

OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")
OLLAMA_GENERATE = OLLAMA_BASE.rstrip("/") + "/api/generate"
DEFAULT_MODEL = os.getenv("OLLAMA_GEN_MODEL", "llama3.2:latest")


def _role_system_prompt(role: str) -> str:
    """
    Return a short system-style instruction depending on role.
    """
    r = role.lower() if role else "student"
    if r == "student":
        return "You are an assistant explaining technical material in a clear, simple, and didactic way. Use short examples and avoid unnecessary jargon."
    if r == "researcher":
        return "You are an expert researcher. Be precise, technical, and cite relevant details. Use domain terminology and be concise but complete."
    if r == "reviewer":
        return "You are a critical reviewer. Identify strengths, weaknesses, assumptions, and provide constructive criticism and possible improvements."
    # fallback
    return "You are an assistant. Answer clearly and helpfully."


def _mode_instructions(mode: Optional[str]) -> str:
    """
    Provide instructions tailored to the mode of generation.
    mode can be 'qa', 'summarize', 'compare', etc.
    """
    m = (mode or "qa").lower()
    if m == "summarize":
        return (
            "Produce a structured summary containing: Problem statement, Key methods, Main contributions, "
            "and Conclusions. Keep it compact and use bullet points where helpful."
        )
    if m == "compare":
        return (
            "Compare the given documents. Provide: Goals, Methods, Strengths, Weaknesses, and Key differences. "
            "Use clear subheadings for each section."
        )
    # default / 'qa'
    return (
        "Answer the question using *only* the provided context. If the answer is not present in the context, "
        "say you could not find the answer in the provided documents. When possible, point to the relevant context "
        "by including the bracketed citation exactly as it appears in the context (e.g. [paper.pdf | page 3])."
    )


def generate_answer(
    context: str,
    question: str,
    mode: Optional[str] = "qa",
    role: str = "student",
    model: Optional[str] = None,
    max_tokens: int = 512,
    timeout: int = 60,
) -> str:

    if model is None:
        model = DEFAULT_MODEL

    system_prompt = _role_system_prompt(role)
    instr = _mode_instructions(mode)

    # Compose a clear prompt: system + context + instructions + question
    prompt = (
        f"{system_prompt}\n\n"
        f"Context:\n{context}\n\n"
        f"Instructions:\n{instr}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        # options: controls tokens/temperature etc.
        "options": {
            # Ollama uses "num_predict" as max tokens in docs/examples
            "num_predict": int(max_tokens),
            # you can customize temperature here if desired
            # "temperature": 0.0
        },
        # no template/system separate field used here (we included system text in prompt)
        "stream": False,
        "raw": False,
    }

    try:
        resp = requests.post(OLLAMA_GENERATE, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        # Ollama returns the final generated text in "response"
        if isinstance(data, dict) and "response" in data:
            text = data.get("response", "")
            # sometimes Ollama returns thinking field; ignore it
            return text.strip()
        # fallback: maybe endpoint returned other shape
        # try to string-concat useful parts
        if isinstance(data, dict):
            # try common keys
            for k in ("response", "text", "content"):
                if k in data:
                    return str(data[k]).strip()
            # last resort: return the full json as string (for debugging)
            return str(data)
        return str(data)
    except requests.exceptions.RequestException as e:
        # network / HTTP errors
        return (
            f"[Generation error] Could not contact Ollama at {OLLAMA_GENERATE}: {e}. "
            "Make sure Ollama is running and the model is available."
        )
    except Exception as e:
        return f"[Generation error] Unexpected error: {e}"
