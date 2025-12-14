def classify_question(question: str) -> str:
    q = question.lower()

    if any(k in q for k in ["summarize", "summary", "what is this paper about", "overview"]):
        return "summary"

    if any(k in q for k in ["explain", "how does", "why"]):
        return "explanation"

    return "fact"
