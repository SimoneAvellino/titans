def trunc(text: str, max_len: int = 50) -> str:
    return text if len(text) <= max_len else f"{text[:max_len-3]}..."
