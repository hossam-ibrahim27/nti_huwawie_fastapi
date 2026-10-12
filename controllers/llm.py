import os
from groq import Groq
import cohere

# 1) Cohere API
# MODEL_NAME = "command-a-plus-05-2026"

# def generate_text(prompt: str, temperature: float = 0.1) -> str:
    # """Call Cohere and return only the text part of the response."""
    # 1) Cohere API
    # api_key = os.getenv("COHERE_API_KEY")
    # if not api_key:
    #         raise ValueError("COHERE_API_KEY is missing.")
    # client = cohere.ClientV2(api_key=api_key)
    # client = Groq(api_key=api_key)

    # response = client.chat(
    #     model=MODEL_NAME,
    #     messages=[{"role": "user", "content": prompt}],
    #     temperature=temperature,
    # )

    # # Cohere can return a thinking item before the actual text item.
    # return "".join(
    #     item.text
    #     for item in response.message.content
    #     if getattr(item, "type", None) == "text"
    # ).strip()


# 2) Groq API (Qween)
MODEL_NAME = "qwen/qwen3.8-27b"
def generate_text(prompt: str, temperature: float = 0.1) -> str:
    """Call Groq and return only the text part of the response."""
     # 2) Groq API (Qween)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing.")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing.")

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )

    return (response.choices[0].message.content or "").strip() 


