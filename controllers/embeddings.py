import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings

ENV_PATH = (
    Path(__file__).resolve().parent.parent
    / "assets"
    / ".env"
)

load_dotenv(ENV_PATH)


def get_embeddings():
    cohere_api_key = os.getenv("COHERE_API_KEY")

    if not cohere_api_key:
        raise ValueError(
            "COHERE_API_KEY is missing. Add it to your .env file."
        )

    return CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=cohere_api_key,
    )
