from langchain_core.documents import Document

from controllers.vector_store import load_vector_store


RELEVANCE_THRESHOLD = 0.25  # adjust after calibration

FALLBACK_MESSAGE = (
    "Sorry, this question is outside the scope of the sources available to me "
    "(the NICE guideline on cataracts in adults). "
    "I can't answer it reliably, so please consult a qualified medical professional."
)


def search_guidelines_with_scores(
    question: str,
    k: int = 5,
) -> list[tuple[Document, float]]:
    """
    Return the top-k relevant chunks with their relevance scores.
    Higher score means the chunk is more relevant.
    """
    vector_store = load_vector_store()

    return vector_store.similarity_search_with_relevance_scores(
        question,
        k=k,
    )


def search_guidelines(
    question: str,
    k: int = 5,
) -> list[Document]:
    """Return only the documents, for code that does not need scores."""
    results = search_guidelines_with_scores(question, k=k)

    return [document for document, _ in results]


def top_relevance_score(results: list[tuple[Document, float]]) -> float:
    """Return the highest relevance score, or 0.0 if there are no results."""
    return max((score for _, score in results), default=0.0)