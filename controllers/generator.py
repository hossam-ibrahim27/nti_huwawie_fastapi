from langchain_core.documents import Document

from controllers.llm import generate_text


def answer_question(
    question: str,
    search_results: list[tuple[Document, float]],
) -> dict:
    """Generate a grounded answer from already-retrieved chunks."""

    documents = [document for document, _ in search_results]

    if not documents:
        return {
            "question": question,
            "answer": "I could not find this in the provided guideline.",
            "retrieved_chunks": [],
            "sources": [],
        }

    context = "\n\n".join(
        (
            f"[Source {index}]\n"
            f"Section: {document.metadata.get('section', 'Unknown')}\n"
            f"Page: {document.metadata.get('page', 0) + 1}\n"
            f"{document.page_content}"
        )
        for index, document in enumerate(documents, start=1)
    )

    prompt = f"""
Answer the user's question using only the supplied clinical-guideline context.

Rules:
- Answer only from the supplied context.
- Do not treat a statement about effectiveness as a recommendation unless
  the context explicitly says "recommend", "offer", "should", or equivalent.
- If the context discusses a treatment but does not explicitly recommend it,
  clearly say that the retrieved evidence describes the treatment but does
  not state a formal recommendation.
- Match the strength of the guideline's wording: say "the guideline recommends",
  "consider" or "may" exactly as the context does. Never sound more certain
  than the context.
- If the context only partly answers the question, say which part is not covered.
- Never give personalised medical advice.
- Do not invent clinical information.
- Add citations such as [Source 1] after every clinical claim.
- If the answer is not in the context, say:
  "I could not find this in the provided guideline."

Question:
{question}

Clinical-guideline context:
{context}
"""

    answer_text = generate_text(prompt, temperature=0.1)

    if not answer_text:
        answer_text = "The model did not return a text answer."

    return {
        "question": question,
        "answer": answer_text,
        "retrieved_chunks": [
            {
                "rank": index,
                "relevance_score": round(score, 3),
                "section": document.metadata.get("section", "Unknown"),
                "page": document.metadata.get("page", 0) + 1,
                "content_preview": document.page_content[:350],
            }
            for index, (document, score) in enumerate(search_results, start=1)
        ],
        "sources": [
            {
                "source_number": index,
                "section": document.metadata.get("section", "Unknown"),
                "page": document.metadata.get("page", 0) + 1,
                "source": document.metadata.get("source", "Unknown"),
            }
            for index, document in enumerate(documents, start=1)
        ],
    }