from controllers.generator import answer_question
from controllers.retriever import (
    FALLBACK_MESSAGE,
    RELEVANCE_THRESHOLD,
    search_guidelines_with_scores,
    top_relevance_score,
)
from controllers.safety import (
    DISCLAIMER,
    FAITHFULNESS_WARNING_THRESHOLD,
    RISK_REFUSAL_MESSAGE,
    confidence_label,
    input_risk_check,
    validate_answer,
)


def _refusal(question: str, message: str, reason: str, top_score=None) -> dict:
    return {
        "question": question,
        "answer": message,
        "refused": True,
        "refusal_reason": reason,
        "top_score": top_score,
        "confidence": "none",
        "retrieved_chunks": [],
        "sources": [],
        "claim_validation": None,
        "warning": None,
    }


def run_pipeline(question: str, k: int = 5) -> dict:
    """Safety decision tree: input risk -> retrieval threshold -> claim validation."""

    # Checkpoint 1: input risk check (before spending any retrieval or LLM call).
    if input_risk_check(question):
        return _refusal(question, RISK_REFUSAL_MESSAGE, "personal_medical_advice")

    # Checkpoint 2: retrieval confidence threshold (no generation below it).
    results = search_guidelines_with_scores(question, k=k)
    top_score = round(top_relevance_score(results), 3)

    if top_score < RELEVANCE_THRESHOLD:
        return _refusal(question, FALLBACK_MESSAGE, "out_of_scope", top_score)

    result = answer_question(question, results)

    # Checkpoint 3: claim validation against the retrieved text.
    validation = validate_answer(
        result["answer"],
        [document for document, _ in results],
    )

    warning = None
    faithfulness = validation.get("faithfulness")

    if faithfulness is not None and faithfulness < FAITHFULNESS_WARNING_THRESHOLD:
        warning = (
            "Some statements in this answer could not be verified against the "
            "guideline text. Please check the cited sources."
        )

    return {
        **result,
        "refused": False,
        "refusal_reason": None,
        "top_score": top_score,
        "confidence": confidence_label(top_score),
        "claim_validation": validation,
        "warning": warning,
        "disclaimer": DISCLAIMER,
    }