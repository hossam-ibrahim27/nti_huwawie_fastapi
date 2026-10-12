import json
import re

from langchain_core.documents import Document

from controllers.llm import generate_text

NOT_FOUND_PHRASE = "could not find this in the provided guideline"

# Below this faithfulness score, the answer gets a visible warning.
FAITHFULNESS_WARNING_THRESHOLD = 0.8

# Confidence labels for the UI. Calibrate with your own score distribution.
HIGH_CONFIDENCE_SCORE = 0.50
MEDIUM_CONFIDENCE_SCORE = 0.35

DISCLAIMER = (
    "This summarises the NICE guideline and is not a substitute for clinical judgement."
)

RISK_REFUSAL_MESSAGE = (
    "I can't give personalised medical advice. This tool only summarises the NICE "
    "guideline on cataracts in adults. Please discuss your specific case with an "
    "eye specialist."
)

# Deliberately narrow: broad patterns like "should I" would wrongly block
# legitimate guideline questions ("What information should I be given?").
PERSONAL_ADVICE_PATTERNS = [
    r"\bmy (father|mother|dad|mom|wife|husband|son|daughter|brother|sister)\b",
    r"\bfor my (father|mother|dad|mom|wife|husband|son|daughter)\b",
    r"\b(is it|are they) safe for me\b",
    r"\bspecifically for me\b",
    r"\bmy (eye|eyes|vision|surgery|operation)\b",
]


def input_risk_check(question: str) -> bool:
    """Checkpoint 1: True if the question asks for personal medical advice."""
    text = question.lower()
    return any(re.search(pattern, text) for pattern in PERSONAL_ADVICE_PATTERNS)


def confidence_label(top_score: float) -> str:
    if top_score >= HIGH_CONFIDENCE_SCORE:
        return "high"
    if top_score >= MEDIUM_CONFIDENCE_SCORE:
        return "medium"
    return "low"


def _parse_claims(raw: str) -> list[dict] | None:
    """Extract the claims list from the validator's JSON output."""
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    start, end = raw.find("{"), raw.rfind("}")

    if start == -1 or end == -1:
        return None

    try:
        return json.loads(raw[start : end + 1]).get("claims", [])
    except (json.JSONDecodeError, AttributeError):
        return None


def validate_answer(answer: str, documents: list[Document]) -> dict:
    """
    Checkpoint 3: split the answer into claims, check each against the
    retrieved text, and score faithfulness and citation accuracy.
    """
    empty = {
        "skipped": True,
        "faithfulness": None,
        "citation_accuracy": None,
        "claims": [],
        "unsupported_claims": [],
    }

    # A correct "not found" answer has no claims to validate.
    if NOT_FOUND_PHRASE in answer.lower():
        return empty

    context = "\n\n".join(
        f"[Source {index}]\n{document.page_content}"
        for index, document in enumerate(documents, start=1)
    )

    prompt = f"""You are a strict fact-checker for a clinical-guideline assistant.

Split the ANSWER into individual factual claims. Keep any [Source n] tags
inside the claim they belong to. For each claim, decide whether the CONTEXT
fully supports it, and list the source numbers that actually support it.

Return ONLY valid JSON in exactly this shape, with no extra text:
{{"claims": [{{"claim": "...", "supported": true, "supporting_sources": [1]}}]}}

ANSWER:
{answer}

CONTEXT:
{context}
"""

    claims = _parse_claims(generate_text(prompt, temperature=0.0))

    if claims is None:
        return {**empty, "error": "validator_returned_invalid_json"}

    supported_count = sum(1 for claim in claims if claim.get("supported"))
    faithfulness = round(supported_count / len(claims), 2) if claims else None

    # Citation accuracy = correct citations / total citations given.
    total_citations = 0
    correct_citations = 0

    for claim in claims:
        cited = {
            int(number)
            for number in re.findall(r"\[Source (\d+)\]", claim.get("claim", ""))
        }
        supporting = {
            int(number)
            for number in claim.get("supporting_sources", [])
            if str(number).isdigit()
        }

        for number in cited:
            total_citations += 1
            if number in supporting:
                correct_citations += 1

    citation_accuracy = (
        round(correct_citations / total_citations, 2) if total_citations else None
    )

    return {
        "skipped": False,
        "faithfulness": faithfulness,
        "citation_accuracy": citation_accuracy,
        "claims": claims,
        "unsupported_claims": [
            claim["claim"] for claim in claims if not claim.get("supported")
        ],
    }