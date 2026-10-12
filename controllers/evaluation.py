import json
from pathlib import Path

from controllers.pipeline import run_pipeline
from controllers.retriever import search_guidelines_with_scores


def _mean(values: list) -> float | None:
    values = [value for value in values if value is not None]
    return round(sum(values) / len(values), 2) if values else None


def evaluate_precision_at_k(
    test_file: str,
    k: int = 5,
) -> dict:
    """Retrieval-only evaluation (no LLM calls): Precision@k, Hit@1, Hit@k, MRR."""

    # Refusal cases have no expected pages, so they are excluded from retrieval metrics.
    test_cases = [
        test_case
        for test_case in json.loads(Path(test_file).read_text(encoding="utf-8"))
        if not test_case.get("expect_refusal")
    ]

    all_scores = []
    hit_at_1 = []
    hit_at_k = []
    reciprocal_ranks = []
    results = []

    for test_case in test_cases:
        question = test_case["question"]
        expected_pages = test_case["expected_pages"]

        retrieved = search_guidelines_with_scores(
            question=question,
            k=k,
        )

        # Chroma page metadata starts from 0, while PDF page numbers start from 1.
        retrieved_pages = [
            document.metadata.get("page", -1) + 1
            for document, _ in retrieved
        ]

        relevant_count = sum(
            1 for page in retrieved_pages if page in expected_pages
        )

        first_hit_rank = next(
            (
                rank
                for rank, page in enumerate(retrieved_pages, start=1)
                if page in expected_pages
            ),
            None,
        )

        precision_at_k = relevant_count / k

        all_scores.append(precision_at_k)
        hit_at_1.append(1.0 if first_hit_rank == 1 else 0.0)
        hit_at_k.append(1.0 if first_hit_rank else 0.0)
        reciprocal_ranks.append(1 / first_hit_rank if first_hit_rank else 0.0)

        results.append(
            {
                "question": question,
                "expected_pages": expected_pages,
                "retrieved_pages": retrieved_pages,
                "relevant_chunks": relevant_count,
                "precision_at_k": round(precision_at_k, 2),
                "first_hit_rank": first_hit_rank,
            }
        )

    return {
        "k": k,
        "average_precision_at_k": _mean(all_scores),
        "hit_at_1": _mean(hit_at_1),
        "hit_at_k": _mean(hit_at_k),
        "mrr": _mean(reciprocal_ranks),
        "results": results,
    }


def evaluate_full_suite(
    test_file: str,
    k: int = 5,
) -> dict:
    """Run the full safety pipeline on every test case and score it."""

    test_cases = json.loads(Path(test_file).read_text(encoding="utf-8"))

    rows = []
    precisions, citation_scores, faithfulness_scores = [], [], []
    refusal_cases = refusals_correct = 0
    answerable_cases = false_refusals = 0

    for test_case in test_cases:
        question = test_case["question"]
        result = run_pipeline(question, k=k)

        # Questions that SHOULD be refused (out of scope / personal advice).
        if test_case.get("expect_refusal"):
            refusal_cases += 1
            refusals_correct += int(result["refused"])
            rows.append(
                {
                    "question": question,
                    "type": "refusal_test",
                    "refused_correctly": result["refused"],
                    "refusal_reason": result["refusal_reason"],
                    "top_score": result["top_score"],
                }
            )
            continue

        # Questions that SHOULD be answered.
        answerable_cases += 1

        if result["refused"]:
            false_refusals += 1
            precisions.append(0.0)
            rows.append(
                {
                    "question": question,
                    "type": "retrieval_test",
                    "false_refusal": True,
                    "refusal_reason": result["refusal_reason"],
                    "top_score": result["top_score"],
                }
            )
            continue

        retrieved_pages = [chunk["page"] for chunk in result["retrieved_chunks"]]
        relevant = sum(
            1 for page in retrieved_pages if page in test_case["expected_pages"]
        )
        precision = relevant / k

        validation = result["claim_validation"] or {}
        citation_accuracy = validation.get("citation_accuracy")
        faithfulness = validation.get("faithfulness")

        precisions.append(precision)
        citation_scores.append(citation_accuracy)
        faithfulness_scores.append(faithfulness)

        rows.append(
            {
                "question": question,
                "type": "retrieval_test",
                "precision_at_k": round(precision, 2),
                "citation_accuracy": citation_accuracy,
                "faithfulness": faithfulness,
                "top_score": result["top_score"],
                "unsupported_claims": validation.get("unsupported_claims", []),
            }
        )

    return {
        "k": k,
        "summary": {
            "avg_precision_at_k": _mean(precisions),
            "avg_citation_accuracy": _mean(citation_scores),
            "avg_faithfulness": _mean(faithfulness_scores),  # slide target: 0.9+
            "refusal_accuracy": (
                round(refusals_correct / refusal_cases, 2) if refusal_cases else None
            ),
            "false_refusal_rate": (
                round(false_refusals / answerable_cases, 2)
                if answerable_cases
                else None
            ),
        },
        "results": rows,
    }