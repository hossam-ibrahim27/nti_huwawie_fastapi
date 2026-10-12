from pathlib import Path

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from controllers.evaluation import evaluate_full_suite, evaluate_precision_at_k
from controllers.pipeline import run_pipeline

askRoute = APIRouter(prefix="/api/v1", tags=["api_v1"])

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_FILE = PROJECT_ROOT / "tests" / "retrieval_test_set.json"


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


@askRoute.post("/ask")
def ask_guideline(
    request: QuestionRequest,
    k: int = Query(default=5, ge=1, le=10),
):
    """Full safety pipeline: input risk, retrieval threshold, claim validation."""
    return run_pipeline(request.question, k=k)


@askRoute.get("/evaluate")
def evaluate(k: int = Query(default=5, ge=1, le=10)):
    """Fast, retrieval-only evaluation (no LLM calls)."""
    return evaluate_precision_at_k(test_file=str(TEST_FILE), k=k)


@askRoute.get("/evaluate/full")
def evaluate_full(k: int = Query(default=5, ge=1, le=10)):
    """Full suite: Precision@k, citation accuracy, faithfulness, refusals."""
    return evaluate_full_suite(test_file=str(TEST_FILE), k=k)