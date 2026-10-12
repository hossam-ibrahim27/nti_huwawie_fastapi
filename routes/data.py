from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from controllers.chunks import create_section_aware_chunks
from controllers.loader import load_pdf_document
from controllers.vector_store import create_vector_store, load_vector_store

dataRoute = APIRouter(prefix="/api/v1", tags=["api_v1"])

PDF_PATH = (
    Path(__file__).resolve().parent.parent
    / "assets"
    / "cataracts-in-adults.pdf"
)


@dataRoute.post("/build-database")
def build_database():
    """Load the PDF, chunk it, embed it, then store it in Chroma (rebuilds from scratch)."""

    if not PDF_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"PDF file not found: {PDF_PATH.name}",
        )

    loaded_documents = load_pdf_document(str(PDF_PATH))
    chunks = create_section_aware_chunks(loaded_documents)

    create_vector_store(chunks)

    return {
        "status": "success",
        "message": (
            f"Loaded {len(loaded_documents)} pages and "
            f"stored {len(chunks)} chunks in Chroma."
        ),
        "chunks_preview": [
            {
                "section": chunk.metadata.get("section", "Unknown"),
                "page": chunk.metadata.get("page", 0) + 1,
                "content_preview": chunk.page_content[:200],
            }
            for chunk in chunks[:10]
        ],
    }


@dataRoute.get("/search")
def search_guideline(
    question: str = Query(..., min_length=3),
    k: int = Query(default=4, ge=1, le=10),
):
    """Search the stored chunks directly, with scores (debugging endpoint)."""

    vector_store = load_vector_store()
    results = vector_store.similarity_search_with_relevance_scores(question, k=k)

    return {
        "question": question,
        "results": [
            {
                "relevance_score": round(score, 3),
                "section": document.metadata.get("section", "Unknown"),
                "page": document.metadata.get("page", 0) + 1,
                "source": Path(document.metadata.get("source", "Unknown")).name,
                "content": document.page_content,
            }
            for document, score in results
        ],
    }