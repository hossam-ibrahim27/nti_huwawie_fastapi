from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document

from controllers.embeddings import get_embeddings


CHROMA_DIRECTORY = "chroma_db"
COLLECTION_NAME = "clinical_guidelines"


def create_vector_store(chunks: list[Document]) -> Chroma:
    """Rebuild the collection from scratch, then embed and store the chunks."""

    # Without this, calling /build-database twice stores every chunk twice,
    # which fills the top-k results with duplicates and distorts Precision@k.
    existing = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIRECTORY,
    )
    existing.delete_collection()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIRECTORY,
    )

    # Make the next search open the freshly built collection.
    load_vector_store.cache_clear()

    return vector_store


@lru_cache(maxsize=1)
def load_vector_store() -> Chroma:
    """Open the existing local Chroma database (cached after the first call)."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIRECTORY,
    )