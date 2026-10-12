import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def is_heading(line: str) -> bool:
    """Return True only for clean, likely document headings."""

    text = " ".join(line.split())

    if not text:
        return False

    if len(text) < 4 or len(text) > 80:
        return False

    if len(text.split()) > 10:
        return False

    # Reject normal sentences and malformed table-of-contents lines.
    if "." in text and not re.match(r"^\d+(?:\.\d+)*\.?\s+", text):
        return False

    # Examples:
    # "Introduction"
    # "Recommendations"
    # "1 Referral for cataract surgery"
    # "1.2 Pre-operative assessment"
    heading_pattern = (
        r"^(?:\d+(?:\.\d+)*\.?\s+)?"
        r"[A-Z][A-Za-z&,/()' -]+$"
    )

    return bool(re.match(heading_pattern, text))

def create_section_aware_chunks(
    documents: list[Document],
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> list[Document]:
    """
    Split PDF documents by detected headings, then split long sections
    into smaller chunks while preserving section and page metadata.
    """
    sections = []
    current_section = "Introduction"

    for document in documents:
        lines = document.page_content.splitlines()
        section_content = []

        for line in lines:
            if is_heading(line):
                # Save text that belongs to the previous section
                text = "\n".join(section_content).strip()

                if text:
                    sections.append(
                        Document(
                            page_content=text,
                            metadata={
                                **document.metadata,
                                "section": current_section,
                            },
                        )
                    )

                current_section = line.strip()
                section_content = []

            else:
                section_content.append(line)

        # Save remaining text from this PDF page
        text = "\n".join(section_content).strip()

        if text:
            sections.append(
                Document(
                    page_content=text,
                    metadata={
                        **document.metadata,
                        "section": current_section,
                    },
                )
            )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(sections)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    return chunks