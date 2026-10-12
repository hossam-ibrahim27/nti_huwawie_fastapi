from langchain_community.document_loaders import PyPDFLoader

def load_pdf_document(file_path: str):
    """
    Load a PDF document using PyPDFLoader.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        List[Document]: A list of Document objects representing the loaded PDF.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents from {file_path}")
    for i, doc in enumerate(documents):
        print(f"Document {i + 1}:")
        print(f"Metadata: {doc.metadata}")
        print(f"Content: {doc.page_content[:100]}...")  # Print first 100 characters of content 
    return documents
