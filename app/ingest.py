import json
import os

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    DATA_DIR,
    VECTORSTORE_DIR,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

def load_pdf(file_path):
    """
    Loads a PDF document and extracts text from each page.
    Returns a list of dictionaries, each containing the text and page number of a page.
    """
    reader = PdfReader(file_path)
    pages = []
    
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        
        if text and text.strip():
            pages.append({
                "text": text,
                "page_number": page_number
            })
    return pages

def split_page_text(page_text):
    """
    Splits the text of a page into smaller chunks using a recursive character text splitter.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    return splitter.split_text(page_text)

def ingest_document(file_path):
    """
    Ingests a PDF document by loading it, splitting it into chunks, generating embeddings for each chunk,
    and storing the embeddings in a FAISS index along with metadata about each chunk.
    """
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    
    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    print(f"Reading PDF document... {file_path}")
    pages = load_pdf(file_path)
    
    all_chunks = []
    metadata = []
    
    chunk_id = 0
    
    print("Splitting text into chunks...")
    
    for page in pages:
        page_number = page["page_number"]
        chunks = split_page_text(page["text"])
        
        for chunk in chunks:
            all_chunks.append(chunk)
            
            metadata.append({
                "chunk_id": chunk_id,
                "page_number": page_number,
                "source": file_path,
                "text": chunk
            })
            
            chunk_id += 1
    if not all_chunks:
        raise ValueError("No text chunks were created from the document. Please check the PDF content.")

    print("Total chunks created:", len(all_chunks))
    print("Generating embeddings...")
    embeddings = embedding_model.encode(
        all_chunks,
        show_progress_bar=True,
        convert_to_numpy=True
    )
        
    embeddings = embeddings.astype('float32')

    dimension = embeddings.shape[1]
    print(f"Embedding dimension: {dimension}")

    print("Creating FAISS index...")
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print(f"Total vectors stored in FAISS index: {index.ntotal}")

    print("Saving FAISS index to disk...")
    faiss.write_index(index, FAISS_INDEX_PATH)

    print("Saving metadata to disk...")
    with open(METADATA_PATH, 'w' ,encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        
    print("Ingestion complete.")

if __name__ == "__main__":
    pdf_path = os.path.join(DATA_DIR, "sample_rag_document.pdf")
    ingest_document(pdf_path)
