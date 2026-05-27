import json
import os

import faiss
import numpy as np\
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
    Reads a PDF file and extracts its text content.
    Returns a list of dictionaries, each containing a chunk of text and its corresponding metadata.
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
    Splits the text of a page into smaller chunks using RecursiveCharacterTextSplitter.
    Returns a list of dictionaries, each containing a chunk of text and its corresponding metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    return splitter.split_text(page_text)

