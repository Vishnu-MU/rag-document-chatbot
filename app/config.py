import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DATA_DIR = "data"
VECTORS_DIR = "vectorstore"

FAISS_INDEX_PATH = "vectorstore/faiss_index.bin"
METADATA_PATH = "vectorstore/metadata.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

TOP_K = 3 

LLM_MODEL = "gpt-4.0-mini"

