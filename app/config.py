import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"

FAISS_INDEX_PATH = "vectorstore/faiss_index.bin"
METADATA_PATH = "vectorstore/metadata.json"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

TOP_K = 3 

LLM_MODEL = "gemini-2.5-flash"

