import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

from app.ingest import ingest_document
from app.retriever import Retriever
from app.llm import generate_answer

app = FastAPI(
    title="RAG Documentation Chatbot",
    description="A document question answering chatbot using FAISS, Sentence Transformers, and Google Generative AI.",
    version="1.0.0",
)

class ChatRequest(BaseModel):
    query: str
    top_k: int = 3
    

class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: list
    
retriever = None

@app.on_event("startup")
def startup_event():
    global retriever
    retriever = Retriever()
    
@app.get("/")
def home():
    return {"message": "Welcome to the RAG Documentation Chatbot API. Use the /chat endpoint to ask questions.",
            "docs": "Go to /docs to test the API"
            }
    
@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/upload")
def upload_documents(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
        
        os.makedirs("data", exist_ok=True)
        
        file_path = os.path.join("data", file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        ingest_document(file_path)
        
        global retriever
        retriever = Retriever()
        
        return {
            "message": "document uploaded and ingested successfully",
            "filename": file.filename
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty.")

        retrieved_chunks = retriever.retrieve(request.query, top_k=request.top_k)

        if not retrieved_chunks:
            return ChatResponse(query=request.query, answer="No relevant information found in the documents.", sources=[])
        
        answer = generate_answer(request.query, retrieved_chunks)
        
        sources = [
            {
                "chunk_id": chunk['chunk_id'],
                "source": chunk['source'],
                "page_number": chunk['page_number'],
                "distance": chunk['distance']
            }
            for chunk in retrieved_chunks
        ]
        
        return {
            "query": request.query,
            "answer": answer,
            "sources": sources
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))