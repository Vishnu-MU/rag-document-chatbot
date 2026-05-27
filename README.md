# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and ask questions based on their content. The system uses semantic search with FAISS and Sentence Transformers to retrieve relevant document chunks and generates context-aware answers using Google's Gemini LLM.

---

## Features

- PDF document upload and ingestion
- Text chunking using LangChain
- Semantic search with FAISS
- Sentence Transformer embeddings
- Context-aware answer generation using Gemini
- FastAPI backend with Swagger UI
- REST API endpoints for upload and chat
- Modular and production-style architecture

---

## Tech Stack

- Python
- FastAPI
- FAISS
- Sentence Transformers
- Google Gemini API
- LangChain Text Splitters
- PyPDF

---

## Project Architecture

```text
PDF → Text Extraction → Chunking → Embeddings → FAISS Vector Store
     → Semantic Retrieval → Gemini LLM → Final Answer
```

---

## Installation

### Clone the repository

```bash
git clone https://github.com/your-username/rag_chatbot.git
cd rag_chatbot
```

### Create virtual environment

```bash
python -m venv venv
```

### Activate virtual environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux/Mac

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:

```env
GOOGLE_API_KEY=your_api_key
```

---

## Run the Application

```bash
uvicorn app.main:app --reload
```

API will run at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

## API Endpoints

### Upload PDF

```http
POST /upload
```

Uploads and ingests PDF documents into the vector database.

### Chat Endpoint

```http
POST /chat
```

Example request:

```json
{
  "query": "Why is my API request failing?",
  "top_k": 3
}
```

---

## Project Structure

```text
rag_chatbot/
│
├── app/
│   ├── main.py
│   ├── ingest.py
│   ├── retriever.py
│   ├── llm.py
│   ├── config.py
│
├── data/
├── vectorstore/
├── requirements.txt
├── .env
├── README.md
```

---

## Future Improvements

- Streamlit frontend
- Multi-document support
- Hybrid search
- Conversation memory
- Reranking models
- Docker deployment

---

## License

This project is licensed under the MIT License.