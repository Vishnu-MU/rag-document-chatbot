import json

import faiss    
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import (
    FAISS_INDEX_PATH,
    METADATA_PATH,
    EMBEDDING_MODEL_NAME,
    TOP_K
)


class Retriever:
    def __init__(self):
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        print("Loading FAISS index and metadata...")
        self.index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
            
    def retrieve(self, query, top_k=TOP_K):
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True).astype('float32')
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        
        for distance, index_id in zip(distances[0], indices[0]):
            if index_id == -1:  # No more results
                continue
            
            item = self.metadata[index_id]
            
            results.append({
                'chunk_id': item['chunk_id'],
                'text': item['text'],
                'source': item['source'],
                'page_number': item['page_number'],
                'distance': float(distance)
            })
        
        return results

if __name__ == "__main__":
    retriever = Retriever()
    query = "Why is my API request failing?"
    results = retriever.retrieve(query, top_k=3)
    
    for result in results:
        print("=" * 80)
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Text: {result['text']}")
        print(f"Source: {result['source']}")
        print(f"Page Number: {result['page_number']}")
        print(f"Distance: {result['distance']}\n")