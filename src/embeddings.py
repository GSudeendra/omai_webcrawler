import os
import time
import numpy as np
import faiss
import openai
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ollama import Client as OllamaClient
from fastapi.testclient import TestClient

# Load env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# FastAPI app
app = FastAPI()

# === Embedding & Search ===
class Embedder:
    def __init__(self):
        self.embedding_dim = None
        self.index = None
        self.url_mapping = []
        self.content_mapping = []
        self.ollama_client = OllamaClient()

    def embed_text(self, text: str) -> np.ndarray:
        try:
            print("[INFO] Trying OpenAI embedding...")
            response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
            embedding = np.array(response['data'][0]['embedding'], dtype=np.float32)
            self._initialize_index_if_needed(embedding)
            return embedding
        except Exception as e:
            print(f"[WARN] OpenAI embedding failed: {e}")

        try:
            print("[INFO] Falling back to local embedding with nomic-embed-text...")
            response = self.ollama_client.embeddings(model='nomic-embed-text', prompt=text)
            embedding = np.array(response['embedding'], dtype=np.float32)
            self._initialize_index_if_needed(embedding)
            return embedding
        except Exception as e:
            print(f"[ERROR] Local embedding failed: {e}")
            raise RuntimeError("Both OpenAI and local embedding failed")

    def _initialize_index_if_needed(self, embedding: np.ndarray):
        if self.index is None:
            self.embedding_dim = embedding.shape[0]
            print(f"[INFO] Initializing FAISS index with dimension {self.embedding_dim}")
            self.index = faiss.IndexFlatL2(self.embedding_dim)

    def store_embedding(self, url: str, embedding: np.ndarray, content: str):
        if url in self.url_mapping:
            print(f"[INFO] Skipping duplicate URL: {url}")
            return
        if embedding.shape[0] != self.embedding_dim:
            print(f"[ERROR] Embedding dimension mismatch: {embedding.shape[0]} != {self.embedding_dim}")
            return
        try:
            self.index.add(np.array([embedding]))
            self.url_mapping.append(url)
            self.content_mapping.append(content)
            print(f"[INFO] Stored embedding for {url}")
        except Exception as e:
            print(f"[ERROR] Error storing embedding: {e}")

    def search_similar(self, query: str, top_k: int = 5):
        query_vec = self.embed_text(query)
        D, I = self.index.search(np.array([query_vec]), top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            if idx < len(self.url_mapping):
                results.append({
                    "url": self.url_mapping[idx],
                    "content": self.content_mapping[idx],
                    "similarity_score": float(dist)
                })
        return results

embedder = Embedder()

# === Request Models ===
class CrawlRequest(BaseModel):
    url: str
    depth: int = 0

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

# === API Routes ===
@app.post("/crawl")
def crawl(request: CrawlRequest):
    try:
        res = requests.get(request.url)
        soup = BeautifulSoup(res.text, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)

        if "Service Temporarily Unavailable" in text or len(text) < 50:
            raise HTTPException(status_code=400, detail="Crawled content is invalid or too short.")

        embedding = embedder.embed_text(text)
        embedder.store_embedding(request.url, embedding, text)
        return {"message": f"Crawled 1 pages and stored embeddings."}
    except Exception as e:
        print(f"[ERROR] /crawl failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
def search(request: SearchRequest):
    try:
        results = embedder.search_similar(request.query, request.top_k)
        for result in results:
            print(f"URL: {result['url']} | Score: {result['similarity_score']:.2f}")
        return {"results": results}
    except Exception as e:
        print(f"[ERROR] /search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
