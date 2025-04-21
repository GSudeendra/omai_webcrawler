from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup

from src.embeddings import Embedder
from llama_streaming import router as llama_router


app = FastAPI()
embedder = Embedder()
app.include_router(llama_router)

class CrawlRequest(BaseModel):
    url: str
    depth: int = 0  # Not used yet

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/crawl")
def crawl(request: CrawlRequest):
    try:
        res = requests.get(request.url)
        soup = BeautifulSoup(res.text, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
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
        return {"results": results}
    except Exception as e:
        print(f"[ERROR] /search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
