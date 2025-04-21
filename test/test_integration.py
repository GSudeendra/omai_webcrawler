import time
from fastapi.testclient import TestClient

from src.embeddings import app

client = TestClient(app)

def test_crawl_and_search():
    # Crawl a valid content-rich page
    crawl_response = client.post("/crawl", json={"url": "https://en.wikipedia.org/wiki/Herman_Melville", "depth": 0})
    print("Crawl Response:", crawl_response.text)
    assert crawl_response.status_code == 200
    assert "message" in crawl_response.json()

    time.sleep(1)

    # Search
    search_response = client.post("/search", json={"query": "Herman", "top_k": 3})
    print("Search Response:", search_response.json())
    assert search_response.status_code == 200
    assert "results" in search_response.json()

