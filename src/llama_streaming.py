# llama_streaming.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Generator
from src.embeddings import Embedder

router = APIRouter()

# Pydantic model for the request body
class AskRequest(BaseModel):
    prompt: str

embedder = Embedder()  # Initialize Embedder for Ollama usage

@router.post("/ask")
def ask_llama(request: AskRequest):
    """
    Stream response from local LLM (llama3.2) using Ollama.
    """
    try:
        def generate_response() -> Generator[str, None, None]:
            # Use the prompt from the validated request body
            prompt = request.prompt
            for chunk in embedder.ollama_client.chat(
                    model="llama3.2",
                    messages=[{"role": "user", "content": prompt}],
                    stream=True
            ):
                content = chunk.get("message", {}).get("content")
                if content:
                    yield content

        return StreamingResponse(generate_response(), media_type="text/plain")

    except Exception as e:
        # Improved error handling with detailed logging
        print(f"[ERROR] /ask failed: {e}")
        raise HTTPException(status_code=422, detail=f"Request failed: {e}")
