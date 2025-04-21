from fastapi.testclient import TestClient
from src.main import app  # Import the FastAPI app

client = TestClient(app)

def test_ask_llama_streaming():
    # Define the prompt you want to test
    prompt = "Tell me about Herman Melville."

    # Send the request to the /ask endpoint with streaming enabled
    response = client.post("/ask", json={"prompt": prompt})

    # Check for successful response
    assert response.status_code == 200

    # Initialize content variable to capture streamed response
    content = ""

    # Read the streamed content in chunks
    for chunk in response.iter_text():
        content += chunk
        if len(content) > 0:  # You can also check other conditions based on your needs
            break  # Break once we have some meaningful content

    # Perform assertions on the content
    assert len(content) > 0, "The response content is empty or not meaningful."  # Ensure there's content
    assert "Melville" in content, f"Expected substring 'Melville' not found in the response."

    # Print for debugging (optional)
    print("Streaming response:", content)
