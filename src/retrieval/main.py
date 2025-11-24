import os

from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Initialize the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Qdrant client
qdrant_client = QdrantClient(
    host=os.getenv("QDRANT_HOST", "localhost"),
    port=int(os.getenv("QDRANT_PORT", "6333")),
    api_key=os.getenv("QDRANT_API_KEY"),
)


class Query(BaseModel):
    text: str


class Response(BaseModel):
    results: list


@app.post("/retrieve", response_model=Response)
async def retrieve(query: Query):
    """Implement hybrid search logic combining keyword and semantic search"""
    try:
        # Generate embedding for the query
        query_embedding = model.encode([query.text])[0].tolist()

        # Perform semantic search using vector similarity
        vector_results = qdrant_client.search(
            collection_name="code_chunks",
            query_vector=query_embedding,
            limit=10,  # Adjust as needed
            with_payload=True,
        )

        # Perform keyword search (BM25-like approach)
        keyword_results = qdrant_client.search(
            collection_name="code_chunks",
            query_text=query.text,  # This would require text-based indexing
            limit=10,  # Adjust as needed
            with_payload=True,
        )

        # Combine and rank results
        # For now, we'll use the vector results as the primary results
        # In a complete implementation, we would merge and re-rank both sets of results
        combined_results = []

        # Add vector search results
        for result in vector_results:
            combined_results.append({
                "id": result.id,
                "score": result.score,
                "payload": result.payload,
                "search_type": "vector",
            })

        # Add keyword search results that aren't already in the list
        for result in keyword_results:
            # Check if this result is already in combined_results to avoid duplicates
            if not any(r["id"] == result.id for r in combined_results):
                combined_results.append({
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload,
                    "search_type": "keyword",
                })

        # Sort by score in descending order
        combined_results.sort(key=lambda x: x["score"], reverse=True)

        # Limit to top 10 results
        final_results = combined_results[:10]

        return {"results": final_results}
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return {"results": []}
