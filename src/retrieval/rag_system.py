from typing import Any

import requests
from rich.console import Console
from rich.markdown import Markdown

from src.config import EmbeddingConfig
from src.config import LLMConfig
from src.config import QdrantConfig
from src.embedding.embedder import EmbeddingGenerator
from src.retrieval.search import QdrantIndexer

console = Console()


class CodeRAG:
    def __init__(self):
        self.embedder = EmbeddingGenerator()
        self.indexer = QdrantIndexer()
        self.llm = LLMConfig()
        # Ensure collections exist with correct dimensions
        self.indexer.create_collections()

    def _get_all_collections(self) -> list[str]:
        """Get list of all code collections"""
        return self.indexer.config.get_collection_names()

    def query_codebase(self, user_query: str) -> str:
        """Main RAG pipeline: search + prompt + LLM"""
        # 1. Generate query embedding
        query_emb = self.embedder.generate_embedding_ollama(user_query)
        if not query_emb:
            return "Failed to generate query embedding."

        # 2. Search across all collections
        all_results = []
        for coll in self._get_all_collections():
            try:
                results = self.indexer.client.search(
                    collection_name=coll,
                    query_vector=("semantic_dense", query_emb),
                    limit=3,  # top 3 per collection
                )
                all_results.extend(results)
            except Exception as e:
                console.print(f"[yellow]Skipping {coll}: {e}[/yellow]")

        # 3. Sort and keep top 5 by score
        all_results.sort(key=lambda x: x.score, reverse=True)
        top_chunks = all_results[:5]
        print(top_chunks)

        if not top_chunks:
            return "No relevant code found."

        # 4. Build prompt
        prompt = self._build_rag_prompt(user_query, top_chunks)
        console.print(f"[red]{prompt}[/red]")

        # 5. Query LLM
        # return self._ask_llm(prompt)  # noqa: ERA001
        return prompt

    def _build_rag_prompt(self, query: str, chunks: list[Any]) -> str:
        """Build prompt with retrieved code"""
        context_blocks = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.payload.get("code", "").strip()
            if not text:
                continue
            file_path = chunk.payload.get("file_path", "unknown")
            chunk_type = chunk.payload.get("type", "code")
            block = f"### Snippet {i}\\nFile: {file_path}\\nType: {chunk_type}\\n```python\\n{text}\\n```"
            context_blocks.append(block)

        context = "\\n\\n".join(context_blocks)
        if not context:
            return f"Question: {query}\\nAnswer: I don't know."

        return f"""You are a senior Python developer who knows the codebase inside out.
Answer the question using ONLY the provided code snippets. If the answer isn't there, say "I don't know".

### Relevant Code:
{context}

### Question:
{query}

### Answer:"""

    def _ask_llm(self, prompt: str) -> str:
        """Send prompt to local llama.cpp server"""
        try:
            payload = {
                "prompt": prompt,
                "model": self.llm_model,
                "max_tokens": 8192,
                "temperature": 0.3,
                "stream": False,
            }
            response = requests.post(
                f"{self.llm_url}/engines/llama.cpp/v1/completions",
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            # Handle both /completions and /chat formats
            if "choices" in data:
                if "text" in data["choices"][0]:
                    return data["choices"][0]["text"].strip()
                if "message" in data["choices"][0]:
                    return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"LLM Error: {e!s}"
        else:
            return "Error: Unexpected LLM response format."


# ===== MAIN ===== (Preserved for standalone usage - all original functionality)
if __name__ == "__main__":
    # Config - matching original exactly
    QDRANT_HOST = "localhost"
    QDRANT_PORT = 6333
    LLM_API_URL = "http://localhost:12434/"
    LLM_MODEL = "ai/llama3.2:latest"  # e.g., "gemma-2b-it", "codellama-7b", etc.
    COLLECTION_PREFIX = "tipsy"  # must match your indexing

    # Initialize with enhanced components
    embedding_config = EmbeddingConfig(
        model_url=f"{LLM_API_URL}/engines/llama.cpp/v1",
        model_name="ai/embeddinggemma",
        embedding_dim=768,
        batch_size=32,
    )
    qdrant_config = QdrantConfig(host=QDRANT_HOST, port=QDRANT_PORT, collection_prefix=COLLECTION_PREFIX)

    rag = CodeRAG(
        embedding_config=embedding_config,
        qdrant_config=qdrant_config,
        llm_api_url=LLM_API_URL,
        llm_model_name=LLM_MODEL,
    )

    # Interactive loop - preserved exactly
    console.print("[bold green]CodeRAG Ready! Ask questions about your codebase.[/bold green]")
    while True:
        try:
            query = input("\\n❓ Your question: ").strip()
            if not query or query.lower() in {"quit", "exit"}:
                break

            console.print("[cyan]🔍 Searching codebase...[/cyan]")
            answer = rag.query_codebase(query)

            console.print("\\n[bold green]🤖 Answer:[/bold green]")
            console.print(Markdown(answer))

        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
