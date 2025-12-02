"""
Context enrichment and building utilities
"""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .generator import LLMClient
from .prompt_constructor import ContextPromptBuilder
from .prompt_constructor import MultiLanguageContextPrompt
from .prompt_constructor import SymbolIndex


def update_summary(record_id: str, summary: str):
    """Update summary for a chunk in the database"""
    db_path = Path("enhanced_chunks.db")
    table_name = "enhanced_code_chunks"
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(f"UPDATE {table_name} SET summary = ? WHERE id = ?", (summary, record_id))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ Failed to update {record_id}: {e}")


class ContextEnricher:
    """Enrich code chunks with AI-generated context summaries"""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        symbol_index: dict[str, Any] | None = None,
        prompt_builder: ContextPromptBuilder | None = None,
        llm_client: LLMClient | None = None,
    ):
        self.chunks = chunks
        self.symbol_index = SymbolIndex(symbol_index) if symbol_index else None
        self.prompt_builder = prompt_builder or MultiLanguageContextPrompt()
        self.llm_client = llm_client or LLMClient()
        self.executor = ThreadPoolExecutor(max_workers=4)

    def _normalize_chunk(self, chunk_data: dict[str, Any]) -> dict[str, Any]:
        """Normalize chunk data to handle both old and new formats"""
        # Ensure context is a dict (handle string context from old format)
        if isinstance(chunk_data.get("context"), str):
            chunk_data["context"] = {"summary": chunk_data["context"]}

        # Ensure all required fields have defaults
        chunk_data.setdefault("metadata", {})
        chunk_data.setdefault("documentation", {})
        chunk_data.setdefault("analysis", {})
        chunk_data.setdefault("relationships", {})
        chunk_data.setdefault("references", [])
        chunk_data.setdefault("defines", [])

        # Handle qualified_name fallback
        if "qualified_name" not in chunk_data:
            chunk_data["qualified_name"] = chunk_data.get("name", "")

        return chunk_data

    async def enrich(self, batch_size: int = 4) -> list[dict[str, Any]]:
        """Enrich chunks with AI-generated summaries in batches"""
        enriched = []
        total_chunks = len(self.chunks)

        # Normalize chunks
        normalized_chunks = [self._normalize_chunk(chunk) for chunk in self.chunks]

        # Filter out already summarized chunks
        summarized_ids = set(get_summarized_chunks_ids())
        filtered_chunks = [chunk for chunk in normalized_chunks if chunk.get("id") not in summarized_ids]

        print(f"Already summarized chunks: {len(summarized_ids)}")
        print(f"Total chunks: {len(normalized_chunks)}")
        print(f"Chunks to process: {len(filtered_chunks)}")

        # Temporary storage for batching
        batch_prompts = []
        batch_chunks = []

        for i, chunk in enumerate(filtered_chunks, 1):
            print(
                f"[{i}/{len(filtered_chunks)}] Preparing context for {chunk.get('type', 'unknown')} '{chunk.get('name', 'unknown')}' (ID: {chunk.get('id')})..."
            )

            # Build prompt
            prompt = self.prompt_builder.build_prompt(chunk, self.symbol_index)

            # Add to batch buffers
            batch_prompts.append(prompt)
            batch_chunks.append(chunk)

            # If batch is full → send to model
            if len(batch_prompts) == batch_size:
                summaries = await self.llm_client.generate_batch(batch_prompts)

                # Assign summaries back
                for chunk_obj, summary in zip(batch_chunks, summaries, strict=False):
                    enriched.append(self._process_chunk(chunk_obj, summary))
                # Reset batch
                batch_prompts = []
                batch_chunks = []

        # Handle last incomplete batch
        if batch_prompts:
            summaries = await self.llm_client.generate_batch(batch_prompts)
            for chunk_obj, summary in zip(batch_chunks, summaries, strict=False):
                enriched.append(self._process_chunk(chunk_obj, summary))

        print("✅ Finished enrichment with batching.")
        return enriched

    def _process_chunk(self, chunk: dict[str, Any], summary: str) -> dict[str, Any]:
        """Process a chunk with its generated summary"""
        chunk_dict = chunk.copy()
        chunk_dict["context"]["summary"] = summary

        # Async DB update via thread pool
        self.executor.submit(update_summary, chunk.get("id"), summary)

        return chunk_dict


def get_summarized_chunks_ids() -> list[str]:
    """Get IDs of chunks that already have summaries"""
    db_path = Path("enhanced_chunks.db")
    table_name = "enhanced_code_chunks"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    query = f"SELECT id FROM {table_name} WHERE summary IS NOT NULL AND summary != ''"

    cur.execute(query)
    results = [row[0] for row in cur.fetchall()]

    conn.close()
    return results


def stats_check(setl: int, chunkl: int, nchunkl: int):
    """Print enrichment statistics"""
    print(f"Already summarized chunks: {setl}")
    print(f"Total chunks count: {chunkl}")
    print(f"To Be summarized chunks: {nchunkl}")
