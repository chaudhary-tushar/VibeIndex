"""
Enhanced preprocessing for code chunks before embedding and indexing
"""

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from tqdm import tqdm

from src.config import settings

console = Console()


@dataclass
class ChunkPreprocessor:
    """Enhanced preprocessing for code chunks before embedding"""

    dedup_hashes: set = None
    stats: dict[str, int] = None

    def __post_init__(self):
        if self.dedup_hashes is None:
            self.dedup_hashes = set()
        if self.stats is None:
            self.stats = {
                "total": 0,
                "duplicates": 0,
                "enhanced": 0,
                "too_large": 0,
            }

    def deduplicate(self, chunks: list[dict]) -> list[dict]:
        """Remove duplicate chunks based on code content"""
        unique_chunks = []

        for chunk in chunks:
            # Create hash from code content
            code_hash = hashlib.md5(chunk["code"].encode()).hexdigest()

            if code_hash not in self.dedup_hashes:
                self.dedup_hashes.add(code_hash)
                unique_chunks.append(chunk)
            else:
                self.stats["duplicates"] += 1

        console.print(f"[yellow]Removed {self.stats['duplicates']} duplicate chunks[/yellow]")
        return unique_chunks

    def enhance_chunk(self, chunk: dict) -> dict:
        """Enhance chunk with additional context for better embeddings"""
        enhanced = chunk.copy()

        # Build rich text representation for embedding
        parts = []

        # 1. Add contextual prefix
        context_prefix = f"# {chunk['language'].upper()} {chunk['type'].upper()}"
        if chunk.get("qualified_name"):
            context_prefix += f": {chunk['qualified_name']}"
        parts.append(context_prefix)

        if isinstance(chunk.get("context"), str):
            parts.append(f"html/css context: {chunk.get('context')}")
        else:
            # 2. Add file context
            if chunk.get("context", {}).get("file_hierarchy"):
                file_ctx = " > ".join(chunk["context"]["file_hierarchy"])
                parts.append(f"# Location: {file_ctx}")

            # 3. Add domain context
            if chunk.get("context", {}).get("domain_context"):
                parts.append(f"# Purpose: {chunk['context']['domain_context']}")

        # 4. Add docstring if available
        if chunk.get("docstring"):
            parts.append(f'"""{chunk["docstring"]}"""')

        # 5. Add signature for functions
        if chunk.get("signature"):
            parts.append(chunk["signature"])

        # 6. Add decorators/metadata
        if chunk.get("metadata", {}).get("decorators"):
            parts.extend(chunk["metadata"]["decorators"])

        # 7. Add the actual code
        parts.append(chunk["code"])

        # 8. Add dependencies as comments
        if chunk.get("dependencies"):
            deps = ", ".join(chunk["dependencies"][:5])  # Limit to 5
            parts.append(f"# Dependencies: {deps}")

        # 9. Add defined symbols
        if chunk.get("defines"):
            parts.append(f"# Defines: {', '.join(chunk['defines'])}")

        # Combine all parts
        enhanced["embedding_text"] = "\n".join(parts)
        enhanced["embedding_text_length"] = len(enhanced["embedding_text"])

        self.stats["enhanced"] += 1
        return enhanced

    def validate_chunk(self, chunk: dict, max_tokens: int = 8192) -> bool:
        """Validate chunk is suitable for embedding"""
        # Rough token estimate (1 token ≈ 4 chars)
        estimated_tokens = len(chunk.get("embedding_text", chunk["code"])) // 4

        if estimated_tokens > max_tokens:
            self.stats["too_large"] += 1
            return False

        return True

    def fill_db(self, chunks: list[dict]) -> None:
        chunks_db = settings.get_project_db_path()
        conn = sqlite3.connect(Path(chunks_db))
        cur = conn.cursor()

        # 3️⃣ Create table (with flexible text columns for complex/nested fields)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS enhanced_code_chunks (
            id TEXT PRIMARY KEY,
            type TEXT,
            name TEXT,
            code TEXT,
            file_path TEXT,
            language TEXT,
            start_line INTEGER,
            end_line INTEGER,
            qualified_name TEXT,
            location TEXT,
            docstring TEXT,
            signature TEXT,
            complexity INTEGER,
            dependencies TEXT,
            parent TEXT,
            "references" TEXT,
            defines TEXT,
            metadata TEXT,
            documentation TEXT,
            analysis TEXT,
            relationships TEXT,
            context TEXT,
            summary TEXT,
            embedding TEXT
        )
        """)

        # 4️⃣ Insert data safely
        for chunk in chunks:
            # Serialize nested fields (lists/dicts) as JSON strings
            def safe_json(value):
                if value is None:
                    return None
                if isinstance(value, (dict, list)):
                    return json.dumps(value, ensure_ascii=False)
                return str(value)

            row = (
                chunk.get("id"),
                chunk.get("type"),
                chunk.get("name"),
                chunk.get("code"),
                chunk.get("file_path"),
                chunk.get("language"),
                chunk.get("start_line"),
                chunk.get("end_line"),
                chunk.get("qualified_name", ""),
                safe_json(chunk.get("location")),
                chunk.get("docstring"),
                chunk.get("signature"),
                chunk.get("complexity", 0),
                safe_json(chunk.get("dependencies")),
                chunk.get("parent"),
                safe_json(chunk.get("references")),
                safe_json(chunk.get("defines")),
                safe_json(chunk.get("metadata")),
                safe_json(chunk.get("documentation")),
                safe_json(chunk.get("analysis")),
                safe_json(chunk.get("relationships")),
                safe_json(chunk.get("context")),
                None,  # summary
                None,  # embedding
            )

            cur.execute(
                """
                INSERT OR IGNORE INTO enhanced_code_chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
                row,
            )

        # 5️⃣ Commit & close
        conn.commit()
        conn.close()

    def process(self, chunks: list[dict]) -> list[dict]:
        """Full preprocessing pipeline"""
        self.stats["total"] = len(chunks)
        console.print(f"[cyan]Starting preprocessing of {len(chunks)} chunks...[/cyan]")

        # 1. Deduplicate
        chunks = self.deduplicate(chunks)

        # 2. Enhance each chunk
        enhanced_chunks = []
        for chunk in tqdm(chunks, desc="Enhancing chunks", unit="chunk"):
            enhanced = self.enhance_chunk(chunk)
            if self.validate_chunk(enhanced):
                enhanced_chunks.append(enhanced)

        self.fill_db(enhanced_chunks)
        console.print(f"[green]✓ Preprocessing complete: {len(enhanced_chunks)} chunks ready[/green]")
        console.print(f"[green]✓ SQL insertion complete: {len(enhanced_chunks)} chunks filled db[/green]")
        console.print(f"  - Duplicates removed: {self.stats['duplicates']}")
        console.print(f"  - Too large (skipped): {self.stats['too_large']}")

        return enhanced_chunks
