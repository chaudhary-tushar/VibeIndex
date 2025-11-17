"""
Enhanced preprocessing for code chunks before embedding and indexing
"""

import json
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from rich.console import Console

console = Console()


@dataclass
class ChunkPreprocessor:
    """Enhanced preprocessing for code chunks before embedding"""

    dedup_hashes: set = None
    stats: Dict[str, int] = None

    def __post_init__(self):
        if self.dedup_hashes is None:
            self.dedup_hashes = set()
        if self.stats is None:
            self.stats = {
                'total': 0,
                'duplicates': 0,
                'enhanced': 0,
                'too_large': 0,
            }

    def deduplicate(self, chunks: List[Dict]) -> List[Dict]:
        """Remove duplicate chunks based on code content"""
        unique_chunks = []

        for chunk in chunks:
            # Create hash from code content
            code_hash = hashlib.md5(chunk['code'].encode()).hexdigest()

            if code_hash not in self.dedup_hashes:
                self.dedup_hashes.add(code_hash)
                unique_chunks.append(chunk)
            else:
                self.stats['duplicates'] += 1

        console.print(f"[yellow]Removed {self.stats['duplicates']} duplicate chunks[/yellow]")
        return unique_chunks

    def enhance_chunk(self, chunk: Dict) -> Dict:
        """Enhance chunk with additional context for better embeddings"""
        enhanced = chunk.copy()

        # Build rich text representation for embedding
        parts = []

        # 1. Add contextual prefix
        context_prefix = f"# {chunk['language'].upper()} {chunk['type'].upper()}"
        if chunk.get('qualified_name'):
            context_prefix += f": {chunk['qualified_name']}"
        parts.append(context_prefix)

        if isinstance(chunk.get('context'), str):
            parts.append(f"html/css context: {chunk.get('context')}")
        else:
            # 2. Add file context
            if chunk.get('context', {}).get('file_hierarchy'):
                file_ctx = " > ".join(chunk['context']['file_hierarchy'])
                parts.append(f"# Location: {file_ctx}")

            # 3. Add domain context
            if chunk.get('context', {}).get('domain_context'):
                parts.append(f"# Purpose: {chunk['context']['domain_context']}")

        # 4. Add docstring if available
        if chunk.get('docstring'):
            parts.append(f'"""{chunk["docstring"]}"""')

        # 5. Add signature for functions
        if chunk.get('signature'):
            parts.append(chunk['signature'])

        # 6. Add decorators/metadata
        if chunk.get('metadata', {}).get('decorators'):
            parts.extend(chunk['metadata']['decorators'])

        # 7. Add the actual code
        parts.append(chunk['code'])

        # 8. Add dependencies as comments
        if chunk.get('dependencies'):
            deps = ", ".join(chunk['dependencies'][:5])  # Limit to 5
            parts.append(f"# Dependencies: {deps}")

        # 9. Add defined symbols
        if chunk.get('defines'):
            parts.append(f"# Defines: {', '.join(chunk['defines'])}")

        # Combine all parts
        enhanced['embedding_text'] = "\n".join(parts)
        enhanced['embedding_text_length'] = len(enhanced['embedding_text'])

        self.stats['enhanced'] += 1
        return enhanced

    def validate_chunk(self, chunk: Dict, max_tokens: int = 8192) -> bool:
        """Validate chunk is suitable for embedding"""
        # Rough token estimate (1 token ≈ 4 chars)
        estimated_tokens = len(chunk.get('embedding_text', chunk['code'])) // 4

        if estimated_tokens > max_tokens:
            self.stats['too_large'] += 1
            return False

        return True

    def process(self, chunks: List[Dict]) -> List[Dict]:
        """Full preprocessing pipeline"""
        self.stats['total'] = len(chunks)
        console.print(f"[cyan]Starting preprocessing of {len(chunks)} chunks...[/cyan]")

        # 1. Deduplicate
        chunks = self.deduplicate(chunks)

        # 2. Enhance each chunk
        enhanced_chunks = []
        for chunk in tqdm(chunks, desc="Enhancing chunks", unit="chunk"):
            enhanced = self.enhance_chunk(chunk)
            if self.validate_chunk(enhanced):
                enhanced_chunks.append(enhanced)

        console.print(f"[green]✓ Preprocessing complete: {len(enhanced_chunks)} chunks ready[/green]")
        console.print(f"  - Duplicates removed: {self.stats['duplicates']}")
        console.print(f"  - Too large (skipped): {self.stats['too_large']}")

        return enhanced_chunks