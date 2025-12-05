"""
Batch processing utilities for embedding generation
"""

from .embedder import EmbeddingConfig
from .embedder import EmbeddingGenerator


class BatchProcessor:
    """Handles batch processing of code chunks for embedding"""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.embedder = EmbeddingGenerator(config)

    def process_batches(self, chunks: list[dict], *, parallel: bool = True) -> list[dict]:
        """Process chunks in batches for embedding"""
        return self.embedder.generate_all(chunks, parallel=parallel)

    def validate_batch_sizes(self, chunks: list[dict]) -> bool:
        """Validate that batch sizes are appropriate"""
        total_chunks = len(chunks)
        if total_chunks == 0:
            return False

        batches_needed = (total_chunks + self.config.batch_size - 1) // self.config.batch_size
        print(f"Total chunks: {total_chunks}, Batch size: {self.config.batch_size}, Batches needed: {batches_needed}")
        return True
