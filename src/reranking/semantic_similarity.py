"""
Semantic Similarity Module
Calculate semantic similarity after re-embedding as defined in rag2.mermaid
"""

import numpy as np
from numpy.linalg import norm

from ..embedding.embedder import EmbeddingGenerator


class SemanticSimilarity:
    """Calculate semantic similarity after re-embedding"""

    def __init__(self, embedding_generator: EmbeddingGenerator):
        """
        Initialize with embedding generator

        Args:
            embedding_generator: EmbeddingGenerator instance for creating embeddings
        """
        self.embedding_gen = embedding_generator

    def calculate_similarity(
        self, query_embedding: list[float], candidate_embeddings: list[list[float]]
    ) -> list[float]:
        """
        Calculate cosine similarity between query embedding and candidate embeddings

        Args:
            query_embedding: Embedding vector for the query
            candidate_embeddings: List of embedding vectors for candidates

        Returns:
            List of similarity scores for each candidate
        """
        similarities = []

        # Normalize query embedding
        query_norm = norm(query_embedding)
        if query_norm == 0:
            return [0.0] * len(candidate_embeddings)

        normalized_query = np.array(query_embedding) / query_norm

        for candidate_emb in candidate_embeddings:
            # Normalize candidate embedding
            candidate_norm = norm(candidate_emb)
            if candidate_norm == 0:
                similarities.append(0.0)
                continue

            normalized_candidate = np.array(candidate_emb) / candidate_norm

            # Calculate cosine similarity
            similarity = np.dot(normalized_query, normalized_candidate)
            similarities.append(float(similarity))

        return similarities

    def calculate_similarity_from_text(self, query: str, candidate_texts: list[str]) -> list[float]:
        """
        Calculate semantic similarity directly from text (generates embeddings internally)

        Args:
            query: Query text
            candidate_texts: List of candidate text strings

        Returns:
            List of similarity scores for each candidate
        """
        # Generate embeddings
        query_embedding = self.embedding_gen.generate_embedding(query)
        if not query_embedding:
            return [0.0] * len(candidate_texts)

        candidate_embeddings = []
        for text in candidate_texts:
            emb = self.embedding_gen.generate_embedding(text)
            candidate_embeddings.append(emb if emb else [0] * len(query_embedding))

        return self.calculate_similarity(query_embedding, candidate_embeddings)
