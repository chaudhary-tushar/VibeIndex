"""
Cross-Encoder Model for Reranking
Implements cross-encoder model as defined in rag2.mermaid
"""

from sentence_transformers import CrossEncoder


class CrossEncoderModel:
    """Cross-encoder model for reranking"""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder model

        Args:
            model_name: Name of the cross-encoder model to use
        """
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[str]) -> list[tuple]:
        """
        Rerank candidates based on query relevance

        Args:
            query: The query string
            candidates: List of candidate text strings

        Returns:
            List of tuples (index, score) sorted by score in descending order
        """
        # Prepare sentence pairs for cross-encoder
        sentence_pairs = [[query, candidate] for candidate in candidates]

        # Get relevance scores
        scores = self.model.predict(sentence_pairs)

        # Convert to list of (index, score) tuples and sort by score (descending)
        indexed_scores = [(i, float(score)) for i, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        return indexed_scores
