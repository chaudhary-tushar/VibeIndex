"""
Initial Candidate Selection
Select top-K results as defined in rag2.mermaid architecture
"""


class InitialCandidateSelector:
    """Select top-K candidates for reranking"""

    def __init__(self, top_k: int = 50):
        """
        Initialize with top-K parameter

        Args:
            top_k: Number of top candidates to select for reranking
        """
        self.top_k = top_k

    def select_candidates(self, search_results: list[dict]) -> list[dict]:
        """
        Select top-K results for reranking

        Args:
            search_results: List of search results with scores

        Returns:
            Top-K candidates selected for reranking
        """
        # Sort by score if available (assuming higher scores are better)
        if search_results and "score" in search_results[0]:
            sorted_results = sorted(search_results, key=lambda x: x.get("score", 0), reverse=True)
        else:
            # If no score is available, return as is
            sorted_results = search_results

        # Return top-K candidates
        return sorted_results[: self.top_k]
