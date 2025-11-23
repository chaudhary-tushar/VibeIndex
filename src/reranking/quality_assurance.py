"""
Quality Assurance Filter
Implements threshold validation as defined in rag2.mermaid
"""


class QualityAssuranceFilter:
    """Filter based on threshold validation"""

    def __init__(self, min_threshold: float = 0.1):
        """
        Initialize with minimum threshold

        Args:
            min_threshold: Minimum score threshold for candidates to pass
        """
        self.min_threshold = min_threshold

    def filter_candidates(self, candidates: list[dict], scores: list[float]) -> list[dict]:
        """
        Filter candidates based on score threshold

        Args:
            candidates: List of candidate documents
            scores: Scores for each candidate

        Returns:
            List of candidates that meet the threshold
        """
        if len(candidates) != len(scores):
            raise ValueError("Number of candidates must match number of scores")

        filtered_candidates = []

        for candidate, score in zip(candidates, scores):
            if score >= self.min_threshold:
                filtered_candidates.append(candidate)

        return filtered_candidates
