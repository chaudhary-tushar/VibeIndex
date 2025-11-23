"""
Reranking Input Processor
Processes context + candidates for reranking as defined in rag2.mermaid
"""


class RerankingInputProcessor:
    """
    Process reranking inputs - context + candidates for reranking
    """

    def __init__(self):
        """
        Initialize the RerankingInputProcessor.
        Currently no specific initialization required.
        """
        # Add any initialization code here if needed in the future

    def process_input(self, query: str, candidates: list[dict], context: dict | None = None) -> list[dict]:
        """
        Prepare candidates for reranking

        Args:
            query: The original query string
            candidates: List of candidate documents/chunks
            context: Optional additional context information

        Returns:
            Processed candidates ready for reranking
        """
        processed_candidates = []

        for candidate in candidates:
            # Extract relevant text from candidate for reranking
            candidate_text_parts = [
                candidate.get("qualified_name", ""),
                candidate.get("name", ""),
                candidate.get("docstring", ""),
                candidate.get("code", ""),
                candidate.get("signature", ""),
            ]

            # Join non-empty parts
            candidate_text = " ".join(filter(None, candidate_text_parts))

            processed_candidate = {
                "id": candidate.get("id"),
                "text": candidate_text,
                "original_candidate": candidate,
                "query": query,
            }

            processed_candidates.append(processed_candidate)

        return processed_candidates
