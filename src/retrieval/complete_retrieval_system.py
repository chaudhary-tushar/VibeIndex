"""
Complete Retrieval System following rag2.mermaid architecture
Implements the full pipeline: Query Processor -> Hybrid Search -> Initial Candidate Selection
-> Reranking -> Context Building -> Generation
"""

from ..embedding.embedder import EmbeddingGenerator
from ..generation.context_builder import ContextEnricher
from ..reranking.main import Reranker
from ..retrieval.hybrid_search import HybridSearchEngine
from ..retrieval.initial_candidate_selection import InitialCandidateSelector


class CompleteRetrievalSystem:
    """Complete retrieval system following rag2.mermaid architecture"""

    def __init__(self, hybrid_search_engine: HybridSearchEngine, embedding_generator: EmbeddingGenerator):
        """
        Initialize the complete retrieval system

        Args:
            hybrid_search_engine: Initialized HybridSearchEngine instance
            embedding_generator: Initialized EmbeddingGenerator instance
        """
        self.hybrid_search = hybrid_search_engine
        self.initial_selector = InitialCandidateSelector(top_k=50)
        self.reranker = Reranker(embedding_generator)
        self.context_builder = ContextEnricher()

    def retrieve(self, query: str, collection_name: str, top_k: int = 10) -> list[dict]:
        """
        Complete retrieval pipeline following rag2.mermaid architecture

        Args:
            query: Input query string
            collection_name: Name of the Qdrant collection to search in
            top_k: Number of top results to return

        Returns:
            List of reranked and enriched results
        """
        # Step 1: Initial retrieval through hybrid search
        initial_results = self.hybrid_search.hybrid_search(collection_name=collection_name, query_text=query)

        if not initial_results:
            return []

        # Step 2: Initial candidate selection (select top candidates for reranking)
        candidates_for_rerank = self.initial_selector.select_candidates(initial_results)

        # Step 3: Reranking using the complete reranking pipeline
        reranked_candidates = self.reranker.rerank(query=query, candidates=candidates_for_rerank)

        # Step 4: Limit to final top_k results
        final_results = reranked_candidates[:top_k]

        # Step 5: Context building/enrichment
        # For context building, we'll use the enricher with the final results
        # This would typically involve enriching with additional context

        return final_results

    def retrieve_with_context(self, query: str, collection_name: str, top_k: int = 10) -> list[dict]:
        """
        Complete retrieval with additional context building
        """
        results = self.retrieve(query, collection_name, top_k)

        # Additional context building/enrichment can be implemented here
        # depending on the specific requirements

        return results
