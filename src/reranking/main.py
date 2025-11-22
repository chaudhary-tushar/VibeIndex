"""
Main Reranker Module
Implements the complete reranking pipeline as defined in rag2.mermaid
"""

from typing import List, Dict, Optional
from .input_processor import RerankingInputProcessor
from .cross_encoder import CrossEncoderModel
from .scoring import RelevanceScorer
from .semantic_similarity import SemanticSimilarity
from .final_ranking import FinalRankingAlgorithm
from .quality_assurance import QualityAssuranceFilter
from ..embedding.embedder import EmbeddingGenerator


class Reranker:
    """Main reranking orchestrator"""
    
    def __init__(self, embedding_generator: EmbeddingGenerator):
        """
        Initialize the reranker with all required components
        
        Args:
            embedding_generator: EmbeddingGenerator instance for semantic similarity
        """
        self.input_processor = RerankingInputProcessor()
        try:
            self.cross_encoder = CrossEncoderModel()
        except Exception:
            # Fallback if cross-encoder model is not available
            self.cross_encoder = None
            print("Warning: Cross-encoder model not available. Using basic reranking.")
        
        self.scorer = RelevanceScorer()
        self.semantic_similarity = SemanticSimilarity(embedding_generator)
        self.ranking_algo = FinalRankingAlgorithm()
        self.qa_filter = QualityAssuranceFilter()
    
    def rerank(self, query: str, candidates: List[Dict], context: Optional[Dict] = None) -> List[Dict]:
        """
        Complete reranking pipeline
        
        Args:
            query: The original query string
            candidates: List of candidate documents to rerank
            context: Optional additional context information
            
        Returns:
            List of reranked candidates
        """
        if not candidates:
            return []
        
        # Step 1: Process input (context + candidates)
        processed_candidates = self.input_processor.process_input(query, candidates, context)
        
        if self.cross_encoder:
            # Step 2: Cross-encoder reranking (if model is available)
            candidate_texts = [pc['text'] for pc in processed_candidates]
            cross_encoder_scores = self.cross_encoder.rerank(query, candidate_texts)
            
            # Reorder candidates based on cross-encoder scores
            ordered_indices = [idx for idx, score in cross_encoder_scores]
            reordered_candidates = [candidates[i] for i in ordered_indices]
            reordered_processed = [processed_candidates[i] for i in ordered_indices]
        else:
            # If cross-encoder is not available, proceed with basic scoring
            reordered_candidates = candidates
            reordered_processed = processed_candidates
        
        # Step 3: Calculate relevance scores
        relevance_scores = self.scorer.calculate_scores(query, reordered_processed)
        
        # Step 4: Calculate semantic similarity scores
        candidate_texts = [pc['text'] for pc in reordered_processed]
        semantic_scores = self.semantic_similarity.calculate_similarity_from_text(query, candidate_texts)
        
        # Step 5: Combine scores for final ranking
        scores = {
            'keyword_relevance': relevance_scores,
            'vector_similarity': semantic_scores
        }
        
        # Step 6: Apply final ranking algorithm
        ranked_candidates = self.ranking_algo.rank_candidates(reordered_candidates, scores)
        
        # Step 7: Apply quality assurance filter
        # For this step, we'll calculate combined scores again
        processed_for_qa = self.input_processor.process_input(query, ranked_candidates, context)
        qa_relevance_scores = self.scorer.calculate_scores(query, processed_for_qa)
        qa_semantic_scores = self.semantic_similarity.calculate_similarity_from_text(query, [pc['text'] for pc in processed_for_qa])
        
        # Calculate combined scores for QA filtering
        combined_scores = []
        for rel_score, sem_score in zip(qa_relevance_scores, qa_semantic_scores):
            combined_score = (rel_score + sem_score) / 2  # Simple average
            combined_scores.append(combined_score)
        
        filtered_candidates = self.qa_filter.filter_candidates(ranked_candidates, combined_scores)
        
        return filtered_candidates