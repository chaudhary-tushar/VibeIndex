"""
Final Ranking Algorithm
Implements weighted score fusion as defined in rag2.mermaid
"""

from typing import List, Dict
import numpy as np


class FinalRankingAlgorithm:
    """Weighted score fusion for final ranking"""
    
    def __init__(self, vector_weight: float = 0.6, keyword_weight: float = 0.3, metadata_weight: float = 0.1):
        """
        Initialize with weights for score fusion
        
        Args:
            vector_weight: Weight for vector similarity scores
            keyword_weight: Weight for keyword relevance scores
            metadata_weight: Weight for metadata-based scores
        """
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.metadata_weight = metadata_weight
        
        # Ensure weights sum to 1.0
        total_weight = vector_weight + keyword_weight + metadata_weight
        if total_weight != 1.0:
            self.vector_weight /= total_weight
            self.keyword_weight /= total_weight
            self.metadata_weight /= total_weight
    
    def rank_candidates(self, candidates: List[Dict], scores: Dict[str, List[float]]) -> List[Dict]:
        """
        Rank candidates using weighted score fusion
        
        Args:
            candidates: List of candidate documents
            scores: Dictionary with different score types for each candidate
                - 'vector_similarity': similarity scores from embedding
                - 'keyword_relevance': relevance scores from keyword matching
                - 'metadata_score': metadata-based scores (optional)
                
        Returns:
            List of candidates ranked by fused scores (highest first)
        """
        fused_scores = []
        
        for i, candidate in enumerate(candidates):
            # Get individual scores (default to 0 if not provided)
            vector_score = scores.get('vector_similarity', [0.0] * len(candidates))[i]
            keyword_score = scores.get('keyword_relevance', [0.0] * len(candidates))[i]
            metadata_score = scores.get('metadata_score', [0.0] * len(candidates))[i]
            
            # Calculate fused score
            fused_score = (
                self.vector_weight * vector_score +
                self.keyword_weight * keyword_score +
                self.metadata_weight * metadata_score
            )
            
            fused_scores.append(fused_score)
        
        # Create list of (candidate, fused_score) pairs and sort by score (descending)
        scored_candidates = list(zip(candidates, fused_scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Return ranked candidates without scores
        return [candidate for candidate, score in scored_candidates]