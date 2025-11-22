"""
Relevance Scoring Module
Calculates relevance scores for query-document pairs as defined in rag2.mermaid
"""

from typing import List, Dict, Tuple
import numpy as np


class RelevanceScorer:
    """Calculate relevance scores for query-document pairs"""
    
    def __init__(self):
        pass
    
    def calculate_scores(self, query: str, documents: List[Dict]) -> List[float]:
        """
        Calculate relevance scores for query-document pairs
        
        Args:
            query: The query string
            documents: List of document dictionaries containing text for scoring
            
        Returns:
            List of relevance scores for each document
        """
        scores = []
        
        for doc in documents:
            # Calculate basic relevance metrics
            doc_text = doc.get('text', '')
            score = self._calculate_basic_relevance_score(query.lower(), doc_text.lower())
            scores.append(score)
        
        return scores
    
    def _calculate_basic_relevance_score(self, query: str, doc_text: str) -> float:
        """
        Calculate a basic relevance score based on term overlap and other simple metrics
        
        Args:
            query: Lowercase query string
            doc_text: Lowercase document text
            
        Returns:
            Relevance score between 0 and 1
        """
        # Split query and document into terms
        query_terms = set(query.split())
        doc_terms = set(doc_text.split())
        
        if not query_terms:
            return 0.0
        
        # Calculate term overlap
        overlap = len(query_terms.intersection(doc_terms))
        overlap_score = overlap / len(query_terms) if query_terms else 0.0
        
        # Calculate query term density in document
        doc_tokens = doc_text.split()
        if not doc_tokens:
            return 0.0
        
        query_term_count = sum(1 for token in doc_tokens if token in query_terms)
        density_score = query_term_count / len(doc_tokens)
        
        # Combine scores with weights
        final_score = 0.6 * overlap_score + 0.4 * density_score
        
        # Ensure score is within [0, 1] range
        return min(1.0, max(0.0, final_score))