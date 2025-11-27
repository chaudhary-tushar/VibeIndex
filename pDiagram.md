# RAG Indexing Function Call Diagram

```mermaid
graph TD
    A[cli.ingest] --> B[parse_project]
    B --> C[CodeParser.parse_project]
    C --> D[CodeParser.discover_files]
    D --> E[CodeParser.parse_file]
    E --> F[CodeParser._determine_language]
    F --> G[CodeParser._get_parser]
    G --> H[CodeParser._parse_with_tree_sitter]
    H --> I[CodeParser._get_language_chunks]
    I --> J[CodeParser._enhance_chunks]
    J --> K[Analyzer.enhance_chunk_completely]
    K --> L[CodeParser.run_ctags]
    B --> M[CodeParser.save_results]
    
    N[cli.preprocess] --> O[ChunkPreprocessor.process]
    O --> P[ChunkPreprocessor.deduplicate]
    O --> Q[ChunkPreprocessor.enhance_chunk]
    O --> R[ChunkPreprocessor.validate_chunk]
    
    S[cli.embed] --> T[EmbeddingGenerator.generate_all]
    T --> U[EmbeddingGenerator.generate_batch]
    U --> V[EmbeddingGenerator.generate_embedding_ollama]
    V --> W[EmbeddingGenerator.validate_embedding_quality]
    
    X[cli.index] --> Y[QdrantIndexer.index_chunks]
    Y --> Z[QdrantIndexer._get_collection_for_chunk]
    Z --> AA[QdrantIndexer._prepare_payload]
    AA --> AB[QdrantIndexer.create_collections]
    AB --> AC[QdrantIndexer.optimize_batch_size]
    AB --> Y
    Y --> AD[QdrantClient.upsert]
    
    AE[cli.index_embedded] --> AF[index_from_embedded_json]
    AF --> AG[QdrantIndexer.create_collections]
    AF --> AH[QdrantIndexer.index_chunks]
    
    AI[cli.hybrid_setup] --> AJ[setup_hybrid_collection]
    AJ --> AK[BM25SparseEncoder.build_vocab_from_texts]
    AK --> AL[HybridSearchEngine.create_hybrid_collection]
    AL --> AM[HybridSearchEngine.reindex_with_sparse_vectors]
    
    AN[cli.rag] --> AO[CodeRAG_2.query_codebase]
    AO --> AP[EmbeddingGenerator.generate_embedding_ollama]
    AP --> AQ[QdrantIndexer.client.search]
    AQ --> AR[CodeRAG_2._build_rag_prompt]
    AR --> AS[CodeRAG_2._ask_llm]
    
    AT[cli.advanced_rag] --> AU[CompleteRetrievalSystem.retrieve]
    AU --> AV[HybridSearchEngine.hybrid_search]
    AV --> AW[InitialCandidateSelector.select_candidates]
    AW --> AX[Reranker.rerank]
    AX --> AY[FinalRankingAlgorithm.rank_candidates]
    AY --> AZ[QualityAssuranceFilter.filter_candidates]
```

## Description of the RAG Indexing Process

The RAG indexing process involves several key stages:

1. **Ingestion**: Parsing the project using Tree-sitter to extract code chunks
2. **Preprocessing**: Deduplicating and enhancing chunks for better embeddings
3. **Embedding**: Generating vector embeddings for each chunk using local models
4. **Indexing**: Storing embeddings in Qdrant vector database collections
5. **Retrieval**: Using the indexed data for semantic search queries

The diagram shows the main functions involved at each stage, with arrows indicating function calls between them.