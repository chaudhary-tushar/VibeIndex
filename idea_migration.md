# Migration Plan: Updating .idea to rag2.mermaid Architecture

This document outlines the steps and considerations for updating the project's architecture documentation from `Rag.mermaid` to `rag2.mermaid`.

The file at `.idea` is not a typical IDE configuration directory but a text file containing a description of the project's RAG architecture. The migration involves updating this file to reflect the more advanced architecture defined in `rag2.mermaid`.

## Key Architectural Differences

The `rag2.mermaid` architecture introduces several key improvements over the older `Rag.mermaid` design:

1.  **Embedding Quality Validator**: A new component in the Embedding Layer to ensure the quality of generated embeddings.
2.  **Expanded Retrieval Process**: The original "Retrieval Layer" is now split into two distinct layers:
    *   **Initial Retrieval Layer**: Responsible for the initial candidate selection using hybrid search.
    *   **Reranking Layer**: A new, sophisticated layer to rerank the initial candidates for better relevance.
3.  **Detailed Reranking Layer**: The new Reranking Layer includes multiple components for advanced relevance tuning:
    *   Reranking Input Processor
    *   Cross-Encoder Model
    *   Relevance Scoring
    *   Semantic Similarity (Re-embedding)
    *   Final Ranking Algorithm
    *   Quality Assurance Filter

## Work Required for `.idea` Migration

To align the `.idea` file with the `rag2.mermaid` architecture, the following changes are needed:

1.  **Update the Architecture Description**: The textual description in the `.idea` file must be updated to include the new layers and components.
2.  **Add the Embedding Quality Validator**: The description of the Embedding Layer should be updated to include the "Embedding Quality Validator".
3.  **Restructure the Retrieval Process**: The description should be modified to show the split of the retrieval process into "Initial Retrieval Layer" and "Reranking Layer".
4.  **Detail the Reranking Layer**: A new section should be added to describe the components and workflow of the new "Reranking Layer".

This migration is primarily a documentation task to ensure the `.idea` file accurately reflects the current architectural goals of the project. It also serves as a clear indicator of the implementation work that is still required, particularly for the unimplemented Reranking Layer.

## Remaining Implementation for Full Architecture

Besides the unimplemented **Reranking Layer**, several other components are only partially implemented and require further work to be considered complete.

### Preprocessing Layer (`src/preprocessing`)

*   **`metadata_extractor.py`**: The metadata extraction is basic. It could be enhanced with more sophisticated analysis, for example, by improving the `has_tests` check to look for test files that import the chunk's file or function.
*   **`dependency_mapper.py`**: The dependency analysis is based on heuristics (regex) and is not always accurate. It could be improved by using full Abstract Syntax Tree (AST) traversal for all languages to build a more robust dependency graph.

### Embedding Layer (`src/embedding`)

*   **`quality_validator.py`**: This file is empty. The validation logic currently exists in `embedder.py` but should be refactored into a dedicated `EmbeddingQualityValidator` class in this file to align with the architecture.

### Initial Retrieval Layer (`src/retrieval`)

*   **`query_processor.py`**: This file is empty and needs to be implemented. A complete implementation would handle query parsing, cleaning, expansion (with synonyms or related terms), intent identification, and query embedding.
*   **`candidate_selection.py`**: This file is empty. A complete implementation would handle the selection and preparation of candidates from the hybrid search results for the reranking layer.

### Generation Layer (`src/generation`)

*   **`context_builder.py`**: The current implementation is for pre-processing (enriching chunks with summaries). The query-time context builder, which assembles the retrieved and reranked chunks into a final context for the LLM, is missing.
*   **`prompt_constructor.py`**: Similar to the context builder, the current implementation is for pre-processing. The query-time prompt constructor, which takes the user's query and the final context to build the prompt for the LLM, is missing.
*   **`generator.py`**: The `LLMClient` is generic. It should be more tightly integrated with the rest of the generation layer to handle the specific inputs (e.g., structured context) and outputs of the RAG task.