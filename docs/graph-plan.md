# Plan: Integrating a Knowledge Graph and Agentic RAG

This document outlines a phased plan to evolve the existing RAG system into a more powerful, graph-aware, and agentic architecture. The goal is to leverage the rich structure of the codebase to provide more accurate and context-aware answers.

## Current State Analysis

-   **Indexing (`scripts/ingest.py`):** The current process is a placeholder for parsing and embedding code chunks, intended for storage in a Qdrant vector database.
-   **Data Structure (`data/build/chunks.json`):** A highly detailed and well-structured JSON output exists, representing code as typed and contextualized chunks. This structure is a massive advantage, containing file paths, line numbers, dependencies, and relationship placeholders. This is the `Codechunk` structure we will build upon.
-   **Retrieval (`src/retrieval/rag_system.py`, `src/retrieval/main.py`):** The system uses a standard vector-based RAG approach. It finds code chunks based on semantic similarity to the query but does not leverage the explicit relationships between code objects.

## Integration Assessment

**Overall Feasibility Score:** 9/10

**Phase-by-Phase Fit Summary:**
- **Phase 0:** ✅ Excellent - Leverages existing preprocessing.
- **Phase
- **Phase 1:** ✅ Strong - Builds on enhanced chunks.
- **Phase 2:** ✅ Good
- **Phase 3:** ✅ Excellent - Agentic hybrid.


### Phase 0: Implement Ingestion Pipeline

Update [`scripts
**Key Strengths:**
- Reuse of preprocessing like [`dependency_mapper.py`](src/preprocessing/dependency_mapper.py), [`analyzer.py`](
## Revised Implementation Plan

### Phase 1: Knowledge Graph Construction

The goal of this phase is to transform the detailed `Codechunk` data into a queryable knowledge graph, making the implicit relationships between code objects explicit.

1.  **Refine and Execute the Ingestion Pipeline:**
    *   Update `scripts/ingest.py` to produce the rich `Codechunk` structure seen in `data/build/chunks.json`.
    *   **Implement Relationship Extraction:** This is the most critical step. After parsing the initial chunks, run a second analysis pass to populate the `relationships` object in each chunk.
        *   **Parent/Child:** Link functions and methods to their parent classes or files.
        *   **Calls/References:** For each function chunk, identify the `id`s of other function chunks it calls.
        *   **Dependencies:** Link chunks to the `id`s of the modules or packages they import.

2.  **Choose and Set Up a Graph Database:**
    *   **Recommendation:** Use **Neo4j** or a similar native graph database that supports property graphs and has a robust query language (like Cypher).
    *   **Action:** Set up a local Neo4j instance (e.g., via Docker).

3.  **Implement the Graph Ingestion Script:**
    *   Create a new script, e.g., `scripts/ingest_graph.py`.
    *   This script will:
        1.  Read the fully processed `chunks.json` file (with populated relationships).
        2.  Connect to the graph database.
        3.  **Create Nodes:** Iterate through each chunk and create a node in the graph. The node's label should be its `type` (e.g., `:Function`, `:Class`, `:File`), and its properties should be the chunk's metadata (e.g., `name`, `file_path`, `code`, `qualified_name`).
        4.  **Create Relationships:** For each node, iterate through its populated `relationships` fields and create the corresponding edges in the graph (e.g., `(func_a)-[:CALLS]->(func_b)`).

### Phase 2: Graph-Native Retrieval (Graph RAG)

This phase focuses on creating a retriever that can query the knowledge graph to fetch contextually relevant information that vector search alone would miss.

1.  **Design Graph Query Strategies:**
    *   Identify common query patterns. For example, a question like "How is the user model used?" could be translated into a graph query that finds the `User` model node and then traverses outgoing `REFERENCED_BY` relationships.
    *   **LLM-to-Graph-Query:** Develop a component that uses an LLM to translate a natural language question into a Cypher query. This can be done with a few-shot prompt that provides examples of questions and their corresponding Cypher queries.

2.  **Create a `GraphRetriever` Tool:**
    *   Create a new Python class `GraphRetriever` in a file like `src/retrieval/graph_search.py`.
    *   This class will have a method like `retrieve(query: str) -> List[CodeChunk]`.
    *   The method will:
        1.  Take the user's plain text query.
        2.  Use the LLM-to-Graph-Query component to generate a Cypher query.
        3.  Execute the Cypher query against the graph database.
        4.  Process the results from the database, retrieving the full code and metadata for the matched nodes, and return them in the same format as the current Qdrant retriever.

### Phase 3: Building the Agentic RAG with LangGraph

This phase integrates the new graph retriever into an intelligent agent that can decide which retrieval strategy is best for a given query.

1.  **Define the Agent State:**
    *   Create a `GraphState` TypedDict. This will be the central object that flows through the agent.
    *   `GraphState` will include keys like `question: str`, `documents: List[Document]`, `rag_strategy: str`, and `generation: str`.

2.  **Build the Agent's Tools (Nodes):**
    *   In `LangGraph`, each tool is a node. You will define the following nodes:
        *   `retrieve_vector`: A node that calls your existing Qdrant vector search (`CodeRAG_2`).
        *   `retrieve_graph`: A node that calls the new `GraphRetriever` from Phase 2.
        *   `generate`: A node that takes the final set of documents and the question to generate an answer (similar to the final step in `CodeRAG_2`).
        *   `grade_documents`: A node that uses an LLM to evaluate the relevance of retrieved documents (as seen in the LangGraph documentation).

3.  **Define the Agent's Logic (Edges):**
    *   This is where the "agentic" behavior comes from. You will define conditional edges that decide the path through the graph based on the current state.
    *   **Example Workflow:**
        1.  **Start:** The user's `question` enters the graph.
        2.  **Initial Retrieval:** The agent could start by calling `retrieve_vector` for a quick semantic search.
        3.  **Grade Relevance:** The `grade_documents` node checks if the results are good enough.
        4.  **Conditional Edge (Router):**
            *   **If** the documents are relevant, proceed directly to the `generate` node.
            *   **If** the documents are not relevant or a specific keyword (e.g., "how is this used", "who calls this") is in the query, the agent should route to the `retrieve_graph` node. The results can then be passed to `generate`.
            *   The agent could even be designed to combine the results from both retrievers before generation.

4.  **Assemble and Run the Graph:**
    *   Use `langgraph.graph.StateGraph` and `compile()` to build the final, executable agent.
    *   Replace the linear `query_codebase` method in `rag_system.py` with a call to your new LangGraph agent.

By following these phases, you will incrementally build a highly sophisticated RAG system that understands the deep structure of your code and can reason about the best way to answer complex questions.
