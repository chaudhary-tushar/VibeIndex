# Graph-Agentic-Vector (GAV) RAG Architecture Migration Plan

## Current State Assessment

### Codebase Structure (`src/`)
The existing codebase provides a robust foundation for vector-based RAG:

- **Preprocessing (`src/preprocessing/`)**: Comprehensive pipeline with [`parser.py`](src/preprocessing/parser.py), [`analyzer.py`](src/preprocessing/analyzer.py), [`chunk.py`](src/preprocessing/chunk.py), [`dependency_mapper.py`](src/preprocessing/dependency_mapper.py), and [`preprocessor.py`](src/preprocessing/preprocessor.py). Uses AST parsing for code chunking, metadata extraction, and dependency mapping. See `src/preprocessing/README.md` for details.

- **Embedding (`src/embedding/`)**: [`embedder.py`](src/embedding/embedder.py) with Gamma model (Docker), [`batch_processor.py`](src/embedding/batch_processor.py), and [`quality_validator.py`](src/embedding/quality_validator.py).

- **Retrieval (`src/retrieval/`)**: [`rag_system.py`](src/retrieval/rag_system.py), [`hybrid_search.py`](src/retrieval/hybrid_search.py), [`complete_retrieval_system.py`](src/retrieval/complete_retrieval_system.py). Supports vector similarity, keyword search, metadata filters.

- **Reranking (`src/reranking/`)**: Fully implemented with cross-encoder (Sentence Transformers), semantic similarity, scoring, and quality filters.

- **Generation (`src/generation/`)**: [`generator.py`](src/generation/generator.py), [`context_builder.py`](src/generation/context_builder.py), Mistral model.

- **Configuration (`src/config/`)**: Centralized with [`settings.py`](src/config/settings.py), [`qdrant_config.py`](src/config/qdrant_config.py), [`neo4j_config.py`](src/config/neo4j_config.py) (prepared for graph integration), embedding/LLM configs.

- **CLI/API (`src/cli.py`, `src/app.py`)**: Commands/endpoints for ingest, preprocess, embed, index, enrich, RAG queries.

### Architecture Diagram (`docs/rag2.mermaid`)
```
graph TB
    subgraph "Code Repository" [A,B,C]
    subgraph "Preprocessing Layer" [D,E,F,G]
    subgraph "Embedding Layer" [H,I,J]
    subgraph "Vector Database" [K,L,M]  // Qdrant
    subgraph "Initial Retrieval Layer" [N,O,P]
    subgraph "Reranking Layer" [Q-U,V]  // Fully implemented
    subgraph "Generation Layer" [W,X,Y]
```
Key flows: Source → Preprocess → Embed → Qdrant → Hybrid Retrieve → Rerank → Generate.

**Strengths**: Mature vector RAG pipeline, chunk-aware preprocessing, hybrid search readiness, Neo4j config stub.

## Gaps and Opportunities

### Gaps
- **No Graph Layer**: Chunks have rich structure (dependencies, references, defines, relationships) but not persisted/queried as graph. `neo4j_config.py` exists but unused.
- **No Agentic Intelligence**: Linear retrieval-generation; no multi-step reasoning, tool-use, or query routing.
- **Limited Hybrid Search**: Vector+keyword only; no vector-graph fusion.
- **No Multimodal**: No frontend snapshots (Playwright/CDP) despite docs guidance.

### Opportunities (Leveraging Docs)
- **Graph from Chunks** (`docs/rag_graph_agent_pipeline.md`): "Your chunks already contain... dependencies, references, defines... enough to build Graph RAG WITHOUT ANY PARSERS."
- **Agentic Tools** (`docs/full_rag_system_implementation_guide.md`): Vector/graph/snapshot tools for LangChain/LangGraph agents.
- **Phased Expansion** (`docs/expansion_steps.md`): Aligns with existing pipeline.
- **LangGraph Integration** (`docs/graph-plan.md`): Agent state with vector/graph retrievers.

**Libraries** (standard best practices, configs exist):
- Neo4j (driver via `neo4j_config.py`)
- LangChain/LangGraph for agents
- Qdrant hybrid enhancements
- Playwright for snapshots (Phase 4)

## Phased Migration Roadmap

### Phase 1: Vector Foundations (1-2 weeks, Low Risk)
**Goal**: Solidify/enhance existing vector pipeline.

**Sub-steps/Tasks**:
1. Optimize Qdrant hybrid search (add metadata filters). Effort: Low. Deps: Existing. Success: >95% recall on benchmarks.
2. Implement/enhance chunk enrichment (`src/generation/enrich`). Effort: Medium. Tools: Existing Mistral.
3. Add evaluation metrics (retrieval accuracy). Effort: Low.

**Risks/Mitigations**: Minimal disruption—extend existing. Test isolation.

**Code Changes** (new files):
- `src/retrieval/evaluator.py`
- `src/embedding/optimizer.py`

### Phase 2: Graph Knowledge Layer (2-3 weeks, Medium Risk)
**Goal**: Build Neo4j graph from chunks.

**Sub-steps/Tasks** (per `docs/rag_graph_agent_pipeline.md`):
1. Implement graph builder: Chunk → Nodes (:Chunk, :Symbol), Edges (DEPENDS_ON, REFERENCES, etc.). Effort: Medium. Deps: Phase 1.
2. Integrate ingestion: Post-embedding → Neo4j upsert. Effort: Medium. Lib: neo4j-driver.
3. Graph queries: Cypher for traversal (e.g., "dependency chain"). Effort: Low.

**Success Criteria**: 100% chunk-to-node sync, <1s traversals.

**Risks/Mitigations**: Schema drift—use MERGE; perf—indexes/constraints.

**Code Changes** (new):
- `src/graph/builder.py`
- `src/graph/neo4j_client.py` (extend config)
- `src/graph/retriever.py`
- Update CLI: `ingest_graph`

### Phase 3: Agentic Orchestration (3-4 weeks, High Risk)
**Goal**: Multi-agent workflow with tools.

**Sub-steps/Tasks** (per `docs/full_rag_system_implementation_guide.md`, `docs/graph-plan.md`):
1. Define tools: `vector_search`, `graph_neighbors`, `fetch_chunk`. Effort: Medium. Lib: LangChain tools.
2. Build LangGraph: StateGraph with router (query type → vector/graph/both). Effort: High. Nodes: retrieve_vector, retrieve_graph, grade_docs, generate.
3. Agents: Retriever Agent, Graph Navigator, Synthesizer. Effort: High.

**Success Criteria**: Agent resolves 80% complex queries (e.g., "trace deps").

**Risks/Mitigations**: Hallucination—grounding prompts; loops—max iterations.

**Code Changes** (new):
- `src/agents/tools.py`
- `src/agents/planning_agent.py`
- `src/agents/agentic_rag.py`
- Update `rag_system.py`: Hybrid → Agentic.

### Phase 4: Integration & Optimization (2 weeks, Medium Risk)
**Goal**: Full GAV fusion, snapshots, prod-ready.

**Sub-steps/Tasks**:
1. Hybrid retriever: Vector + Graph scores. Effort: Low.
2. Add snapshots (Playwright). Effort: Medium. Per `docs/full_rag_system_implementation_guide.md`.
3. Optimize: Caching, async, monitoring. Effort: Low.
4. Evals: End-to-end benchmarks.

**Success Criteria**: <2s latency, >90% accuracy uplift.

**Risks/Mitigations**: Data bloat—pruning; perf—profiling.

**Code Changes** (new):
- `src/render/snapshotter.py`
- `src/retrieval/hybrid_gav.py`
- `src/optimization/cache.py`

## References
- "`docs/full_rag_system_implementation_guide.md`": Full phases, diagram.
- "`docs/rag_graph_agent_pipeline.md`": "Build graph from chunk fields: dependencies → DEPENDS_ON".
- "`docs/expansion_steps.md`": Detailed steps, risks.
- "`docs/graph-plan.md`": LangGraph agent.

## High-Level Timeline
| Phase | Duration | Cumulative |
|-------|----------|------------|
| 1     | 1-2w    | 2w        |
| 2     | 2-3w    | 5w        |
| 3     | 3-4w    | 9w        |
| 4     | 2w      | 11w       |

**Total: ~11 weeks**

## Next-Action Checklist
- [ ] Review/approve this plan
- [ ] Switch to `code` mode for Phase 1 impl
- [ ] Setup Neo4j instance
- [ ] Run benchmarks on current vector RAG
- [ ] Create `docs/architecture.mermaid` (next)
