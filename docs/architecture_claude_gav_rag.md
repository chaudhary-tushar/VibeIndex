# 📘 Graph-Agentic-Vector RAG Migration Plan for geminIndex

**Document Version**: 1.0
**Date**: December 2, 2025
**Target Architecture**: GAV (Graph + Agentic + Vector) RAG System
**Based on**: Current geminIndex codebase analysis + expansion_steps.md + full_rag_system_implementation_guide.md + rag_graph_agent_pipeline.md

---

## 1. Executive Summary

The geminIndex project is a **mature, modular RAG system** with strong preprocessing, embedding, and retrieval foundations. This document outlines a phased migration path to expand the system into a **Graph-Agentic-Vector (GAV) RAG architecture** that combines:

- **Vector RAG**: Current semantic search (Qdrant + hybrid retrieval) ✅
- **Graph RAG**: Knowledge graph layer (Neo4j + topological reasoning) ⚠️ Planned
- **Agentic RAG**: Multi-step reasoning with tool orchestration (LangChain + LangGraph) ⚠️ Planned

**Current State**:
- ✅ Sophisticated preprocessing (Tree-sitter + AST analysis)
- ✅ Quality-validated embeddings (768-dim, Ollama/OpenAI)
- ✅ Hybrid vector retrieval (dense + BM25 sparse)
- ✅ Production-grade reranking (cross-encoder)
- ✅ Neo4j config exists but **unused**
- ❌ No graph layer implementation
- ❌ Limited agentic capabilities

**Estimated Effort**: 12–18 weeks for full GAV implementation

---

## 2. Current State Assessment

### 2.1 Architecture Overview

```
Source Code
    ↓
Preprocessing (Parser + Analyzer + DependencyMapper)
    ↓
CodeChunk Objects (rich metadata: dependencies, references, defines)
    ↓
Embedding (768-dim, Ollama/OpenAI)
    ↓
Qdrant Indexing (Dense + Sparse vectors)
    ↓
Hybrid Retrieval (0.7 dense + 0.3 BM25)
    ↓
Reranking (Cross-Encoder: ms-marco-MiniLM-L-6-v2)
    ↓
Generation (Ollama/OpenAI LLM)
```

### 2.2 Key Components

#### **Preprocessing Layer** (Layers 1–2)
- **Parser** (`src/preprocessing/parser.py`, 322 LOC)
  - Tree-sitter + language-specific extractors
  - Supports: Python (libCST), JavaScript, HTML, CSS, Markdown, PDF
  - Outputs: CodeChunk objects with fine-grained metadata
  - Status: ✅ **Complete & Robust**

- **Analyzer** (`src/preprocessing/analyzer.py`, 1,767 LOC)
  - AST walking, decorator extraction, function/class/method extraction
  - Nested element detection, selector extraction
  - Metadata enrichment, complexity analysis
  - Status: ✅ **Complete & Well-tested**

- **DependencyMapper** (`src/preprocessing/dependency_mapper.py`, 234 LOC)
  - Python/JS import extraction, HTML/CSS dependency linking
  - Symbol usage detection
  - Status: ✅ **Functional**

- **MetadataExtractor** (`src/preprocessing/metadata_extractor.py`, 114 LOC)
  - Docstring extraction, signature parsing, complexity metrics (radon)
  - Status: ✅ **Complete**

- **ChunkPreprocessor** (`src/preprocessing/preprocessor.py`, 100 LOC)
  - Deduplication (MD5 hash), enrichment (embedding_text construction)
  - Token count validation (max 8,192)
  - Status: ✅ **Production-ready**

#### **Embedding Layer** (Layer 3)
- **EmbeddingGenerator** (`src/embedding/embedder.py`, 294 LOC)
  - Providers: Ollama (local), OpenAI (API)
  - Quality validation (dimension, magnitude, variance, NaN checks)
  - Batch processing with retry logic
  - Status: ✅ **Mature & Validated**

#### **Retrieval Layer** (Layer 4)
- **QdrantIndexer** (`src/retrieval/search.py`, 596 LOC)
  - 3 collections: functions, classes, modules
  - Dense vectors (COSINE distance), sparse vectors (BM25)
  - Health checks, collection management
  - Status: ✅ **Sophisticated**

- **HybridSearch** (`src/retrieval/hybrid_search.py`, 545 LOC)
  - Weighted fusion (0.7 dense + 0.3 sparse)
  - Top-K selection, score thresholding
  - Configurable parameters (k1, b, weights)
  - Status: ✅ **Advanced**

#### **Reranking Layer** (Layer 5)
- **CrossEncoder** (`src/reranking/cross_encoder.py`, 40 LOC)
  - Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Sentence pair scoring
  - Status: ✅ **Integrated**

#### **Generation Layer** (Layer 6)
- **LLMClient** (`src/generation/generator.py`, 60 LOC)
  - Ollama + OpenAI support
  - Async batch processing
  - Status: ✅ **Functional**

- **ContextEnricher** (`src/generation/context_builder.py`, 120+ LOC)
  - AI-generated summaries with SQLite caching
  - Symbol index integration
  - Status: ⚠️ **Needs Enhancement**

#### **Configuration** (`src/config/`)
- **EmbeddingConfig**: Validated settings for embeddings
- **QdrantConfig**: Vector database configuration
- **Neo4jConfig**: **Exists but UNUSED** ⚠️
- **Settings**: Centralized configuration with .env support
- Status: ✅ **Well-structured**

### 2.3 Key Data Structures

#### **CodeChunk** (Foundation for GAV)
```python
@dataclass
class CodeChunk:
    # Identification
    type: str                    # 'function', 'class', 'method', etc.
    name: str
    code: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    id: str                      # UUID
    qualified_name: str | None

    # GRAPH-CRITICAL FIELDS
    dependencies: list[str]      # External + internal imports
    references: list[str]        # Symbols called/used
    defines: list[str]           # Symbols defined

    # Rich metadata
    metadata: dict[str, Any]     # Decorators, access modifiers
    documentation: dict[str, Any]# Docstrings, summaries
    analysis: dict[str, Any]     # Complexity, token count, hash
    relationships: dict[str, Any]# CRITICAL: Imports, children, parents
    context: dict[str, Any]      # Module, project, domain hierarchy
```

**Why This Matters**: The `dependencies`, `references`, `defines`, and `relationships` fields already contain the graph structure needed for Neo4j—no additional parsing required.

### 2.4 Dependencies (Current)

| Category | Package | Status |
|----------|---------|--------|
| **Core** | fastapi, uvicorn, pydantic, python-dotenv | ✅ |
| **Parsing** | tree-sitter, libcst, pathspec | ✅ |
| **Embedding** | transformers, langchain, langchain-ollama, langchain-openai | ✅ |
| **Vector DB** | qdrant-client | ✅ |
| **Analysis** | radon | ✅ |
| **CLI/UI** | click, rich | ✅ |
| **Graph DB** | neo4j | ❌ **MISSING** |
| **Agents** | (LangChain hooks exist, tools not implemented) | ⚠️ |
| **Browser** | playwright | ❌ **Optional** |

---

## 3. Gaps & Opportunities for GAV Expansion

### 3.1 Graph RAG Layer—NOT IMPLEMENTED

**Current Gap**: Neo4j configuration exists but no graph builder, query engine, or integration.

**Opportunities**:
- Convert rich CodeChunk metadata → Neo4j property graph
- Implement topological retrieval (graph traversal + neighbor expansion)
- Build graph-aware reranking (connectivity + semantic similarity)
- Support complex queries: "Find all code that affects X", "Trace dependency chains"

**New Modules Required**:
```
src/graph/                      # NEW
├── graph_schema.py            # Node/relationship type definitions
├── graph_builder.py           # CodeChunk → Neo4j converter
├── neo4j_client.py            # CRUD wrapper around neo4j driver
├── query_engine.py            # Graph traversal queries
├── graph_retriever.py         # Graph-based result fetching
└── graph_aware_reranker.py    # Reranking with graph metrics
```

**Effort**: ⭐⭐⭐⭐ (HIGH, 2–3 weeks)

### 3.2 Agentic RAG Layer—PARTIALLY PREPARED

**Current Gap**: LangChain integration exists, but no agent executor, tools, or reasoning loops.

**Opportunities**:
- Build planning agent for multi-step code understanding
- Implement domain-specific tools (code search, graph query, dependency analyzer, test locator)
- Support complex workflows: debugging, impact analysis, refactoring suggestions
- Enable conversation history + memory management

**New Modules Required**:
```
src/agents/                     # NEW
├── agent_executor.py          # Main agent loop
├── tools/
│   ├── code_search_tool.py    # Semantic search (uses HybridSearch)
│   ├── graph_query_tool.py    # Graph traversal queries
│   ├── dependency_analyzer_tool.py
│   ├── test_locator_tool.py
│   ├── documentation_tool.py
│   └── tool_registry.py       # Tool management
├── memory/
│   └── conversation_memory.py # Conversation history + context
└── planning/
    └── planner_agent.py       # Multi-step reasoning
```

**Effort**: ⭐⭐⭐⭐⭐ (VERY HIGH, 3–4 weeks)

### 3.3 Rendering/Snapshot Layer—OPTIONAL

**Current Gap**: No browser automation for frontend debugging.

**Opportunities** (Nice-to-have):
- Capture DOM snapshots, computed styles, screenshots
- Link frontend code to rendered output
- Support visual debugging workflows

**Estimated Effort**: ⭐⭐⭐⭐ (HIGH, 2–3 weeks)

---

## 4. Phased Migration Roadmap

### **Phase 1: Vector Foundations & Graph Setup** (1–2 weeks)
**Objective**: Prepare vector layer for graph integration; implement basic graph infrastructure.

#### 1.1 Enhance Vector Retrieval Configuration
**Tasks**:
- [ ] Add graph retrieval option flags to `HybridSearchConfig`
- [ ] Extend `RetrievalConfig` to support hybrid vector-graph mode
- [ ] Document retrieval strategy options in configuration

**Effort**: 2–3 days | **Dependencies**: None | **Files**:
- `src/retrieval/hybrid_search.py` (extend config)
- `src/config/settings.py` (add graph retrieval flags)

**Success Criteria**:
- Configuration supports both vector-only and vector+graph modes
- CLI accepts `--retrieval-mode` flag (vector | graph | hybrid)

---

#### 1.2 Implement Neo4jClient Wrapper
**Tasks**:
- [ ] Create `src/graph/neo4j_client.py` (120–150 LOC)
  - CRUD operations: `create_node()`, `create_relationship()`, `query()`
  - Connection pooling + transaction management
  - Error handling + retry logic
  - Health check/ping method (reuse existing `neo4j_config.py`)
- [ ] Unit tests for Neo4jClient

**Effort**: 3–4 days | **Dependencies**: `neo4j>=5.0` (add to pyproject.toml) | **Libraries**:
- `neo4j.GraphDatabase` (standard Neo4j driver)

**Success Criteria**:
- Neo4j connection established + health check passes
- CRUD operations work (create node, relationship, query)
- Tests cover happy path + error scenarios

---

#### 1.3 Define Graph Schema
**Tasks**:
- [ ] Create `src/graph/graph_schema.py` (80–120 LOC)
  - Define Node types: `CodeChunk`, `Symbol`, `File`, `Module`
  - Define Relationship types: `DEPENDS_ON`, `REFERENCES`, `DEFINES`, `IMPORTS`, `INHERITS_FROM`, `HAS_CHILD`, `IN_FILE`, `IN_MODULE`
  - Document properties for each node/relationship type

**Effort**: 2–3 days | **Dependencies**: None | **Example Schema**:

```python
# Node Types
NodeType.CODE_CHUNK = {
    "label": "CodeChunk",
    "properties": {
        "id": "string (uuid)",
        "type": "string (function|class|method|file|html_element|css_rule|js_function)",
        "name": "string",
        "qualified_name": "string",
        "file_path": "string",
        "language": "string",
        "start_line": "int",
        "end_line": "int",
        "complexity": "int",
        "docstring": "string",
    }
}

NodeType.SYMBOL = {
    "label": "Symbol",
    "properties": {
        "name": "string",
        "language": "string",
        "first_defined_in": "string (chunk_id)",
    }
}

# Relationship Types
RelationType.DEPENDS_ON = {
    "type": "DEPENDS_ON",
    "properties": {"imported_from": "string"},
}
RelationType.REFERENCES = {
    "type": "REFERENCES",
    "properties": {"reference_type": "string"},
}
```

**Success Criteria**:
- Schema document complete + reviewed
- All chunk relationship types mapped to Neo4j edges

---

#### 1.4 Implement GraphBuilder
**Tasks**:
- [ ] Create `src/graph/graph_builder.py` (150–200 LOC)
  - `build_nodes_from_chunks(chunks: list[CodeChunk]) → list[neo4j.Node]`
  - `build_relationships_from_chunks(chunks: list[CodeChunk]) → list[neo4j.Relationship]`
  - Batch insertion with deduplication (MERGE operations)
  - Error logging + statistics tracking

**Effort**: 4–5 days | **Dependencies**: `neo4j_client.py`, `graph_schema.py` | **Libraries**:
- `neo4j.graph` (graph object models)

**Example Implementation**:
```python
def build_nodes_from_chunks(chunks: list[CodeChunk]) -> list[dict]:
    """Convert CodeChunk objects to Neo4j node dicts"""
    nodes = []
    for chunk in chunks:
        node = {
            "id": chunk.id,
            "type": chunk.type,
            "name": chunk.name,
            "qualified_name": chunk.qualified_name,
            "file_path": chunk.file_path,
            "language": chunk.language,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "complexity": chunk.complexity,
        }
        nodes.append(node)
    return nodes

def build_relationships_from_chunks(chunks: list[CodeChunk]) -> list[dict]:
    """Extract relationships from chunk.relationships field"""
    rels = []
    for chunk in chunks:
        # Extract from dependencies field
        for dep in chunk.dependencies:
            rels.append({
                "source": chunk.id,
                "target": dep,
                "type": "DEPENDS_ON",
            })
        # Extract from relationships dict
        for called_func in chunk.relationships.get("called_functions", []):
            rels.append({
                "source": chunk.id,
                "target": called_func,
                "type": "CALLS",
            })
    return rels
```

**Success Criteria**:
- Nodes created correctly from chunks
- Relationships extracted from dependencies, references, defines
- Deduplication works (no duplicate nodes/edges)

---

**Phase 1 Summary**:
- **Output**: Graph infrastructure ready; Neo4j connectivity verified
- **Timeline**: 1–2 weeks
- **Risk**: Neo4j version compatibility; graph deduplication correctness

---

### **Phase 2: Graph Knowledge Layer** (2–3 weeks)
**Objective**: Fully populate and query the knowledge graph.

#### 2.1 Integrate GraphBuilder into Ingestion Pipeline
**Tasks**:
- [ ] Modify `main.py` CLI to include graph indexing step
  - New command: `python main.py index-graph --input embedded.json`
  - Call `GraphBuilder` after Qdrant indexing
  - Log statistics (nodes created, relationships created)
- [ ] Add transactional error handling (rollback on failure)
- [ ] Update `.env` example with Neo4j credentials

**Effort**: 2–3 days | **Dependencies**: `graph_builder.py`, `neo4j_client.py` | **Files**:
- `main.py` (add CLI command)
- `env.example` (add NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

**Success Criteria**:
- CLI command indexes chunks into Neo4j
- Error handling prevents partial indexing
- Statistics logged correctly

---

#### 2.2 Implement Graph Query Engine
**Tasks**:
- [ ] Create `src/graph/query_engine.py` (150–200 LOC)
  - `find_dependencies(chunk_id: str) → list[CodeChunk]` — All symbols this chunk depends on
  - `find_dependents(symbol: str) → list[CodeChunk]` — All chunks that depend on this symbol
  - `find_shortest_path(from_id: str, to_id: str) → list[str]` — Dependency chain
  - `find_neighbors(chunk_id: str, depth: int = 1) → list[CodeChunk]` — Connected nodes within N hops
  - `find_by_type_and_language(type: str, language: str) → list[CodeChunk]` — Filter queries

**Effort**: 4–5 days | **Dependencies**: `neo4j_client.py` | **Cypher Examples**:

```cypher
-- Find dependencies
MATCH (a:CodeChunk {id: $chunk_id})-[:DEPENDS_ON]->(b:CodeChunk)
RETURN b

-- Find dependents
MATCH (a:CodeChunk)-[:DEPENDS_ON]->(b:CodeChunk {id: $symbol_id})
RETURN a

-- Shortest path
MATCH path = shortestPath((a:CodeChunk {id: $from_id})-[*]->(b:CodeChunk {id: $to_id}))
RETURN [node in nodes(path) | node.id]

-- Neighbors (depth 1)
MATCH (a:CodeChunk {id: $chunk_id})-[*1..1]-(b:CodeChunk)
RETURN b
```

**Success Criteria**:
- All query methods return correct results
- Performance acceptable (< 200ms for typical queries)
- Tests cover happy path + edge cases (non-existent IDs, cycles)

---

#### 2.3 Implement Graph-Aware Retrieval
**Tasks**:
- [ ] Create `src/graph/graph_retriever.py` (100–150 LOC)
  - `retrieve(query: str, depth: int = 1) → list[CodeChunk]`
  - Workflow:
    1. Use vector search to find initial candidates
    2. Expand using graph traversal (up to `depth` hops)
    3. Deduplicate + rerank
    4. Return combined results
- [ ] Integrate with existing hybrid search (option to use graph expansion)

**Effort**: 3–4 days | **Dependencies**: `query_engine.py`, `HybridSearch` | **Integration**:
- Add `--graph-expansion-depth` flag to retrieval CLI

**Success Criteria**:
- Graph expansion retrieves contextually relevant chunks
- Combined results more comprehensive than vector-only
- Performance impact acceptable (< 500ms additional)

---

#### 2.4 Graph-Aware Reranking
**Tasks**:
- [ ] Create `src/graph/graph_aware_reranker.py` (80–120 LOC)
  - Score candidates using:
    - Cross-encoder score (existing)
    - Graph connectivity (number of neighbors, centrality)
    - Graph distance (hops from query seed)
  - Weighted combination: `score = 0.6 * cross_encoder + 0.2 * connectivity + 0.2 * proximity`
- [ ] Integrate with reranking pipeline

**Effort**: 3–4 days | **Dependencies**: `query_engine.py`, `cross_encoder.py` | **Example**:

```python
def rerank_with_graph(candidates, query, neo4j_client):
    """Combine cross-encoder scores with graph metrics"""
    scores = []
    for candidate in candidates:
        # Cross-encoder score
        ce_score = cross_encoder.score([(query, candidate["code"])])[0]

        # Graph metrics
        neighbors = len(neo4j_client.query_neighbors(candidate["id"]))
        connectivity = normalize(neighbors)

        # Combined score
        final_score = 0.6 * ce_score + 0.2 * connectivity + 0.2 * (1 / (distance + 1))
        scores.append((candidate, final_score))

    return sorted(scores, key=lambda x: x[1], reverse=True)
```

**Success Criteria**:
- Graph-aware scores improve retrieval quality
- No performance regression
- Weights tunable via configuration

---

**Phase 2 Summary**:
- **Output**: Full graph RAG capability; graph-enhanced retrieval + reranking
- **Timeline**: 2–3 weeks
- **Risk**: Cypher query performance at scale; graph data consistency

---

### **Phase 3: Agentic RAG Infrastructure** (3–4 weeks)
**Objective**: Build agent executor, tools, and planning capability.

#### 3.1 Implement Tool Registry
**Tasks**:
- [ ] Create `src/agents/tools/tool_registry.py` (60–80 LOC)
  - `register_tool(name: str, tool: Callable, description: str, parameters: dict)`
  - `get_tool(name: str) → Callable`
  - `list_tools() → list[dict]`
  - Validation + error handling

**Effort**: 1–2 days | **Dependencies**: None | **Example**:

```python
class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register_tool(self, name: str, tool: Callable, description: str):
        self.tools[name] = {
            "callable": tool,
            "description": description,
        }

    def get_tool(self, name: str) -> Callable:
        return self.tools[name]["callable"]

    def list_tools(self) -> list[dict]:
        return [{"name": k, "description": v["description"]} for k, v in self.tools.items()]
```

**Success Criteria**:
- Tools can be registered + retrieved
- Tool descriptions clear for LLM prompts

---

#### 3.2 Implement Core Tools
**Tasks**:
- [ ] Create `src/agents/tools/code_search_tool.py` (60–80 LOC)
  - `search_code(query: str, top_k: int = 5) → list[CodeChunk]`
  - Uses existing HybridSearch + reranking
  - Tool description: "Search code by semantic similarity"

- [ ] Create `src/agents/tools/graph_query_tool.py` (100–150 LOC)
  - `find_dependencies(chunk_id: str) → list[CodeChunk]`
  - `find_dependents(symbol: str) → list[CodeChunk]`
  - `trace_path(from_id: str, to_id: str) → list[str]`
  - Tool description: "Query code relationships and dependencies"

- [ ] Create `src/agents/tools/dependency_analyzer_tool.py` (80–100 LOC)
  - `analyze_impact(chunk_id: str) → dict` — What breaks if this changes?
  - `find_circular_dependencies(start_id: str) → list[list[str]]` — Find cycles
  - Tool description: "Analyze code dependencies and impact"

- [ ] Create `src/agents/tools/test_locator_tool.py` (60–80 LOC)
  - `find_related_tests(chunk_id: str) → list[CodeChunk]` — Tests for this code
  - Tool description: "Find tests related to code"

**Effort**: 5–7 days | **Dependencies**: `query_engine.py`, `HybridSearch` | **Integration**:
- Register all tools in `tool_registry`

**Success Criteria**:
- Each tool returns correct results
- Tool descriptions clear for LLM
- Error handling for invalid inputs

---

#### 3.3 Implement Conversation Memory
**Tasks**:
- [ ] Create `src/agents/memory/conversation_memory.py` (80–100 LOC)
  - In-memory storage of conversation history
  - Context window management (keep last N messages)
  - `add_message(role: str, content: str)`
  - `get_context() → str` — Formatted history for LLM

**Effort**: 2–3 days | **Dependencies**: None | **Example**:

```python
class ConversationMemory:
    def __init__(self, max_messages: int = 20):
        self.messages = []
        self.max_messages = max_messages

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)

    def get_context(self) -> str:
        return "\n".join([f"{m['role']}: {m['content']}" for m in self.messages])
```

**Success Criteria**:
- Message history maintained correctly
- Context formatting works for LLM prompts

---

#### 3.4 Implement Agent Executor
**Tasks**:
- [ ] Create `src/agents/agent_executor.py` (200–250 LOC)
  - Main agent loop:
    1. Receive user question
    2. LLM decides which tool(s) to call + parameters
    3. Execute tool(s)
    4. Aggregate results
    5. Generate response
  - Implement LangChain agent pattern (e.g., ReAct)
  - Error handling + fallbacks
  - Iteration limit (max 10 steps)

**Effort**: 6–8 days | **Dependencies**: `tool_registry`, `ConversationMemory`, LLM client | **Libraries**:
- `langchain.agents` (agent framework)
- `langchain_core.tools` (tool wrappers)

**Example Agent Flow**:
```
User: "How does the ProfileAdmin class affect the Profile model?"

Agent Step 1:
  → Tool call: graph_query_tool.find_dependents("ProfileAdmin")
  → Returns: [chunk_profile_admin]

Agent Step 2:
  → Tool call: code_search_tool.search_code("Profile model integration")
  → Returns: [chunk_profile_model]

Agent Step 3:
  → Tool call: dependency_analyzer_tool.analyze_impact("ProfileAdmin")
  → Returns: {impacted: [...], circular_deps: [...]}

Agent Reasoning:
  → Synthesize results into coherent response
  → Return to user with explanation
```

**Success Criteria**:
- Agent selects correct tools for query
- Iteration limit prevents infinite loops
- Error handling prevents agent crashes

---

**Phase 3 Summary**:
- **Output**: Functional agent with 4+ tools; multi-step reasoning
- **Timeline**: 3–4 weeks
- **Risk**: LLM model selection (accuracy + speed); tool parameter specification

---

### **Phase 4: Advanced Agent Capabilities** (2–3 weeks)
**Objective**: Add planning, reasoning, and multi-turn interaction.

#### 4.1 Implement Planning Agent
**Tasks**:
- [ ] Create `src/agents/planning/planner_agent.py` (150–200 LOC)
  - Decompose complex queries into sub-tasks
  - Plan optimal tool sequence
  - Example: "Find all code affected by removing function X"
    → Decompose: 1) Find function, 2) Find dependents, 3) Analyze impact, 4) Suggest tests

**Effort**: 4–5 days | **Dependencies**: `agent_executor.py` | **Libraries**:
- `langchain.agents` (chain-of-thought)

**Success Criteria**:
- Complex queries decomposed into steps
- Sub-task results correctly aggregated

---

#### 4.2 Implement Specialized Agents
**Tasks**:
- [ ] **Code Debugging Agent**: Find issues in code
  - Tools: search, graph query, test locator, documentation
  - Workflow: Locate problem code → trace dependencies → suggest fix

- [ ] **Refactoring Agent**: Suggest code improvements
  - Tools: search, dependency analyzer, graph query
  - Workflow: Identify candidates → analyze impact → suggest changes

- [ ] **Documentation Agent**: Generate code documentation
  - Tools: search, code search, graph query
  - Workflow: Extract code structure → find related code → generate docs

**Effort**: 5–7 days | **Dependencies**: `agent_executor.py`, all tools | **Integration**:
- Add CLI commands: `python main.py debug --query "..."`, `python main.py refactor --query "..."`, etc.

**Success Criteria**:
- Each agent handles its domain correctly
- Results useful for developers

---

#### 4.3 Add Retrieval-Augmented Generation (RAG) Improvements
**Tasks**:
- [ ] Implement context optimization: "What is the best context for this query?"
- [ ] Add relevance scoring: Filter irrelevant results before passing to LLM
- [ ] Implement adaptive context window: Adjust based on query complexity
- [ ] Add source attribution: Track where each piece of information came from

**Effort**: 3–4 days | **Dependencies**: All layers | **Example**:

```python
def optimize_context(query: str, candidates: list[CodeChunk], llm_client) -> str:
    """Select most relevant candidates for LLM context"""
    scores = []
    for candidate in candidates:
        # Relevance to query
        relevance = semantic_similarity(query, candidate.code)
        # Importance (connectivity, complexity, documentation)
        importance = compute_importance(candidate)
        # Combined
        score = 0.7 * relevance + 0.3 * importance
        scores.append((candidate, score))

    # Select top candidates until context window limit
    selected = []
    total_tokens = 0
    for candidate, score in sorted(scores, key=lambda x: x[1], reverse=True):
        tokens = count_tokens(candidate.code)
        if total_tokens + tokens <= MAX_CONTEXT_TOKENS:
            selected.append(candidate)
            total_tokens += tokens
        else:
            break

    return format_context(selected)
```

**Success Criteria**:
- Context quality improved (fewer irrelevant results)
- LLM response quality improved
- No performance regression

---

**Phase 4 Summary**:
- **Output**: Advanced agentic workflows; specialized agents for debugging/refactoring/documentation
- **Timeline**: 2–3 weeks
- **Risk**: LLM hallucinations; tool parameter complexity

---

### **Phase 5: Integration & Optimization** (1–2 weeks)
**Objective**: Unify all layers; optimize performance; enable production deployment.

#### 5.1 Create GAV Query Router
**Tasks**:
- [ ] Create `src/retrieval/query_router.py` (80–120 LOC)
  - Route queries to optimal strategy:
    - Simple queries → Vector search only (fast)
    - Complex queries → Agent + all tools (accurate)
    - Graph-heavy queries → Graph retrieval + vector reranking
  - Heuristics: Query length, keyword detection ("how", "why", "trace", "impact")

**Effort**: 2–3 days | **Dependencies**: All retrieval layers | **Example**:

```python
def route_query(query: str) -> str:
    """Determine best retrieval strategy"""
    tokens = len(query.split())
    keywords = ["trace", "impact", "affect", "dependency", "related", "refactor"]

    if tokens <= 5 and not any(kw in query.lower() for kw in keywords):
        return "vector"  # Fast path
    elif "trace" in query.lower() or "dependency" in query.lower():
        return "graph"  # Graph-heavy
    else:
        return "agent"  # Complex reasoning
```

**Success Criteria**:
- Routing decisions reasonable
- Performance improvement for simple queries

---

#### 5.2 API Endpoint Enhancement
**Tasks**:
- [ ] Add new endpoints to FastAPI server:
  - `POST /query/vector` — Vector search only
  - `POST /query/graph` — Graph-based retrieval
  - `POST /query/agent` — Agentic reasoning
  - `POST /query/hybrid` — Combined vector + graph
  - `GET /graph/nodes` — Graph exploration
  - `GET /graph/relationships` — Relationship queries

**Effort**: 2–3 days | **Dependencies**: All layers, FastAPI | **Files**:
- `main.py` (add routes)

**Success Criteria**:
- Endpoints functional + documented (Swagger)
- Response times acceptable
- Error handling comprehensive

---

#### 5.3 Performance Optimization
**Tasks**:
- [ ] Profile critical paths (retrieval, agent execution)
- [ ] Add caching for:
  - Query results (Redis or in-memory)
  - Graph neighbor lookups
  - Embedding computations
- [ ] Implement batch processing for multi-query scenarios
- [ ] Add observability: Logging, metrics, tracing

**Effort**: 3–5 days | **Dependencies**: All layers | **Libraries**:
- `redis` (optional, for distributed caching)
- `opentelemetry` (optional, for tracing)

**Success Criteria**:
- Agent queries < 2 seconds (P99)
- Graph queries < 500ms
- Caching reduces repeat query latency by 80%+

---

#### 5.4 Testing & Quality Assurance
**Tasks**:
- [ ] Expand test suite:
  - Unit tests for all new modules (graph, agents)
  - Integration tests for end-to-end workflows
  - Performance benchmarks
  - Regression tests (ensure vector/reranking still work)
- [ ] Add test data: Representative queries + expected results
- [ ] Implement continuous benchmarking

**Effort**: 3–5 days | **Dependencies**: All layers | **Files**:
- `tests/test_graph_*.py`
- `tests/test_agent_*.py`
- `tests/test_integration_*.py`

**Success Criteria**:
- > 80% test coverage for new code
- No performance regression
- Documented test cases

---

#### 5.5 Documentation & Deployment
**Tasks**:
- [ ] Update `QWEN.md` with GAV architecture overview
- [ ] Create `docs/graph_rag_guide.md` — Graph layer usage
- [ ] Create `docs/agentic_rag_guide.md` — Agent development
- [ ] Docker Compose update: Add Neo4j service
- [ ] Deployment guide: How to run full GAV system
- [ ] API documentation: Swagger + examples

**Effort**: 2–3 days | **Files**:
- `QWEN.md` (update)
- `docs/` (new guides)
- `docker-compose.yml` (update)

**Success Criteria**:
- Documentation clear + complete
- New developers can set up system in < 1 hour
- Deployment guide tested

---

**Phase 5 Summary**:
- **Output**: Unified GAV system; production-ready; well-tested + documented
- **Timeline**: 1–2 weeks
- **Risk**: Integration bottlenecks; performance under load

---

## 5. Implementation Timeline Summary

| Phase | Duration | Key Deliverables | Status |
|-------|----------|------------------|--------|
| **1 - Vector + Graph Setup** | 1–2 weeks | Neo4jClient, GraphBuilder, graph schema | 🎯 Start here |
| **2 - Graph Knowledge** | 2–3 weeks | Query engine, graph retrieval, graph reranking | Building on 1 |
| **3 - Agentic Infrastructure** | 3–4 weeks | Tool registry, core tools, agent executor | Depends on 2 |
| **4 - Advanced Agents** | 2–3 weeks | Planning agent, specialized agents, RAG improvements | Building on 3 |
| **5 - Integration & Optimization** | 1–2 weeks | Query router, endpoints, performance, testing | Final integration |

**Total**: 12–18 weeks (3–4 months) for **complete GAV RAG system**

**Quick Wins** (start immediately):
- ✅ Week 1: Neo4jClient + GraphBuilder (Phases 1.2–1.3)
- ✅ Week 2–3: GraphBuilder integration + query engine (Phases 1.4, 2.2)
- ✅ Week 4–5: Core tools + agent executor (Phase 3.1–3.4)

---

## 6. Code Structure Changes

### 6.1 New Modules (Don't Edit Existing)

```
src/
├── graph/                         # NEW
│   ├── __init__.py
│   ├── graph_schema.py            # Node/relationship definitions
│   ├── graph_builder.py           # CodeChunk → Neo4j converter
│   ├── neo4j_client.py            # Neo4j CRUD wrapper
│   ├── query_engine.py            # Graph traversal queries
│   ├── graph_retriever.py         # Graph-based retrieval
│   └── graph_aware_reranker.py    # Graph-aware scoring
│
├── agents/                        # NEW
│   ├── __init__.py
│   ├── agent_executor.py          # Main agent loop
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── tool_registry.py       # Tool management
│   │   ├── code_search_tool.py
│   │   ├── graph_query_tool.py
│   │   ├── dependency_analyzer_tool.py
│   │   ├── test_locator_tool.py
│   │   └── documentation_tool.py
│   ├── memory/
│   │   ├── __init__.py
│   │   └── conversation_memory.py # Conversation history
│   └── planning/
│       ├── __init__.py
│       └── planner_agent.py       # Multi-step planning
│
├── retrieval/
│   ├── query_router.py            # NEW: Route queries to strategies
│   └── [existing files unchanged]
│
└── [all other existing modules unchanged]

tests/
├── test_graph_schema.py           # NEW
├── test_graph_builder.py          # NEW
├── test_neo4j_client.py           # NEW
├── test_query_engine.py           # NEW
├── test_agent_executor.py         # NEW
├── test_tools_*.py                # NEW: Tool tests
├── test_integration_*.py          # NEW: End-to-end tests
└── [existing tests unchanged]

docs/
├── architecture_claude_gav_rag.md # THIS FILE (migration plan)
├── claudearchitecture.mermaid     # NEW: GAV architecture diagram
├── graph_rag_guide.md             # NEW: Graph layer usage
├── agentic_rag_guide.md           # NEW: Agent development
└── [existing docs unchanged]
```

### 6.2 Modified (Extended) Modules

```
src/config/
├── neo4j_config.py               # EXTEND: Add graph-specific settings
├── settings.py                   # EXTEND: Add retrieval_mode, graph_enabled
└── [embedding_config, qdrant_config unchanged]

src/retrieval/
├── hybrid_search.py              # EXTEND: Add graph expansion option
├── search.py                     # EXTEND: Add graph-aware filtering
└── rag_system.py                 # EXTEND: Add query router integration

main.py                            # EXTEND: Add new CLI commands (index-graph, query-graph, debug, etc.)
docker-compose.yml               # EXTEND: Add Neo4j service
env.example                       # EXTEND: Add NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
QWEN.md                           # EXTEND: Update architecture overview
```

### 6.3 Unchanged (Keep As-Is)

```
src/preprocessing/    # No changes needed; data already includes graph structure
src/embedding/        # No changes needed; quality validation sufficient
src/generation/       # No changes for Phase 1-3; optional enhancements in Phase 4
src/reranking/        # Enhanced, not replaced
```

---

## 7. Dependencies & Libraries

### 7.1 New Dependencies to Add (pyproject.toml)

```toml
[dependencies]
# ... existing ...

# Graph RAG
neo4j = ">=5.0"                  # Neo4j driver
networkx = ">=3.0"               # Optional: local graph analysis

# Agentic RAG
langchain-core = ">=0.3.0"       # Already present; extend usage
langchain-graph = ">=0.1.0"      # Optional: graph utilities

# Performance & Observability
redis = ">=5.0"                  # Optional: caching
opentelemetry-api = ">=1.0"      # Optional: tracing
opentelemetry-sdk = ">=1.0"      # Optional: tracing

# Browser (Optional for rendering layer)
playwright = ">=1.40.0"          # Optional: frontend snapshots
```

### 7.2 Environment Variables (.env)

```bash
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# Graph Configuration
GRAPH_ENABLED=true
GRAPH_BATCH_SIZE=100
GRAPH_DEDUP_ENABLED=true

# Retrieval Configuration
RETRIEVAL_MODE=hybrid           # vector | graph | hybrid | agent
VECTOR_WEIGHT=0.7
GRAPH_WEIGHT=0.3

# Agent Configuration
AGENT_MAX_ITERATIONS=10
AGENT_LLM_MODEL=ai/llama3.2:latest
AGENT_TEMPERATURE=0.3

# Performance
CACHE_ENABLED=true
CACHE_TTL=3600                  # seconds
```

---

## 8. Success Criteria & Validation

### 8.1 Phase-wise Validation

#### **Phase 1 Success**:
- ✅ Neo4jClient connection established + health check passes
- ✅ GraphBuilder creates nodes/relationships from CodeChunks
- ✅ Graph schema documented + validated
- ✅ All nodes + relationships stored correctly

#### **Phase 2 Success**:
- ✅ Graph queries return correct results (dependencies, dependents, paths)
- ✅ Graph retrieval enhances vector results (more contextual)
- ✅ Graph-aware reranking improves ranking quality
- ✅ Graph expansion depth configurable + performant

#### **Phase 3 Success**:
- ✅ Tool registry manages 5+ tools
- ✅ Core tools return correct results
- ✅ Agent executor selects correct tools for queries
- ✅ Multi-step reasoning produces coherent results

#### **Phase 4 Success**:
- ✅ Planning agent decomposes complex queries
- ✅ Specialized agents (debug, refactor, docs) functional
- ✅ RAG improvements increase response quality
- ✅ No performance regression

#### **Phase 5 Success**:
- ✅ Query router makes intelligent routing decisions
- ✅ New API endpoints functional + documented
- ✅ Performance benchmarks met (agent < 2s, graph < 500ms)
- ✅ Test coverage > 80%
- ✅ Documentation complete + accessible

### 8.2 Quality Metrics

| Metric | Target | Phase |
|--------|--------|-------|
| Graph relationship accuracy | > 95% | 2 |
| Vector-graph retrieval precision | > 0.85 | 2 |
| Agent tool selection accuracy | > 90% | 3 |
| Average agent response time | < 2 sec | 4 |
| Graph query latency (P99) | < 500 ms | 2 |
| Test coverage (new code) | > 80% | 5 |
| Documentation completeness | 100% | 5 |

---

## 9. Risk Assessment & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| **Graph data inconsistency** (Qdrant ↔ Neo4j out of sync) | Medium | High | Implement transactional indexing; health checks; periodic reconciliation |
| **Neo4j performance at scale** (1M+ nodes/edges) | Medium | Medium | Early benchmarking; query optimization; indexing strategy |
| **LLM hallucinations in agent** | High | Medium | Tool validation; confidence scores; human-in-the-loop option |
| **Tool parameter complexity** (agent struggles to call tools) | Medium | Medium | Few-shot examples in prompt; tool description clarity; validation |
| **Circular dependencies in graph** | Low | Low | Cycle detection; warning logs; optional cycle handling |
| **Performance regression** (new layers slow down retrieval) | Medium | Medium | Caching; query optimization; early benchmarking |
| **Neo4j version incompatibility** | Low | High | Pin driver version; test compatibility; document requirements |
| **Memory usage** (agent holds conversation history) | Low | Low | Periodic cleanup; configurable history limit |

---

## 10. Quick Start Checklist

### Immediate Next Steps

- [ ] **Week 1**:
  - [ ] Add `neo4j` to `pyproject.toml`
  - [ ] Create `src/graph/` directory
  - [ ] Implement `neo4j_client.py` (CRUD wrapper)
  - [ ] Implement `graph_schema.py` (node/relationship types)
  - [ ] Write unit tests

- [ ] **Week 2**:
  - [ ] Implement `graph_builder.py` (CodeChunk → Neo4j)
  - [ ] Create CLI command: `index-graph`
  - [ ] Test on sample codebase (e.g., geminIndex itself)
  - [ ] Verify data consistency

- [ ] **Week 3**:
  - [ ] Implement `query_engine.py` (graph traversal)
  - [ ] Implement `graph_retriever.py` (enhanced retrieval)
  - [ ] Benchmark performance
  - [ ] Document APIs

- [ ] **Week 4+**:
  - [ ] Begin Phase 3 (agent infrastructure)
  - [ ] Implement tool registry + core tools
  - [ ] Build agent executor

---

## 11. References to Source Documents

### Expansion Steps
- **Phase 1 — Infrastructure & Configuration Setup**: [expansion_steps.md - Phase 1](../docs/expansion_steps.md#phase-1---infrastructure--configuration-setup)
- **Phase 2 — Knowledge Graph Implementation**: [expansion_steps.md - Phase 2](../docs/expansion_steps.md#phase-2---knowledge-graph-implementation)
- **Phase 3 — Rendering & CDP Snapshot Pipeline**: [expansion_steps.md - Phase 3](../docs/expansion_steps.md#phase-3---rendering--cdp-snapshot-pipeline)

### Full RAG System Guide
- **Directory Structure**: [full_rag_system_implementation_guide.md - Section 1](../docs/full_rag_system_implementation_guide.md#-1-directory-structure-recommended-foundation)
- **Phase 1 — Foundations**: [full_rag_system_implementation_guide.md - Section 2](../docs/full_rag_system_implementation_guide.md#-2-phase-1--foundations--configuration)
- **Phase 4 — Knowledge Graph**: [full_rag_system_implementation_guide.md - Section 5](../docs/full_rag_system_implementation_guide.md#-5-phase-4--knowledge-graph-neo4j)
- **Phase 7 — Agentic RAG**: [full_rag_system_implementation_guide.md - Section 8](../docs/full_rag_system_implementation_guide.md#-8-phase-7--agentic-rag)

### Graph + Agentic Pipeline
- **Graph RAG from Chunk Structure**: [rag_graph_agent_pipeline.md - Section 2–4](../docs/rag_graph_agent_pipeline.md#2-build-your-knowledge-graph-from-chunks)
- **Agentic Workflow Example**: [rag_graph_agent_pipeline.md - Section 6](../docs/rag_graph_agent_pipeline.md#6-agentic-workflow-example)

---

## 12. Conclusion

The geminIndex project has a **solid foundation** for GAV expansion. With its rich CodeChunk metadata, sophisticated preprocessing, and modular architecture, adding graph and agentic layers is achievable in **3–4 months**. The phased approach minimizes risk and allows for incremental validation.

**Key Strengths**:
- ✅ CodeChunk structure already contains graph relationships
- ✅ Neo4j infrastructure configured but unused
- ✅ LangChain integration ready for agents
- ✅ Well-tested preprocessing + retrieval layers

**Next Step**: Start **Phase 1** by implementing `neo4j_client.py` and `graph_builder.py`. Test on the geminIndex codebase itself to validate the approach.

---

**Document prepared**: December 2, 2025
**Prepared by**: Claude (AI Architecture Assistant)
**Status**: Ready for implementation
