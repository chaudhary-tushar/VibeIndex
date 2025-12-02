# 📘 Architecture Migration Plan: GAV RAG System for geminIndex

**Document Version**: 1.0
**Date**: December 2, 2025
**Prepared for**: geminIndex Expansion
**Target Architecture**: Graph-Agentic-Vector (GAV) RAG System

---

## Executive Summary

This document provides a comprehensive migration path to expand the existing geminIndex RAG system into a full **Graph-Agentic-Vector (GAV)** architecture. The current system (as outlined in `rag2.mermaid`) features sophisticated preprocessing, embedding, and vector retrieval capabilities. This plan builds upon that foundation by adding:

1. **Knowledge Graph Layer** — Neo4j-based graph database for structural code relationships
2. **Agentic Orchestration Layer** — Multi-step reasoning with tool-use capabilities
3. **Hybrid Retrieval System** — Unified vector + graph + semantic search
4. **Interactive Query Router** — Agent-driven query planning and tool selection

The migration is designed to be **incremental and non-disruptive**, preserving all existing functionality while adding new capabilities.

---

## Part I: Current State Assessment

### A. Existing Architecture Overview

The current geminIndex system follows the pipeline defined in `rag2.mermaid`:
Input Files → Preprocessing (Parser, Analyzer, Mapper)
→ Embedding (Quality Validator)
→ Vector DB (Qdrant, 3 collections: functions/classes/modules)
→ Hybrid Retrieval (Dense 0.7 + Sparse 0.3 BM25 fusion)
→ Reranking (Cross-Encoder, MS-MARCO MiniLM L-6-v2)
→ Generation (LLM context building & prompting)


**Key Components**:

| Layer | Component | Status | Technology |
|-------|-----------|--------|-----------|
| **Preprocessing** | Parser, Analyzer, MetadataExtractor, DependencyMapper, ChunkPreprocessor | ✅ Complete | Tree-sitter, libCST, pathspec |
| **Embedding** | EmbeddingGenerator, QualityValidator, BatchProcessor | ✅ Complete | Ollama/OpenAI, Transformers |
| **Vector Store** | QdrantIndexer (3 collections, sparse+dense vectors) | ✅ Complete | Qdrant 1.15.1+, cosine distance |
| **Retrieval** | HybridSearchEngine, BM25 tokenizer | ✅ Complete | Qdrant fusion queries |
| **Reranking** | CrossEncoderReranker | ✅ Complete | sentence-transformers |
| **Generation** | LLMClient, ContextEnricher, PromptConstructor | ⚠️ Partial | LangChain 1.0.8+, Ollama/OpenAI |
| **Graph Layer** | Neo4jConfig (infrastructure only) | ❌ **Missing** | Neo4j 5.x driver (not installed) |
| **Agent Layer** | LangChain hooks (not utilized) | ❌ **Missing** | LangChain 1.0.8+, no tools |
| **Orchestration** | CLI (main.py, cli.py) | ⚠️ Limited | Click 8.0.0+, basic routing |

### B. Data Structures & Integration Points

#### CodeChunk - The Foundation

**Location**: `src/preprocessing/chunk.py` (90 lines)

All chunks contain graph-relevant metadata:

```python
@dataclass
class CodeChunk:
    # Core ID
    id: str                              # UUID, used as graph node ID

    # Type & location
    type: str                            # 'function', 'class', 'file', 'method', etc.
    name: str
    file_path: str
    language: str
    start_line: int
    end_line: int

    # GRAPH-RELEVANT FIELDS ✅ Already populated
    dependencies: list[str]              # Imports, external deps → graph edges
    references: list[str]                # Symbols used → graph edges
    defines: list[str]                   # Symbols defined → graph nodes

    # Rich metadata
    metadata: dict[str, Any]             # Decorators, access modifiers, tags
    relationships: dict[str, Any]        # Imports, children, parents, inheritance
    context: dict[str, Any]              # Module hierarchy, domain

Why This Is Ideal for GAV:

dependencies, references, defines are already extracted during preprocessing
relationships.* contains structured import/inheritance/containment info
context enables domain-aware queries
Chunks are idempotent (SHA256 hash-based, deduped)


Qdrant Payload Structure
Each chunk is stored in Qdrant with:
{
  "id": "uuid",
  "type": "function",
  "qualified_name": "module.Class.method",
  "code": "...",
  "dependencies": ["module_A", "Symbol_B"],
  "references": ["Symbol_C"],
  "relationships": {
    "imports": [...],
    "children": [...],
    "class_inheritance": [...]
  },
  "context": {...},
  "embedding": [float, ...],    // 768-dim
  "embedding_quality": "validated"
}
C. Preprocessing Pipeline (Layers 1-2)
Architecture: 5-stage pipeline with language-specific extraction.

Stage	Component	Lines	Input	Output	Language Support
Parse	parser.py::get_file_paths() → language detection	322	Files, .gitignore	CodeChunk[] (basic)	Python, JS, HTML, CSS, Markdown
Analyze	analyzer.py → AST extraction (1,767 LOC)	1,767	CodeChunk[]	Enhanced metadata + dependencies	Python (libCST), JS, HTML, CSS
Map Dependencies	dependency_mapper.py	234	Enhanced chunks	Import/reference/symbol edges	Python, JS, HTML, CSS
Extract Metadata	metadata_extractor.py	114	AST nodes	Docstrings, signatures, complexity	Multi-language
Preprocess	preprocessor.py → dedup + enrich	100	All enhanced chunks	embedding_text field + validation	All
Key Insight: By Phase 2a (Preprocessing Complete), all chunks have dependencies, references, defines, and relationships fully populated. This is the perfect input for graph construction.

D. Current Integration Points
CLI Entry Points (main.py, cli.py):

ingest → parse project
preprocess → dedup & enrich
embed → generate embeddings
index → insert into Qdrant
rag → query using CodeRAG_2 (vector-only)
advanced_rag → query with reranking
FastAPI Server (main.py):

POST /parse-project → ingest
GET /parse-file → parse single file
Swagger at docs
Neo4j Configuration (Exists but Unused):

neo4j_config.py → Neo4jConfig class with .ping() method
Connection infrastructure in place, no driver operations
E. Identified Gaps & Opportunities
Gap 1: Graph Layer (Critical)
Current: No Neo4j integration beyond configuration
Needed:
Neo4jClient → CRUD operations (create/read/update/delete nodes/edges)
GraphBuilder → Chunk JSON → Neo4j nodes/relationships
GraphSchema → Node labels, relationship types, constraints
Graph-aware retrieval queries (Cypher)
Effort: 2-3 weeks (HIGH)
Opportunity: Enables dependency tracing, inheritance traversal, impact analysis
Gap 2: Agentic Orchestration (Critical)
Current: LangChain is installed but agents are not implemented
Needed:
AgentExecutor → LLM-driven loop with tool calls
ToolRegistry → Centralized tool management
Agent tools: CodeSearchTool, GraphQueryTool, DependencyAnalyzer, TestLocator
AgentMemory → Conversation history and reasoning traces
Effort: 3-4 weeks (VERY HIGH)
Opportunity: Multi-step code understanding, context-aware recommendations
Gap 3: Hybrid Retrieval Coordination
Current: Vector retrieval only, no graph retrieval alternative
Needed:
HybridRetrievalOrchestrator → Route to vector, graph, or both
ResultsCombiner → Merge and dedupe vector + graph results
Graph-aware reranking scorer
Effort: 1-2 weeks (MEDIUM)
Dependency: Requires Graph Layer
Gap 4: Visual Rendering (Optional)
Current: No rendering infrastructure
Needed:
GraphVisualizer → Interactive Cytoscape/D3.js frontend
DependencyGraphRenderer → ASCII/HTML rendering for CLI/web
Effort: 2-3 weeks (HIGH, depends on UI choice)
Priority: Nice-to-have, not essential for GAV core
Gap 5: Agentic Debugging (Optional)
Current: No frontend debugging pipeline
Needed:
Playwright snapshotter (frontend rendering)
DOM analysis tools
CSS-HTML matching graph
Effort: 2-3 weeks (HIGH)
Priority: Not in core GAV scope, future enhancement
F. Test Coverage & Quality
Existing Tests (Comprehensive):

test_parser.py → Parsing logic
test_preprocessing.py → Chunk enrichment
test_dependency_mapper.py → Import tracking
test_metadata_extractor.py → Metadata extraction
test_config.py → Configuration validation
test_embedding_ext.py → Embedding generation
test_retrieval_ext.py → Vector search
test_reranking.py → Reranking logic
test_generation_ext.py → LLM integration
Gaps for GAV:

No tests for Neo4j operations
No tests for agent execution
No tests for hybrid retrieval
No integration tests for full GAV pipeline
Part II: Gaps & Opportunities Analysis
A. Graph RAG Opportunity
Gap: No knowledge graph exists; chunks are only indexed as vectors.

Opportunity:

Extract structural knowledge from chunk metadata (dependencies, references, relationships)
Enable topological queries: "What calls this function?" "What imports this module?" "Trace dependency chain"
Support symbol resolution: "Where is this class defined?" "Who inherits from this?"
Enable impact analysis: "Changing this function affects..." (reverse dependency graph)
Improve retrieval quality by combining semantic (vector) + structural (graph) signals
Technical Approach:

Graph Schema: Define node types (CodeChunk, Symbol, File, Module) and edge types (DEPENDS_ON, REFERENCES, INHERITS_FROM, IMPORTS, CALLS, CONTAINS)
Data Flow: Preprocessing → chunks → Neo4j graph (parallel to Qdrant)
Query Engine: Cypher-based traversal with LLM translation
Integration: Hybrid retriever chooses vector, graph, or both based on query intent
B. Agentic RAG Opportunity
Gap: Query handling is linear (parse query → embed → retrieve → rerank → generate). No multi-step reasoning or tool use.

Opportunity:

Enable multi-step problem solving: Agent reasons "I need to find the model, then find tests, then analyze impact"
Provide tool ecosystem: Search, graph traversal, test location, documentation lookup
Support self-correction: Agent can evaluate results and retry with different tools
Enable conversation context: Remember previous queries and build on them
Improve answer quality: Agent can ask clarifying questions or provide multiple perspectives
Technical Approach:

Agent Framework: LangChain AgentExecutor with ReAct (Reasoning + Acting) pattern
Tool Registry: 5-10 tools covering search, graph, testing, documentation
Memory: Conversation history + reasoning traces
Orchestration: Route incoming queries to appropriate tool combinations
C. Hybrid Retrieval Opportunity
Gap: Current system uses vector retrieval. Graph retrieval is not available.

Opportunity:

Query "What calls this function?" → Graph retrieval (graph-native query)
Query "How to implement user authentication?" → Vector retrieval (semantic similarity)
Query "Find tests for the UserModel and its dependencies" → Hybrid (both layers)
Enable query intent detection: Agent recognizes query type and selects retrieval strategy
Technical Approach:


Query Intent Classifier: Use LLM to detect "structural" vs "semantic" vs "hybrid" queries
Dual Retrieval: Call both vector and graph retrievers in parallel when hybrid
Result Fusion: Merge results, dedupe, rerank with unified scoring
D. LLM Enhancement Opportunity
Current: Context building is basic (collect top-K chunks, format prompt).

Opportunity:

AI-generated summaries: Pre-compute summaries of key chunks for faster context building
Symbol index: Quick lookup of symbols (class definitions, function signatures)
Semantic compression: Use LLM to condense context while preserving key information
Multi-perspective generation: Generate answers from different angles (code view, documentation view, dependency view)
Technical Approach:

Summary Cache: SQLite cache of LLM-generated summaries (already partially implemented in context_builder.py)
Symbol Index: Pre-built mapping of symbol → definition chunk
Compression Pipeline: LLM-based abstractive summarization during context building
Prompt Variants: Different prompt templates for different query types
Part III: Phased Migration Roadmap
Phase 1: Vector Foundations & Configuration (1-2 weeks)
Goal: Strengthen the embedding and retrieval layers, add enhanced configuration for GAV expansion.

1.1: Enhance Embedding Configuration
Task: Extend embedding_config.py with embedding strategy options.

New Config Fields:
class EmbeddingConfig:
    # Existing
    model_url: str
    model_name: str
    embedding_dim: int

    # NEW for GAV
    embedding_strategy: Literal["dense", "sparse", "hybrid"] = "dense"
    include_graph_aware_embedding: bool = False  # Pre-compute chunk importance via graph
    cache_embeddings: bool = True
    chunk_text_enrichment: Literal["none", "basic", "full"] = "full"

New File: None (extend existing)

Effort: 2-3 days

Success Criteria:

Configuration validates and loads from .env
Tests pass for new config fields
Documentation updated in QWEN.md
Dependencies: None (config only)

1.2: Enhance Qdrant Configuration for Multi-Indexing
Task: Extend qdrant_config.py to support graph-aware indexing options.

New Fields:
class QdrantConfig:
    # Existing
    host: str
    port: int
    collection_prefix: str

    # NEW for GAV
    enable_graph_sync: bool = False
    graph_collection_suffix: str = "_graph"
    graph_metadata_fields: list[str] = ["dependencies", "references", "relationships"]

New File: None (extend existing)

Effort: 2-3 days

Success Criteria:

Validates graph-specific settings
Backwards-compatible with existing configs
Dependencies: None

1.3: Create Enhanced Embedding Text for GAV
Task: Modify preprocessor.py to enrich embedding_text with graph-aware context.

Changes:

Include relationships and context info in embedding text
Add chunk importance signals (complexity, references count)
Prefix language + type for better semantic separation
Example Enhanced Text:
[PYTHON_FUNCTION] parse_file
Location: src/preprocessing/parser.py::100-150
Dependencies: [pathspec, tree_sitter, ConfigDict]
References: [LanguageConfig, Parser]
Defines: [parse_file]
Complexity: 3 (medium)

Parse an entire project directory and return CodeChunk objects...
[Code snippet here]

Effort: 3-4 days

Success Criteria:

All chunks have enriched embedding text
Token count validation passes (max 8,192)
Embedding quality improves measurably
Tests pass for preprocessing
Dependencies: None (preprocessing already complete)

1.4: Add Health Checks & Diagnostics
Task: Create src/health_check.py to verify all system components.

New File: src/health_check.py (100 lines)

Features:
async def health_check_all() -> dict:
    return {
        "embedding_service": await check_embedding_service(),
        "qdrant_service": await check_qdrant_service(),
        "neo4j_service": await check_neo4j_service(),  # Will fail until Phase 2
        "llm_service": await check_llm_service(),
    }

Effort: 2-3 days

Success Criteria:

Health check endpoint available
Integrated into FastAPI (GET /health)
Provides detailed status for each component
Dependencies: None

1.5: Create Vector Retrieval Tests
Task: Enhance test_retrieval_ext.py with comprehensive vector retrieval tests.

New Tests:

Dense vector search accuracy
Sparse BM25 search accuracy
Fusion algorithm correctness
Payload filtering
Deduplication
Effort: 3-4 days

Success Criteria:

All tests pass
Coverage > 80% for retrieval module
Performance benchmarks established
Dependencies: None

Subtotal Phase 1: 2 weeks

Phase 2: Knowledge Graph Layer (2-3 weeks)
Goal: Build Neo4j integration, graph schema, and graph-based retrieval.

2.1: Define Graph Schema
Task: Create src/graph/graph_schema.py with node and relationship type definitions.

New File: src/graph/graph_schema.py (150-200 lines)

Schema Definition:

# Node Types (from chunk.type)
NODE_TYPES = {
    "function": NodeType(label="Function", properties=["name", "qualified_name", "complexity"]),
    "class": NodeType(label="Class", properties=["name", "qualified_name", "inheritance"]),
    "file": NodeType(label="File", properties=["path", "language"]),
    "module": NodeType(label="Module", properties=["name", "path"]),
    "method": NodeType(label="Method", properties=["name", "parent_class"]),
    "property": NodeType(label="Property", properties=["name"]),
    # ... HTML, CSS, JS types
}

# Relationship Types (from chunk.relationships and chunk metadata)
RELATIONSHIP_TYPES = {
    "DEPENDS_ON": "A depends on B (import/external dependency)",
    "REFERENCES": "A references B (uses symbol B)",
    "IMPORTS": "A imports from B (module import)",
    "CALLS": "A calls B (function call)",
    "INHERITS_FROM": "A inherits from B (class inheritance)",
    "CONTAINS": "A contains B (file contains function)",
    "IN_MODULE": "A is in module B",
    "RENDERS_TEMPLATE": "View renders template",
    "MATCHES_CSS": "HTML element matches CSS selector",
    # ... more relationship types
}

# Constraints
CONSTRAINTS = [
    "CREATE CONSTRAINT unique_chunk_id IF NOT EXISTS FOR (c:CodeChunk) REQUIRE c.id IS UNIQUE",
]

# Indexes
INDEXES = [
    "CREATE INDEX IF NOT EXISTS ON (c:CodeChunk) (name)",
    "CREATE INDEX IF NOT EXISTS ON (c:CodeChunk) (qualified_name)",
    "CREATE INDEX IF NOT EXISTS ON (c:CodeChunk) (type)",
    "CREATE INDEX IF NOT EXISTS ON (c:CodeChunk) (language)",
]

Effort: 1-2 days

Success Criteria:

Schema is comprehensive and covers all chunk types
Schema is documented with examples
Schema can be instantiated and validated
Dependencies: None

2.2: Implement Neo4j Client
Task: Create src/graph/neo4j_client.py with CRUD operations.

New File: src/graph/neo4j_client.py (300-400 lines)

API:

class Neo4jClient:
    def __init__(self, config: Neo4jConfig):
        self.driver = GraphDatabase.driver(config.uri, auth=(config.username, config.password))

    async def create_node(self, label: str, properties: dict) -> dict:
        """Create a single node"""

    async def create_relationship(self, from_id: str, rel_type: str, to_id: str, props: dict = None) -> bool:
        """Create a relationship between two nodes"""

    async def batch_create_nodes(self, label: str, items: list[dict]) -> int:
        """Batch create nodes with MERGE (upsert)"""

    async def batch_create_relationships(self, relationships: list) -> int:
        """Batch create relationships with MERGE"""

    async def query(self, cypher: str, parameters: dict = None) -> list[dict]:
        """Execute Cypher query"""

    async def find_dependents(self, chunk_id: str, depth: int = 1) -> list:
        """Find all chunks that depend on this chunk"""

    async def find_dependencies(self, chunk_id: str, depth: int = 1) -> list:
        """Find all chunks this depends on"""

    async def find_shortest_path(self, from_id: str, to_id: str) -> list:
        """Find shortest path between two chunks"""

    async def get_neighbors(self, chunk_id: str, depth: int = 1, relationship_types: list = None) -> dict:
        """Get all neighbors within depth hops"""

    async def ping(self) -> bool:
        """Test connectivity"""

    async def close(self):
        """Close connection"""

Effort: 3-4 days

Success Criteria:

All methods implemented and tested
Error handling for Neo4j errors
Async support for concurrent operations
Performance: batch operations complete in < 5 seconds for 1000 items
Dependencies:

neo4j>=5.0.0 (must add to pyproject.toml)
src/graph/graph_schema.py
2.3: Implement Graph Builder
Task: Create src/graph/graph_builder.py to convert chunks to graph.

New File: src/graph/graph_builder.py (250-300 lines)

API:

class GraphBuilder:
    def __init__(self, neo4j_client: Neo4jClient, schema: GraphSchema):
        self.client = neo4j_client
        self.schema = schema

    async def build_from_chunks(self, chunks: list[CodeChunk]) -> dict:
        """
        Convert chunks to graph:
        1. Create nodes for each chunk
        2. Create nodes for symbols
        3. Create edges from relationships
        4. Create file/module hierarchy
        """
        stats = {
            "nodes_created": 0,
            "relationships_created": 0,
            "errors": [],
            "duration": 0,
        }

    async def sync_chunk(self, chunk: CodeChunk) -> bool:
        """Update single chunk in graph (for incremental updates)"""

    async def delete_file_chunks(self, file_path: str) -> int:
        """Delete all chunks for a file (for updates)"""

    async def get_build_report(self) -> dict:
        """Report on graph structure and consistency"""

Implementation Details:

Use MERGE to create idempotent operations
Batch operations for performance (1000 nodes/relationships per batch)
Deduplication via SHA256 hash of chunk ID
Relationship creation from chunk.relationships and chunk.dependencies
Error handling with detailed logging
Effort: 3-4 days

Success Criteria:

Builds complete graph from chunk set
Handles incremental updates
Performance: 10,000 chunks → graph in < 30 seconds
Consistency: No orphaned nodes, all edges have targets
Tests pass with sample chunk sets
Dependencies:

neo4j_client.py
graph_schema.py
2.4: Implement Graph Retrieval Engine
Task: Create src/graph/query_engine.py for Cypher-based retrieval.

New File: src/graph/query_engine.py (200-250 lines)

API:

class GraphQueryEngine:
    def __init__(self, neo4j_client: Neo4jClient):
        self.client = neo4j_client

    async def find_by_name(self, name: str, chunk_type: str = None, limit: int = 10) -> list[CodeChunk]:
        """Find chunks by name"""

    async def find_dependents(self, chunk_id: str, depth: int = 2, limit: int = 50) -> list:
        """Find all chunks that depend on this one"""

    async def find_dependencies(self, chunk_id: str, depth: int = 2, limit: int = 50) -> list:
        """Find all chunks this depends on"""

    async def find_symbols_in_file(self, file_path: str) -> list:
        """Get all chunks in a file"""

    async def find_related_tests(self, chunk_id: str) -> list:
        """Find test files related to chunk"""

    async def analyze_impact(self, chunk_id: str) -> dict:
        """Analyze what changes if this chunk is modified"""

    async def translate_query_to_cypher(self, natural_language: str) -> str:
        """Use LLM to convert query to Cypher (optional enhancement)"""

Cypher Query Examples:
// Find dependents
MATCH (chunk:CodeChunk {id: $chunk_id})<-[:DEPENDS_ON|REFERENCES|CALLS]-(dependent)
RETURN dependent

// Find dependencies
MATCH (chunk:CodeChunk {id: $chunk_id})-[:DEPENDS_ON|REFERENCES|CALLS]->(dep)
RETURN dep

// Trace impact
MATCH (chunk:CodeChunk {id: $chunk_id})<-[:DEPENDS_ON*1..3]-(impacted)
RETURN DISTINCT impacted

// Find shortest path
MATCH path = shortestPath((from:CodeChunk {id: $from_id})-[*]-(to:CodeChunk {id: $to_id}))
RETURN path

Effort: 3-4 days

Success Criteria:

All query methods implemented
Queries execute in < 100ms for typical graphs (10K nodes)
Results returned in CodeChunk format for consistency
Documentation with query examples
Dependencies:

neo4j_client.py
2.5: Integrate Graph into Ingestion Pipeline
Task: Modify CLI and main.py to index into both Qdrant and Neo4j.

Files Modified:

cli.py → Add --enable-graph flag to ingest command
main.py → Parallel indexing to Qdrant + Neo4j
Create new CLI command: graph-index for standalone graph building
New Commands:
# Ingest and index both vector + graph
python -m src.cli ingest --path . --enable-graph

# Index existing chunks into graph
python -m src.cli graph-index --input chunks.json

# Query graph directly
python -m src.cli graph-query --query "What calls parse_file?"

Effort: 2-3 days

Success Criteria:

CLI commands work end-to-end
Both Qdrant and Neo4j are populated
Chunk IDs are consistent across systems
Tests pass for ingestion pipeline
Dependencies:

All Phase 2 components above
2.6: Create Graph Retrieval Tests
Task: Add comprehensive tests for graph operations.

New Tests (tests/test_graph.py):

Neo4j client CRUD operations
Graph builder chunk-to-graph conversion
Graph query engine retrieval accuracy
Relationship correctness
Integration with preprocessing chunks
Effort: 2-3 days

Success Criteria:

80%+ coverage of graph modules
All tests pass
Performance benchmarks established
Dependencies: None (tests only)

Subtotal Phase 2: 2-3 weeks

Phase 3: Agentic Orchestration Layer (3-4 weeks)
Goal: Build agent infrastructure, tools, and multi-step reasoning capabilities.

3.1: Define Agent Architecture
Task: Create src/agents/agent_config.py with agent settings and tool definitions.

New File: src/agents/agent_config.py (100-150 lines)

Configuration:
@dataclass
class AgentConfig:
    model_name: str = "ai/llama3.2:latest"
    model_url: str = "http://localhost:12434"
    temperature: float = 0.3
    max_iterations: int = 10
    memory_type: Literal["buffer", "summary", "none"] = "buffer"
    max_memory_tokens: int = 4096
    verbose: bool = True
    enable_graph_tools: bool = True
    enable_vector_tools: bool = True
    enable_test_tools: bool = True

class ToolDefinition:
    name: str
    description: str
    input_schema: dict  # JSON Schema
    output_type: str
    async_capable: bool
    requires_graph: bool = False

Effort: 1-2 days

Success Criteria:

Config validates and loads from .env
Tool definitions are comprehensive
Integration with LangChain StructuredTool
Dependencies: None

3.2: Implement Tool Registry
Task: Create src/agents/tool_registry.py to manage tools.

New File: src/agents/tool_registry.py (150-200 lines)

API:

class ToolRegistry:
    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool"""

    def get_tool(self, name: str) -> BaseTool:
        """Get tool by name"""

    def get_all_tools(self, filter_enabled: bool = True) -> list[BaseTool]:
        """Get all registered tools"""

    def get_tools_for_query(self, query: str) -> list[BaseTool]:
        """Suggest tools for a query (simple heuristic)"""

    def validate_tool_result(self, tool_name: str, result: Any) -> bool:
        """Validate tool output"""

Effort: 2 days

Success Criteria:

Registry manages tools
Tools can be enabled/disabled via config
Tool discoverability works
Dependencies: None

3.3: Implement Core Agent Tools
Task: Create src/agents/tools/ with 5-6 core tools.

New Files (src/agents/tools/):

3.3.1: CodeSearchTool (code_search.py, 100 lines)
class CodeSearchTool(BaseTool):
    """Search code by semantic similarity using vector retrieval"""
    name = "code_search"
    description = "Search for code snippets matching a query"

    async def _arun(self, query: str, limit: int = 5, language: str = None) -> list[CodeChunk]:
        # Use existing vector retrieval (HybridSearchEngine)

3.3.2: GraphQueryTool (graph_query.py, 150 lines)
class GraphQueryTool(BaseTool):
    """Query knowledge graph for relationships"""
    name = "graph_query"
    description = "Find relationships between code chunks"

    async def _arun(self, query_type: str, chunk_id: str, depth: int = 2) -> list:
        # query_type: "dependents", "dependencies", "related", "impact"
        # Use GraphQueryEngine

3.3.3: DependencyAnalyzerTool (dependency_analyzer.py, 120 lines)
class DependencyAnalyzerTool(BaseTool):
    """Analyze dependencies and imports"""
    name = "analyze_dependencies"
    description = "Trace dependency chains and identify circular imports"

    async def _arun(self, chunk_id: str, show_circular: bool = True) -> dict:
        # Return dependency tree, circular dependencies, impact analysis

3.3.4: TestLocatorTool (test_locator.py, 100 lines)
class TestLocatorTool(BaseTool):
    """Find tests related to code"""
    name = "find_tests"
    description = "Locate test files for given code"

    async def _arun(self, chunk_id: str, test_type: str = "all") -> list:
        # test_type: "unit", "integration", "all"
        # Use graph (TEST_FOR relationships) or pattern matching

3.3.5: DocumentationTool (documentation.py, 100 lines)
class DocumentationTool(BaseTool):
    """Search documentation and docstrings"""
    name = "search_documentation"
    description = "Find documentation related to query"

    async def _arun(self, query: str) -> list[str]:
        # Search docstrings, comments, markdown files
        # Use vector search on documentation chunks

3.3.6: SymbolResolverTool (symbol_resolver.py, 120 lines)
class SymbolResolverTool(BaseTool):
    """Resolve symbols to definitions"""
    name = "resolve_symbol"
    description = "Find definition of a symbol"

    async def _arun(self, symbol_name: str, language: str = None) -> CodeChunk:
        # Find where symbol is defined
        # Use graph (DEFINES relationships) or vector search

Effort: 5-7 days (1 day per tool)

Success Criteria:

All tools are implemented and tested
Tools integrate with LangChain BaseTool interface
Tool results are consistent and validated
Documentation includes usage examples
Dependencies:

tool_registry.py
graph_query_engine.py
hybrid_search_engine.py
3.4: Implement Agent Memory & State
Task: Create src/agents/memory/ for conversation tracking.

New Files (src/agents/memory/):

3.4.1: ConversationMemory (conversation.py, 150 lines)
class ConversationMemory:
    """Manage conversation history and reasoning traces"""

    async def add_message(self, role: str, content: str, metadata: dict = None):
        """Add message to history (user, assistant, tool)"""

    async def get_context(self, max_tokens: int = 4096) -> str:
        """Get conversation history within token limit"""

    async def get_reasoning_trace(self) -> list:
        """Get trace of tool calls and decisions"""

    async def clear_history(self):
        """Clear conversation"""

3.4.2: QueryState (state.py, 100 lines)
@dataclass
class QueryState:
    """Agent execution state"""
    query: str
    planning_steps: list[str]
    retrieved_chunks: list[CodeChunk]
    tool_calls: list[dict]
    reasoning: str
    answer: str
    confidence: float
    metadata: dict

Effort: 2-3 days

Success Criteria:

Memory persists across turns
Token counting is accurate
State can be serialized/logged
Dependencies: None

3.5: Implement Agent Executor
Task: Create src/agents/agent_executor.py with the main agent loop.

New File: src/agents/agent_executor.py (300-400 lines)

API:
class CodeAgentExecutor:
    def __init__(self, config: AgentConfig, tool_registry: ToolRegistry, memory: ConversationMemory):
        self.config = config
        self.tools = tool_registry
        self.memory = memory
        self.llm = LangChainLLM(config)  # LangChain wrapper

    async def execute(self, query: str) -> dict:
        """
        Main agent loop:
        1. Parse query and plan steps
        2. Identify relevant tools
        3. Execute tools
        4. Reason over results
        5. Generate answer
        """
        state = QueryState(query=query)

        # Step 1: Planning
        state.planning_steps = await self._plan_steps(query)

        # Step 2-4: Tool execution loop
        for iteration in range(self.config.max_iterations):
            # Get next action from LLM
            action = await self._get_next_action(state)

            if action.type == "finish":
                break
            elif action.type == "tool_use":
                result = await self._execute_tool(action)
                state.tool_calls.append({
                    "tool": action.tool_name,
                    "input": action.tool_input,
                    "output": result,
                })

            # Add to memory
            await self.memory.add_message("agent", f"Used {action.tool_name}")

        # Step 5: Generate answer
        state.answer = await self._generate_answer(state)
        return state.to_dict()

    async def _plan_steps(self, query: str) -> list[str]:
        """Use LLM to create action plan"""

    async def _get_next_action(self, state: QueryState) -> Action:
        """Use LLM to decide next action"""

    async def _execute_tool(self, action: Action) -> Any:
        """Execute tool and return result"""

    async def _generate_answer(self, state: QueryState) -> str:
        """Synthesize final answer from tool results"""

Implementation Strategy:

Use LangChain AgentExecutor with ReAct (Reasoning + Acting) pattern
LLM function calling to select tools
Structured output parsing for tool inputs
Error handling and tool retry logic
Verbose logging for debugging
Effort: 5-7 days

Success Criteria:

Agent executes end-to-end
Tool selection is accurate
Multi-step reasoning works
Performance: agent completes query in < 30 seconds
Tests pass with example queries
Dependencies:

All tools from Phase 3.3
memory.py, agent_config.py, tool_registry.py
LangChain 1.0.8+
3.6: Integrate Agent into API & CLI
Task: Add agent endpoints and CLI commands.

Files Modified:

main.py → New FastAPI endpoints for agent queries
cli.py → New CLI commands for agent interaction
New Endpoints (FastAPI):
@app.post("/query/agent")
async def agent_query(request: QueryRequest) -> QueryResponse:
    """Execute query using agentic RAG"""
    executor = CodeAgentExecutor(...)
    result = await executor.execute(request.query)
    return QueryResponse(...)

@app.get("/agent/tools")
async def list_agent_tools() -> list[ToolInfo]:
    """List available agent tools"""

@app.post("/agent/debug")
async def debug_agent_execution(request: QueryRequest) -> dict:
    """Execute query with detailed debugging output"""

New CLI Commands:
# Interactive agent mode
python -m src.cli agent --interactive

# Single query
python -m src.cli agent-query --query "What calls parse_file?"

# Debug mode with verbose output
python -m src.cli agent --interactive --debug

# List available tools
python -m src.cli agent-tools

Effort: 2-3 days

Success Criteria:

API endpoints work end-to-end
CLI commands are user-friendly
Documentation includes usage examples
Dependencies: All agent components

3.7: Create Agent Tests & Benchmarks
Task: Comprehensive testing of agent functionality.

New Tests (tests/test_agent.py):

Tool registry functionality
Individual tool execution
Agent planning and routing
Multi-step reasoning accuracy
Memory management
Error handling and retries
Integration tests with all components
Benchmarks (tests/benchmarks/agent_benchmarks.py):

Query execution time
Tool selection accuracy
Token usage per query
Success rate of complex queries
Effort: 3-4 days

Success Criteria:

80%+ test coverage
All tests pass
Benchmarks establish baseline performance
Dependencies: None (tests only)

Subtotal Phase 3: 3-4 weeks

Phase 4: Hybrid Retrieval & Integration (1-2 weeks)
Goal: Unify vector + graph retrieval, optimize performance, and integrate all layers.

4.1: Implement Hybrid Retrieval Orchestrator
Task: Create src/retrieval/hybrid_orchestrator.py to coordinate vector + graph retrieval.

New File: src/retrieval/hybrid_orchestrator.py (200-250 lines)

API:
class HybridRetrievalOrchestrator:
    def __init__(self, vector_engine: HybridSearchEngine, graph_engine: GraphQueryEngine,
                 config: HybridConfig):
        self.vector = vector_engine
        self.graph = graph_engine
        self.config = config

    async def retrieve(self, query: str, retrieval_strategy: str = "auto") -> list[CodeChunk]:
        """
        Unified retrieval:
        - "vector" → vector-only (semantic)
        - "graph" → graph-only (structural)
        - "hybrid" → both + fusion
        - "auto" → LLM selects strategy
        """

    async def _detect_query_intent(self, query: str) -> str:
        """Classify query as structural/semantic/hybrid"""

    async def _fuse_results(self, vector_results: list, graph_results: list) -> list:
        """Merge and dedupe results with unified ranking"""

Effort: 2-3 days

Success Criteria:

Orchestrator selects appropriate retrieval strategy
Hybrid fusion produces high-quality results
Performance: retrieval completes in < 500ms
Dependencies:

graph_query_engine.py
hybrid_search_engine.py
4.2: Create Unified Results Combiner
Task: Create result fusion algorithm for vector + graph combining.

New File: src/retrieval/result_combiner.py (150-200 lines)

API:
class ResultsCombiner:
    def fuse_vector_and_graph(self, vector_results: list[CodeChunk],
                              graph_results: list[CodeChunk],
                              strategy: str = "weighted_average") -> list[CodeChunk]:
        """
        Combine results from both retrievers:
        - strategy: "weighted_average", "reciprocal_rank_fusion", "weighted_sum"
        - Remove duplicates
        - Rerank by combined score
        """

    def deduplicate(self, results: list[CodeChunk]) -> list[CodeChunk]:
        """Remove duplicate chunks by ID"""

    def rerank_combined(self, results: list[CodeChunk], query: str) -> list[CodeChunk]:
        """Rerank combined results using cross-encoder or LLM"""

Effort: 2-3 days

Success Criteria:

Fusion produces better results than individual retrievers
Deduplication works correctly
Reranking improves relevance
Dependencies: None

4.3: Implement Graph-Aware Reranking
Task: Extend reranker to consider graph connectivity.

Task: Modify cross_encoder.py to optionally include graph scores.

New Features:
class GraphAwareCrossEncoder:
    def rerank_with_graph_signals(self, query: str, candidates: list[CodeChunk],
                                   graph_scores: dict = None) -> list[CodeChunk]:
        """
        Rerank considering:
        - Cross-encoder semantic score
        - Graph connectivity to query context
        - Chunk importance (in-degree/out-degree)
        """

Effort: 1-2 days

Success Criteria:

Graph signals improve reranking accuracy
Performance impact is minimal
Dependencies:

graph_query_engine.py
4.4: Integrate Hybrid Retrieval into API & CLI
Task: Update API and CLI to use hybrid retrieval.

Files Modified:

main.py → Update endpoints to support retrieval strategy selection
cli.py → Add --retrieval-strategy flag
New Endpoints:
@app.post("/query/hybrid")
async def hybrid_query(request: QueryRequest) -> QueryResponse:
    """Execute query with hybrid retrieval"""

@app.get("/query/strategies")
async def get_retrieval_strategies() -> list[str]:
    """List available retrieval strategies"""

Effort: 1-2 days

Success Criteria:

Hybrid retrieval works end-to-end
API clearly documents available strategies
CLI examples include hybrid usage
Dependencies: All hybrid components

4.5: Performance Optimization
Task: Benchmark and optimize all layers.

Optimizations to Consider:

Caching: Cache frequently accessed chunks, graph traversals
Batch Processing: Batch tool calls in agent execution
Parallel Execution: Run vector + graph retrieval in parallel
Connection Pooling: Neo4j and Qdrant connection optimization
Index Tuning: Database index optimization
Benchmarking (tests/benchmarks/performance.py):

Query latency (vector, graph, hybrid, agent)
Throughput (queries per second)
Memory usage
Database size growth
Effort: 2-3 days

Success Criteria:

Hybrid query completes in < 500ms
Agent query completes in < 30 seconds
Throughput: > 10 queries/second
Database size: < 2GB for 10K chunks
Dependencies: None (performance tuning)

4.6: Documentation & Examples
Task: Create comprehensive documentation and usage examples.

New Files:

docs/GAV_RAG_GUIDE.md → User guide for all features
docs/AGENT_TOOLS.md → Tool reference
docs/QUERY_EXAMPLES.md → Example queries and expected results
examples/agent_queries.py → Python code examples
Effort: 2-3 days

Success Criteria:

Users can understand all new features
Examples are clear and executable
Documentation is searchable and complete
Dependencies: None

4.7: Integration Testing
Task: End-to-end testing of full GAV pipeline.

New Tests (tests/test_integration_gav.py):

Full pipeline: ingest → embed → index (Qdrant + Neo4j) → retrieve → rerank → generate
Hybrid retrieval accuracy
Agent multi-step reasoning
Error handling and fallbacks
Performance under load
Effort: 2-3 days

Success Criteria:

All integration tests pass
System behaves correctly end-to-end
Fallback mechanisms work when components are unavailable
Dependencies: All GAV components

Subtotal Phase 4: 1-2 weeks

Part IV: Implementation Timeline & Effort Estimation
Overall Timeline
Phase 1 (Vector Foundations):     2 weeks    [Week 1-2]
Phase 2 (Graph Layer):            2-3 weeks  [Week 3-5]
Phase 3 (Agentic Orchestration):  3-4 weeks  [Week 6-9]
Phase 4 (Integration & Polish):   1-2 weeks  [Week 10-11]
                                   ──────────
Total GAV Migration:              8-11 weeks [2-3 months]
Effort Distribution by Role
Phase	Backend	DevOps	QA/Testing	Product	Total
Phase 1	4d	1d	3d	1d	2w
Phase 2	12d	2d	3d	1d	2-3w
Phase 3	18d	2d	4d	2d	3-4w
Phase 4	6d	2d	3d	2d	1-2w
Total	40d	7d	13d	6d	8-11w
Interpretation:

Backend: Primary implementation work (~40 developer-days)
DevOps: Deployment, infrastructure setup (~7 developer-days)
QA/Testing: Test coverage, validation (~13 developer-days)
Product: Documentation, user communication (~6 developer-days)
Dependency Graph
Phase 1 (Vector Foundations)
├─ Embedding Config Enhancements
├─ Qdrant Config Enhancements
├─ Enhanced Embedding Text
├─ Health Checks
└─ Vector Retrieval Tests
    ↓
Phase 2 (Graph Layer)
├─ Graph Schema Definition
├─ Neo4j Client Implementation
├─ Graph Builder
├─ Graph Query Engine
├─ Ingestion Pipeline Integration
└─ Graph Tests
    ↓
Phase 3 (Agentic Orchestration)
├─ Agent Config
├─ Tool Registry
├─ Core Tools (5-6)
├─ Agent Memory
├─ Agent Executor
├─ API & CLI Integration
└─ Agent Tests
    ↓
Phase 4 (Integration & Optimization)
├─ Hybrid Retrieval Orchestrator
├─ Result Combiner
├─ Graph-Aware Reranking
├─ API/CLI Integration
├─ Performance Optimization
├─ Documentation
└─ Integration Tests


Critical Path
Must Complete in Order:

Phase 1 → Foundation (cannot be skipped)
Phase 2 → Graph (required for hybrid retrieval)
Phase 3 → Agents (optional but provides significant value)
Phase 4 → Integration (ties everything together)
Parallelizable Work (Can happen in parallel):

Phase 1 components (except tests depend on implementation)
Phase 2 Neo4j client vs Graph builder (after schema is defined)
Phase 3 individual tools (after tool registry exists)
Part V: Success Criteria & Risk Management
Success Criteria by Phase
Phase 1 Success
✅ All configuration changes validated
✅ Enhanced embedding text improves retrieval by 5-10%
✅ Health checks detect all system failures
✅ Vector retrieval tests have 80%+ coverage
✅ No impact on existing vector retrieval performance
Phase 2 Success
✅ Graph contains all chunks and relationships from preprocessing
✅ Graph-based retrieval finds relevant chunks
✅ Dependency analysis works correctly (no orphaned nodes)
✅ Graph queries complete in < 100ms
✅ Graph indexing is idempotent (can re-run without duplication)
✅ Graph tests have 80%+ coverage
Phase 3 Success
✅ All 5-6 tools are implemented and tested
✅ Agent planning creates reasonable multi-step plans
✅ Tool execution and result handling work correctly
✅ Agent successfully answers multi-step queries
✅ Memory persists conversation context
✅ Agent tests have 80%+ coverage
✅ Agent queries complete in < 30 seconds
Phase 4 Success
✅ Hybrid retrieval combines vector + graph results effectively
✅ Result fusion improves accuracy vs individual retrievers
✅ Hybrid query latency is < 500ms
✅ Graph-aware reranking improves ranking by 5-10%
✅ API provides clear endpoints for all retrieval strategies
✅ Documentation is comprehensive and user-friendly
✅ End-to-end integration tests pass
Risk Management
High-Risk Areas
Risk 1: Graph Consistency (Probability: Medium, Impact: High)

Issue: Qdrant and Neo4j can diverge (chunks updated in Qdrant but not in graph)
Mitigation:
Use consistent chunk IDs across systems
Implement batch sync operations
Add consistency checks in health monitoring
Document manual sync procedures
Risk 2: Agent Performance (Probability: Medium, Impact: Medium)

Issue: Agent may be slow (multiple LLM calls + tool executions)
Mitigation:
Cache LLM responses for common patterns
Parallelize tool execution where possible
Set execution timeouts
Provide fast-path alternatives (vector-only for simple queries)
Risk 3: Neo4j Scalability (Probability: Low, Impact: Medium)

Issue: Graph grows too large or queries become slow
Mitigation:
Use Neo4j profiling to identify slow queries
Create appropriate indexes (done in Phase 2.1)
Implement query result caching
Plan for sharding/clustering if needed (future enhancement)
Risk 4: Tool Integration Complexity (Probability: Medium, Impact: Medium)

Issue: Tool outputs don't integrate cleanly with agent executor
Mitigation:
Enforce strict tool output schemas
Add validation layers
Provide comprehensive tool testing
Create tool development guidelines
Medium-Risk Areas
Risk 5: Backward Compatibility (Probability: Low, Impact: Low)

Issue: Changes break existing vector-only functionality
Mitigation:
All new features are additive (no changes to existing code)
Comprehensive regression testing
Feature flags for new functionality
Risk 6: Configuration Complexity (Probability: Medium, Impact: Low)

Issue: Too many configuration options confuse users
Mitigation:
Provide sensible defaults for all new configs
Documentation with clear examples
Validation errors that guide users to solutions
Mitigation Strategies (General)
Regular Integration Testing: Run end-to-end tests at phase boundaries
Performance Monitoring: Establish performance baselines, monitor regressions
Documentation: Maintain up-to-date docs as implementation progresses
Code Review: Mandatory review of all PRs, especially for core layers
Rollback Plans: Keep vector-only pipeline as fallback
Stakeholder Communication: Regular updates on progress and challenges

Part VI: Code Structure Changes
New Modules to Create

src/
├── graph/                              # NEW - Graph RAG Layer (Phase 2)
│   ├── __init__.py
│   ├── graph_schema.py                 # Node/relationship type definitions
│   ├── neo4j_client.py                 # CRUD operations
│   ├── graph_builder.py                # Chunk → Graph conversion
│   └── query_engine.py                 # Cypher-based retrieval
│
├── agents/                             # NEW - Agentic Orchestration (Phase 3)
│   ├── __init__.py
│   ├── agent_config.py                 # Agent settings
│   ├── agent_executor.py               # Main agent loop (ReAct pattern)
│   ├── tool_registry.py                # Tool management
│   ├── memory/                         # NEW
│   │   ├── __init__.py
│   │   ├── conversation.py             # Conversation history
│   │   └── state.py                    # Query state tracking
│   └── tools/                          # NEW
│       ├── __init__.py
│       ├── code_search.py              # Semantic search tool
│       ├── graph_query.py              # Graph traversal tool
│       ├── dependency_analyzer.py      # Dependency analysis tool
│       ├── test_locator.py             # Test discovery tool
│       ├── documentation.py            # Documentation search tool
│       └── symbol_resolver.py          # Symbol resolution tool
│
├── retrieval/                          # MODIFIED - Add hybrid orchestration
│   ├── hybrid_orchestrator.py          # NEW - Unified retrieval
│   ├── result_combiner.py              # NEW - Result fusion
│   └── (existing files preserved)
│
├── reranking/                          # MODIFIED - Add graph-aware scoring
│   ├── cross_encoder.py                # Enhanced with graph signals
│   └── (existing files preserved)
│
├── health_check.py                     # NEW (Phase 1) - System diagnostics
│
└── config/
    ├── neo4j_config.py                 # MODIFIED - Already exists, needs driver
    └── (existing files preserved)

docs/
├── architecture_claude_gav_rag.md      # NEW - This document
├── claudearchitecture.mermaid          # NEW - Target architecture diagram
├── GAV_RAG_GUIDE.md                    # NEW - User guide
├── AGENT_TOOLS.md                      # NEW - Tool reference
├── QUERY_EXAMPLES.md                   # NEW - Example queries
└── (existing files preserved)

examples/
├── agent_queries.py                    # NEW - Python examples
└── (existing files preserved)

tests/
├── test_graph.py                       # NEW (Phase 2)
├── test_agent.py                       # NEW (Phase 3)
├── test_integration_gav.py             # NEW (Phase 4)
├── benchmarks/
│   ├── agent_benchmarks.py             # NEW (Phase 3)
│   └── performance.py                  # NEW (Phase 4)
└── (existing files preserved)


Files to Modify Minimally
The following files should be extended with minimal changes (no breaking changes to existing functionality):

embedding_config.py: Add new optional fields for graph-aware embedding
qdrant_config.py: Add new optional fields for graph sync
preprocessor.py: Enhance embedding_text generation
main.py: Add new CLI commands and FastAPI endpoints
cli.py: Add new CLI commands
search.py: Add optional graph-aware metadata tracking
cross_encoder.py: Add optional graph signal support
pyproject.toml: Add neo4j>=5.0.0 dependency
Dependencies to Add
New Python Dependencies (to pyproject.toml):
[project]
dependencies = [
    # Existing
    "fastapi>=0.121.1",
    "uvicorn>=0.38.0",
    "pydantic>=2.12.4",
    "python-dotenv>=1.2.1",
    "qdrant-client>=1.15.1",
    # ... other existing deps

    # NEW for GAV
    "neo4j>=5.0.0",              # Neo4j driver
    "langchain>=1.0.8",           # Already exists, confirm version
    "playwright>=1.40.0",         # For optional rendering
]

No Other External Dependencies Needed (LangChain already provides agent framework)

Part VII: Implementation Checklist
Phase 1 Checklist (Vector Foundations)
 Enhance embedding_config.py with new fields
 Enhance qdrant_config.py with graph sync options
 Update preprocessor.py to enrich embedding text with graph context
 Create src/health_check.py with system diagnostics
 Write vector retrieval tests (expand test_retrieval_ext.py)
 Update QWEN.md with Phase 1 changes
 Test end-to-end (ingest → embed → index)
 Verify no regression in existing vector retrieval
Phase 2 Checklist (Graph Layer)
 Create src/graph/graph_schema.py with complete schema
 Create src/graph/neo4j_client.py with CRUD operations
 Create src/graph/graph_builder.py for chunk-to-graph conversion
 Create src/graph/query_engine.py for Cypher-based retrieval
 Add --enable-graph flag to ingestion pipeline
 Create graph-index CLI command
 Create graph-query CLI command
 Write comprehensive graph tests (tests/test_graph.py)
 Verify graph consistency (no orphaned nodes, correct relationships)
 Benchmark graph query performance
 Update QWEN.md with Phase 2 changes
Phase 3 Checklist (Agentic Orchestration)
 Create src/agents/agent_config.py
 Create src/agents/tool_registry.py
 Create all 5-6 tools in src/agents/tools/
 Create src/agents/memory/conversation.py
 Create src/agents/memory/state.py
 Create src/agents/agent_executor.py with ReAct pattern
 Add POST /query/agent endpoint to FastAPI
 Add agent and agent-query CLI
