This is a highly specific and advanced use case: **Automated Root Cause Analysis (RCA) via RAG**. Because your goal is to feed raw error logs (stack traces, console errors) and get a solution, standard text-based RAG will fail. You need a **Structural and Code-Aware RAG**.

Here is the blueprint for building the "Perfect" RAG pipeline for Django/JS error resolution.

---

### 1. The Tech Stack & Python Packages

To build a pipeline that understands *code structure* (not just text), you need these specific tools:

#### Core Frameworks

* **LlamaIndex (`llama-index`):** *Why?* It is currently superior to LangChain for complex indexing strategies (Hierarchical indices, Knowledge Graphs) which are essential for codebases.
* **Tree-sitter (`tree-sitter`, `tree-sitter-python`, `tree-sitter-javascript`):** *Why?* You cannot simply split code by "characters." You must split by "Abstract Syntax Tree" (AST)—classes, functions, methods. This tool is non-negotiable for accurate code chunking.

#### Vector & Graph Databases

* **Qdrant (`qdrant-client`):** *Why?* Best-in-class support for **Hybrid Search** (Dense Vectors + Sparse/Keyword search). Error logs often contain specific variable names (keywords) that semantic search misses.
* **Neo4j (`neo4j`):** *Why?* For **GraphRAG**. Django is relational (View → Model → Serializer). If an error happens in a View, the RAG needs to fetch the related Model, even if the Model isn't mentioned in the stack trace.

#### Embeddings & Ranking

* **Voyage AI (`voyageai`):** *Why?* Their `voyage-code-2` model is currently SOTA for code retrieval, beating OpenAI on coding benchmarks.
* **Cohere Rerank (`cohere`):** *Why?* After fetching 20 code snippets, the Reranker will ensure the snippet containing the *actual* bug is at the top.

#### Utilities for Django/JS

* **Unstructured (`unstructured`):** For parsing HTML/Django Templates.
* **Esprima (Python wrappers):** For deeper parsing of JS/Alpine logic if Tree-sitter isn't enough.

---

### 2. Sources of Truth (The Knowledge Base)

Your RAG is only as good as the data it accesses. Do not just index your code. Index these specific "Truths":

1. **Your Codebase:** (Views, Models, URLs, Serializers, Templates, JS files).
2. **Django Source Code:** Index the actual GitHub repository of the Django version you are using. Often the error comes from deep within Django internals; the RAG needs to see that code to explain *why* it crashed.
3. **Documentation (MD/RST files):**
    * Django Official Docs.
    * HTMX Docs (crucial for `hx-swap` errors).
    * Alpine.js Docs.
4. **Sentry Issues (Historical):** Export your solved Sentry issues (Error + Solution) and index them. This is "Few-Shot" gold.

---

### 3. Best Practices & Indexing Strategies

#### A. The Semantic Code Index (Chunks)

* **Use Case:** Conceptual errors (e.g., "Why is my HTMX request returning 403?").
* **Implementation:** Use `CodeSplitter` (LlamaIndex) based on Tree-sitter. Chunk by `Class` and `Function`.
* **Trick:** Prepend the file path and class name to every chunk. An orphan function `def get_context_data` is useless without knowing it belongs to `ProfileDetailView`.

#### B. The Keyword Index (Sparse Vectors)

* **Use Case:** Exact error matching (e.g., `TemplateSyntaxError: Could not parse...`).
* **Implementation:** Use Qdrant's Sparse Vectors (BM25). When an error log says `variable_x is not defined`, you need to find the exact file where `variable_x` is defined, not a "semantically similar" variable.

#### C. The Knowledge Graph Index (Dependency Mapping)

* **Use Case:** "The error is in `views.py`, but the bug is in `models.py`."
* **Implementation:** Build a graph where nodes are files/functions and edges are `IMPORTS`, `INHERITS_FROM`, or `CALLS`.
  * *Query:* If stack trace points to `views.py`, the Graph retrieval traverses the edges to pull the imported `forms.py` and `models.py` into context.

---

### 4. Implementation Plan (Task List)

#### Phase 1: Environment & Parsing (The Foundation)

1. **Setup Tree-sitter:** Write a script that iterates through your `django_app/` directory.
2. **AST Parsing:** Use Tree-sitter to identify all classes and functions.
3. **Metadata Extraction:** For every function, extract:
    * File Path.
    * Function Name.
    * Arguments.
    * Import statements (to build the graph later).
4. **Chunking:** Save these as `Document` objects.

#### Phase 2: Indexing (The Brain)

5. **Hybrid Index Setup:** Initialize Qdrant with dense vectors (Voyage-code-2) and sparse vectors (BM25).
6. **Ingest Code:** Upsert your parsed Documents into Qdrant.
7. **Ingest Docs:** Scrape Django/HTMX docs and upsert them into a separate collection (or use metadata filtering).
8. **Ingest Logs:** Create a separate index for "Known Errors" (previous Sentry solutions).

#### Phase 3: The Retrieval Pipeline (The Engine)

9. **Query Pre-processing:**
    * *Input:* Raw Stack Trace / Console Log.
    * *Action:* Use a small LLM (like GPT-3.5 or specialized Regex) to extract **File Paths**, **Line Numbers**, and **Error Messages** from the raw log.
10. **The "Surrounding Context" Fetch:**
    * Take the File Path and Line Number from the error.
    * Programmatically fetch lines `N-50` to `N+50` from that specific file (bypass vector search for this—you need the exact crime scene).
11. **The Dependency Fetch:**
    * Use the Graph/Vector store to find: "What definitions are used in this crash site?" (e.g., if line 50 calls `calculate_total()`, fetch the code for `calculate_total` even if it's in another file).
12. **Ranking:**
    * Pass the "Crime Scene" + "Dependencies" + "Documentation matches" to Cohere Rerank. Keep the top 5 most relevant chunks.

#### Phase 4: Prompt Construction (The Output)

13. **Context Assembly:** Format the prompt:

    ```text
    ERROR LOG:
    [Paste Raw Error]

    CRIME SCENE (Exact Code):
    [Paste File Content around line number]

    RELATED DEFINITIONS (Retrieved via RAG):
    [Paste definition of functions called in the crime scene]

    RELEVANT DOCS:
    [Django/HTMX docs related to the error type]
    ```

14. **System Prompt:** "You are a Senior Django/HTMX Engineer. Analyze the error log against the provided code context. Fix the bug."

---

### 5. Instructional Videos & Resources

**To Learn the Concepts:**

* **Andrej Karpathy (Let's build GPT tokenizer/intro):** Understand how LLMs see code.
* **Greg Kamradt (Data Independent):** His tutorials on "Code Understanding with LangChain" are foundational.
* **Jerry Liu (LlamaIndex CEO):** Search for his talks on "Advanced RAG" and "GraphRAG".

**Specific Tutorials:**

1. **"Building a Code RAG with LangChain & DeepLake"** (Search this on YouTube - applies to Qdrant/LlamaIndex concepts too).
2. **"GraphRAG for Codebases"** (Look for Neo4j developer blog posts).
3. **"Tree-sitter Python Tutorial"** (Read the docs or search GitHub for implementation examples).

**Github Repos to Study (Sources of Truth):**

* **`sweepai/sweep`:** An open-source AI junior developer. Look at how they index code.
* **`cursor-ai`:** (Not open source, but read their blog on "Shadow Workspace" and indexing).
* **`gpt-engineer`:** Review their code parsing logic.

### 6. Summary of Use Cases for Your Pipeline

1. **Semantic RAG:** "How do I implement a modal using HTMX and Alpine?" (Retrieves your existing patterns).
2. **Code-Focused / Exact Match:** "Fix `AttributeError: 'NoneType' object has no attribute 'user'` in `views.py:45`." (Retrieves `views.py` and the function that seemingly returned `None`).
3. **Agentic RAG:** "This view is slow." (Agent retrieves the view, sees the ORM query, retrieves the Model index, checks for missing indices, and suggests `select_related`).

### Why this works for your specific Stack

Django errors are often "distance" errors. You get an error in the Template because the View didn't pass the context. A simple semantic search of the Template error won't find the View code. **Graph/Dependency RAG** solves this by linking the Template file to the View file in your index.
