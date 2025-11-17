import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from abc import ABC, abstractmethod
import asyncio

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from textwrap import dedent
import sys
import sqlite3
from concurrent.futures import ThreadPoolExecutor

# Load environment
load_dotenv()

class Chunk(BaseModel):
    """Enhanced chunk model matching your new format"""
    id: str
    type: str
    name: str
    qualified_name: str
    code: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    location: Dict[str, Any]
    dependencies: List[str]
    context: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    documentation: Dict[str, Any] = Field(default_factory=dict)
    analysis: Dict[str, Any] = Field(default_factory=dict)
    relationships: Dict[str, Any] = Field(default_factory=dict)
    references: List[str] = Field(default_factory=list)
    defines: List[str] = Field(default_factory=list)
    # Optional fields for backward compatibility
    signature: Optional[str] = None
    complexity: Optional[int] = None
    parent: Optional[str] = None
    docstring: Optional[str] = None

class SymbolIndex:
    """Enhanced symbol index for lookup with improved info extraction"""
    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self.data = data or {}

    def get_info(self, symbol: str) -> str:
        entries = self.data.get(symbol, [])
        if not entries:
            return "No additional info"

        # Take first entry (usually sufficient)
        entry = entries[0]
        kind = entry.get("kind", "unknown")
        file_path = entry.get("file", "unknown file")
        line = entry.get("line", "unknown line")

        # Make relative or just show basename
        short_file = os.path.basename(file_path)

        # Enhanced info based on symbol kind
        if kind == "class":
            return f"Class defined in {short_file}:{line}"
        elif kind == "function":
            return f"Function defined in {short_file}:{line}"
        elif kind == "method":
            return f"Method defined in {short_file}:{line}"
        else:
            return f"{kind} defined in {short_file}:{line}"


class ContextPromptBuilder(ABC):
    @abstractmethod
    def build_prompt(self, chunk: Chunk, symbol_index: Optional[SymbolIndex] = None) -> str:
        pass


class MultiLanguageContextPrompt(ContextPromptBuilder):
    """Enhanced prompt builder that handles multiple languages and chunk types"""

    def build_prompt(self, chunk: Chunk, symbol_index: Optional[SymbolIndex] = None) -> str:
        # Build dependency context
        deps_info = self._build_dependency_context(chunk, symbol_index)

        # Get domain-specific context
        domain_context = self._get_domain_specific_context(chunk)

        # Build language-specific instructions
        language_instructions = self._get_language_instructions(chunk.language)

        # Build chunk-type specific context
        chunk_type_context = self._get_chunk_type_context(chunk)

        prompt = dedent(f"""You are an expert software developer specializing in {chunk.language} and web development.
        Summarize the following code chunk in one clear, concise sentence.
        Focus on its purpose, behavior, and role in the application.

        {domain_context}

        File: {chunk.file_path}
        Type: {chunk.type}
        Name: {chunk.name}
        Qualified Name: {chunk.qualified_name}
        Location: Lines {chunk.location.get('start_line', '?')}-{chunk.location.get('end_line', '?')}

        {chunk_type_context}

        Dependencies: {', '.join(chunk.dependencies) or 'None'}
        {deps_info}

        {language_instructions}

        Code:
        ```{chunk.language}
        {chunk.code.strip()}
        ```

        Summary (one sentence, no markdown, no prefix):
        """).strip()
        return prompt

    def _build_dependency_context(self, chunk: Chunk, symbol_index: Optional[SymbolIndex]) -> str:
        """Build context about dependencies and relationships"""
        deps_info = []
        if symbol_index:
            for dep in chunk.dependencies[:5]:  # Limit to top 5 for brevity
                info = symbol_index.get_info(dep)
                if info != "No additional info":
                    deps_info.append(f"- {dep}: {info}")

        # Add relationship context
        relationship_context = []
        if chunk.relationships:
            imports = chunk.relationships.get("imports", [])
            if imports:
                relationship_context.append(f"Imports: {', '.join(imports[:3])}")

            called_functions = chunk.relationships.get("called_functions", [])
            if called_functions:
                relationship_context.append(f"Calls: {', '.join(called_functions[:3])}")

            references = chunk.relationships.get("references", [])
            if references:
                relationship_context.append(f"References: {', '.join(references[:3])}")

        context_parts = []
        if deps_info:
            context_parts.append("Dependency details:\n" + "\n".join(deps_info))
        if relationship_context:
            context_parts.append("Relationships:\n" + "; ".join(relationship_context))

        return "\n".join(context_parts) if context_parts else "No additional dependency or relationship info."

    def _get_domain_specific_context(self, chunk: Chunk) -> str:
        """Get domain-specific context based on file path and content"""
        domain_ctx = chunk.context.get("domain_context", "General code")
        module_ctx = chunk.context.get("module_context", "Unknown module")
        project_ctx = chunk.context.get("project_context", "Project codebase")

        # Enhanced domain context based on file path
        file_path = chunk.file_path.lower()
        if any(term in file_path for term in ['admin', 'modeladmin']):
            domain_ctx = "Django admin configuration and model management"
        elif any(term in file_path for term in ['model', 'schema']):
            domain_ctx = "Data models and database schema"
        elif any(term in file_path for term in ['view', 'controller']):
            domain_ctx = "Application logic and request handling"
        elif any(term in file_path for term in ['template', 'html']):
            domain_ctx = "User interface templates and presentation"
        elif any(term in file_path for term in ['static', 'css']):
            domain_ctx = "Styling and user interface design"
        elif any(term in file_path for term in ['static', 'js']):
            domain_ctx = "Client-side functionality and interactivity"

        return f"Module: {module_ctx}\nDomain: {domain_ctx}\nProject: {project_ctx}"

    def _get_language_instructions(self, language: str) -> str:
        """Get language-specific instructions for the LLM"""
        instructions = {
            'python': "Focus on Python-specific patterns, Django conventions if applicable, and the object's role in the application.",
            'javascript': "Focus on JavaScript patterns, DOM manipulation if applicable, and the function's role in client-side logic.",
            'html': "Focus on the template's structure, included components, and its role in the page layout and user interface.",
            'css': "Focus on the styling rules, layout impact, and visual design role in the application.",
            'java': "Focus on Java patterns, object-oriented design, and the class/method's role in the application architecture.",
            'cpp': "Focus on C++ patterns, memory management considerations, and performance characteristics.",
            'go': "Focus on Go patterns, concurrency if applicable, and the function's role in the system.",
            'rust': "Focus on Rust patterns, ownership system, and safety characteristics."
        }
        return instructions.get(language, "Focus on the code's purpose and behavior in the application.")

    def _get_chunk_type_context(self, chunk: Chunk) -> str:
        """Get context specific to the chunk type"""
        type_contexts = {
            'class': "This is a class definition. Describe its responsibility and how it might be used.",
            'function': "This is a function. Describe what it does and its input/output behavior.",
            'method': "This is a method within a class. Describe its specific role and how it modifies object state.",
            'html_file': "This is a complete HTML template. Describe its overall structure and purpose in the UI.",
            'html_element': "This is an HTML element or component. Describe its role in the page layout.",
            'css_rule': "This is a CSS rule. Describe its styling purpose and visual impact."
        }

        base_context = type_contexts.get(chunk.type, "Describe the purpose and behavior of this code.")

        # Add metadata-specific context
        metadata_context = []
        if chunk.metadata:
            if chunk.metadata.get('decorators'):
                metadata_context.append(f"Decorators: {', '.join(chunk.metadata['decorators'])}")
            if chunk.metadata.get('base_classes'):
                metadata_context.append(f"Inherits from: {', '.join(chunk.metadata['base_classes'])}")
            if chunk.metadata.get('is_abstract'):
                metadata_context.append("This is an abstract class/method.")
            if chunk.metadata.get('export_type') and chunk.metadata['export_type'] != 'none':
                metadata_context.append(f"Export type: {chunk.metadata['export_type']}")

        if metadata_context:
            return base_context + " " + "; ".join(metadata_context)
        return base_context


class LLMClient:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "ollama")
        self.model = os.getenv("LLM_MODEL", "ai/llama3.2:latest")
        self.base_url = os.getenv("LLM_BASE_URL", "http://localhost:12434")
        self.max_tokens = int(os.getenv("MAX_TOKENS", "256"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.3"))

        if self.provider == "ollama":
            self.llm = ChatOpenAI(
                model=self.model,
                base_url=self.base_url,
                temperature=self.temperature,
                openai_api_key="not-needed",
                max_tokens=self.max_tokens

            )
        else:
            raise NotImplementedError(f"Provider {self.provider} not supported")

    async def generate(self, prompt: str) -> str:
        chain = ChatPromptTemplate.from_messages([("user", "{input}")]) | self.llm | StrOutputParser()
        return await chain.abatch(prompt)
        # return chain.invoke({"input": prompt}).strip()
        # return chain.apply({"input": prompt}).strip()


def update_summary(record_id: str, summary: str):
    db_path = Path("enhanced_chunks.db")
    table_name = "enhanced_code_chunks"
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            f"UPDATE {table_name} SET summary = ? WHERE id = ?",
            (summary, record_id)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ Failed to update {record_id}: {e}")

class ContextEnricher:
    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        symbol_index: Optional[Dict[str, Any]] = None,
        prompt_builder: ContextPromptBuilder = MultiLanguageContextPrompt(),
        llm_client: LLMClient = LLMClient()
    ):
        # Convert dict chunks to Chunk objects, handling both old and new formats
        self.chunks = self._normalize_chunks(chunks)
        self.symbol_index = SymbolIndex(symbol_index) if symbol_index else None
        self.prompt_builder = prompt_builder
        self.llm_client = llm_client
        self.executor = ThreadPoolExecutor(max_workers=4)


    def _normalize_chunks(self, chunks: List[Dict[str, Any]]) -> List[Chunk]:
        """Normalize chunks to handle both old and new formats"""
        normalized = []
        for chunk_data in chunks:
            # Ensure context is a dict (handle string context from old format)
            if isinstance(chunk_data.get('context'), str):
                chunk_data['context'] = {'summary': chunk_data['context']}

            # Ensure all required fields have defaults
            chunk_data.setdefault('location', {})
            chunk_data.setdefault('metadata', {})
            chunk_data.setdefault('documentation', {})
            chunk_data.setdefault('analysis', {})
            chunk_data.setdefault('relationships', {})
            chunk_data.setdefault('references', [])
            chunk_data.setdefault('defines', [])

            # Handle qualified_name fallback
            if 'qualified_name' not in chunk_data:
                chunk_data['qualified_name'] = chunk_data.get('name', '')

            try:
                normalized.append(Chunk(**chunk_data))
            except Exception as e:
                print(f"⚠️ Failed to normalize chunk {chunk_data.get('id', 'unknown')}: {e}")
                continue

        return normalized

    # def enrich(self) -> List[Dict[str, Any]]:
    #     enriched = []
    #     total_chunks = len(self.chunks)

    #     for i, chunk in enumerate(self.chunks, 1):
    #         # if i >= 10:
    #         #   break

    #         print(f"[{i}/{total_chunks}] Generating context for {chunk.type} '{chunk.name}' (ID: {chunk.id})...")

    #         prompt = self.prompt_builder.build_prompt(chunk, self.symbol_index)
    #         try:
    #             summary = self.llm_client.generate(prompt)
    #         except Exception as e:
    #             print(f"⚠️ Failed for {chunk.id}: {e}")
    #             summary = "Context generation failed."

    #         # Add summary to context
    #         chunk_dict = chunk.model_dump()
    #         chunk_dict["context"]["summary"] = summary
    #         self.executor.submit(
    #             update_summary,
    #             chunk.id,
    #             summary
    #         )

    #         # Preserve existing context while adding summary
    #         enriched.append(chunk_dict)
    #         # print(summary)
    #         # sys.exit()

    #     return enriched

    async def enrich(self, batch_size: int = 4) -> List[Dict[str, Any]]:
        enriched = []
        total_chunks = len(self.chunks)

        # Temporary storage for batching
        batch_prompts = []
        batch_chunks = []

        for i, chunk in enumerate(self.chunks, 1):

            print(f"[{i}/{total_chunks}] Preparing context for {chunk.type} '{chunk.name}' (ID: {chunk.id})...")

            # Build prompt
            prompt = self.prompt_builder.build_prompt(chunk, self.symbol_index)

            # Add to batch buffers
            batch_prompts.append([prompt])
            batch_chunks.append(chunk)

            # If batch is full → send to model
            if len(batch_prompts) == batch_size:
                # print(batch_prompts)
                summaries = await self._generate_batch(batch_prompts)

                # Assign summaries back
                for chunk_obj, summary in zip(batch_chunks, summaries):
                    # asyncio.create_task(
                    #     asyncio.to_thread(update_summary, chunk_obj.id, summary)
                    # )
                    enriched.append(
                        self._process_chunk(chunk_obj, summary)
                    )
                # sys.exit()
                # Reset batch
                batch_prompts = []
                batch_chunks = []

        # Handle last incomplete batch
        if batch_prompts:
            summaries = await self._generate_batch(batch_prompts)
            for chunk_obj, summary in zip(batch_chunks, summaries):
                enriched.append(
                    self._process_chunk(chunk_obj, summary)
                )

        print("✅ Finished enrichment with batching.")
        return enriched


    async def _generate_batch(self, prompts: List[str]) -> List[str]:
        """
        Calls the model runner once with a list of prompts.
        Expects llm_client.generate() to accept list of prompts and return list of summaries.
        If your model runner only accepts single prompt, you can implement wrapper logic here.
        """
        try:
            summaries = await self.llm_client.generate(prompts)  # <- batch call
            print(len(summaries))
            return summaries
        except Exception as e:
            print(f"⚠️ Batch generation failed: {e}")
            return ["Context generation failed." for _ in prompts]


    def _process_chunk(self, chunk, summary: str) -> Dict[str, Any]:
        """
        Updates chunk dict, fires async db update
        """
        chunk_dict = chunk.model_dump()
        chunk_dict["context"]["summary"] = summary

        # Async DB update via thread pool
        self.executor.submit(update_summary, chunk.id, summary)

        return chunk_dict



def get_summarized_chunks_ids() -> List[Any]:
    """
    Connects to a SQLite database and returns all chunk ids from table
    where summary is not NULL.

    Returns:
        List[Any]: List of values from the first column (column index 1).
    """
    db_path = Path("enhanced_chunks.db")
    table_name = "enhanced_code_chunks"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    col1 = "id"
    col4 = "summary"

    query = f"""
        SELECT {col1}
        FROM {table_name}
        WHERE {col4} IS NOT NULL AND {col4} != ''
    """

    cur.execute(query)
    results = [row[0] for row in cur.fetchall()]

    conn.close()
    return results

def stats_check(setl, chunkl, nchunkl):
    print(f"Already summarized chunks =:= {setl}")
    print(f"Total chunks count =:= {chunkl}")
    print(f"To Be summarized chunks =:= {nchunkl}")


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Enrich code chunks with AI-generated context.")
    parser.add_argument("--input", required=True, help="Input JSON file with chunks")
    parser.add_argument("--output", required=True, help="Output enriched JSON file")
    parser.add_argument("--symbol-index", help="Optional symbol index JSON file")
    parser.add_argument("--model", default="ai/llama3.2:latest", help="LLM model to use (default: llama3.2:lates)")

    args = parser.parse_args()

    # Load chunks
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both direct chunk arrays and wrapped formats
    if "chunks" in data:
        chunks = data["chunks"]
    else:
        chunks = data  # Assume it's directly the chunks array
    summarized_ids = set(get_summarized_chunks_ids())
    filtered_chunks = [item for item in chunks if item.get("id") not in summarized_ids]
    stats_check(len(summarized_ids), len(chunks), len(filtered_chunks))

    # Load symbol index (optional)
    symbol_index = None
    if args.symbol_index:
        with open(args.symbol_index, "r", encoding="utf-8") as f:
            symbol_index = json.load(f)

    # Set model if provided
    if args.model:
        os.environ["LLM_MODEL"] = args.model

    # Enrich
    enricher = ContextEnricher(chunks=filtered_chunks, symbol_index=symbol_index)
    enriched_chunks = await enricher.enrich()

    # Save with same structure as input
    output_data = {
        "project_path": data.get("project_path", ""),
        "total_chunks": len(enriched_chunks),
        "statistics": data.get("statistics", {}),
        "chunks": enriched_chunks
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Enriched {len(enriched_chunks)} chunks. Saved to {args.output}")


if __name__ == "__main__":
    # main()
    asyncio.run(main())
    # client = LLMClient()

    # test_prompt = "Explain what a neural network is in one short paragraph."
    # try:
    #     print(f"\n🧠 Sending prompt:\n{test_prompt}\n")
    #     response = client.generate(test_prompt)
    #     print("✅ Response received:\n")
    #     print(response)
    # except Exception as e:
    #     print(f"❌ Test failed: {e}")