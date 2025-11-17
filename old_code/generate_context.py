import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from abc import ABC, abstractmethod
import sys

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from textwrap import dedent

# Load environment
load_dotenv()

class Chunk(BaseModel):
    id: str
    type: str
    name: str
    code: str
    file_path: str
    language: str
    dependencies: List[str]
    context: Dict[str, Any]

class SymbolIndex:
    """Optional symbol index for lookup (e.g., type definitions, doc links)"""
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
      # Make relative or just show basename
      short_file = os.path.basename(file_path)
      return f"{kind} defined in {short_file}"


class ContextPromptBuilder(ABC):
    @abstractmethod
    def build_prompt(self, chunk: Chunk, symbol_index: Optional[SymbolIndex] = None) -> str:
        pass


class DjangoCodeContextPrompt(ContextPromptBuilder):
    def build_prompt(self, chunk: Chunk, symbol_index: Optional[SymbolIndex] = None) -> str:
      deps_info = []
      if symbol_index:
          for dep in chunk.dependencies:
              info = symbol_index.get_info(dep)
              if info != "No additional info":
                  deps_info.append(f"- {dep}: {info}")

      deps_context = "\n".join(deps_info) if deps_info else "No additional dependency info."

      domain_ctx = chunk.context.get("domain_context", "General code")
      module_ctx = chunk.context.get("module_context", "Unknown module")

      prompt = dedent(f"""You are an expert Python and Django developer.
        Summarize the following code chunk in one clear, concise sentence.
        Focus on its purpose, behavior, and role in a Django web application.

        Module: {module_ctx}
        Domain context: {domain_ctx}
        File: {chunk.file_path}
        Type: {chunk.type}
        Name: {chunk.name}
        Dependencies: {', '.join(chunk.dependencies) or 'None'}

        {deps_context}

        Code:
        ```{chunk.language}
        {chunk.code.strip()}```
        Summary (one sentence, no markdown, no prefix):
        """).strip()
      return prompt

class LLMClient:
  def __init__(self):
    self.provider = os.getenv("LLM_PROVIDER", "ollama")
    self.model = os.getenv("LLM_MODEL", "phi3")
    self.base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434 ")
    self.max_tokens = int(os.getenv("MAX_TOKENS", "256"))
    self.temperature = float(os.getenv("TEMPERATURE", "0.3"))
    print(self.max_tokens)
    sys.exit()

    if self.provider == "ollama":
      self.llm = ChatOllama(
        model=self.model,
        base_url=self.base_url,
        temperature=self.temperature,
        num_predict=self.max_tokens
        )
    else:
      raise NotImplementedError(f"Provider {self.provider} not supported")

  def generate(self, prompt: str) -> str:
    chain = ChatPromptTemplate.from_messages([("user", "{input}")]) | self.llm | StrOutputParser()
    return chain.invoke({"input": prompt}).strip()

class ContextEnricher:
  def __init__(
    self,
    chunks: List[Dict[str, Any]],
    symbol_index: Optional[Dict[str, Any]] = None,
    prompt_builder: ContextPromptBuilder = DjangoCodeContextPrompt(),
    llm_client: LLMClient = LLMClient()
    ):
    self.chunks = [Chunk(**ch) for ch in chunks]
    self.symbol_index = SymbolIndex(symbol_index) if symbol_index else None
    self.prompt_builder = prompt_builder
    self.llm_client = llm_client

  def enrich(self) -> List[Dict[str, Any]]:
    enriched = []
    for chunk in self.chunks:
      print(f"Generating context for {chunk.type} '{chunk.name}' (ID: {chunk.id})...")
      prompt = self.prompt_builder.build_prompt(chunk, self.symbol_index)
      try:
        summary = self.llm_client.generate(prompt)
      except Exception as e:
        print(f"⚠️ Failed for {chunk.id}: {e}")
        summary = "Context generation failed."

      # Add summary under context.summary
      chunk_dict = chunk.model_dump()
      chunk_dict["context"]["summary"] = summary
      enriched.append(chunk_dict)
    return enriched

def main():
  import argparse

  parser = argparse.ArgumentParser(description="Enrich code chunks with AI-generated context.")
  parser.add_argument("--input", required=True, help="Input JSON file with chunks")
  parser.add_argument("--output", required=True, help="Output enriched JSON file")
  parser.add_argument("--symbol-index", help="Optional symbol index JSON file")

  args = parser.parse_args()

  # Load chunks
  with open(args.input, "r", encoding="utf-8") as f:
    chunks = json.load(f)

  # Load symbol index (optional)
  symbol_index = None
  if args.symbol_index:
    with open(args.symbol_index, "r", encoding="utf-8") as f:
      symbol_index = json.load(f)

  # Enrich
  enricher = ContextEnricher(chunks=chunks["chunks"], symbol_index=symbol_index)
  enriched_chunks = enricher.enrich()

  # Save
  Path(args.output).parent.mkdir(parents=True, exist_ok=True)
  with open(args.output, "w", encoding="utf-8") as f:
    json.dump(enriched_chunks, f, indent=2, ensure_ascii=False)

  print(f"✅ Enriched {len(enriched_chunks)} chunks. Saved to {args.output}")


if __name__ == "__main__":
    # main()
    llm = LLMClient()
    print(llm.provider, llm.base_url, llm.max_tokens)