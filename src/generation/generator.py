"""
LLM client and generation utilities
"""

import os
from typing import List, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

# Load environment
load_dotenv()


class LLMClient:
    """Unified LLM client supporting multiple providers"""
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "ollama")
        self.model = os.getenv("LLM_MODEL", "ai/llama3.2:latest")
        self.base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434")
        self.max_tokens = int(os.getenv("MAX_TOKENS", "256"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.3"))

        if self.provider == "ollama":
            self.llm = ChatOllama(
                model=self.model,
                base_url=self.base_url,
                temperature=self.temperature,
                num_predict=self.max_tokens
            )
        elif self.provider == "openai":
            self.llm = ChatOpenAI(
                model=self.model,
                base_url=self.base_url,
                temperature=self.temperature,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                max_tokens=self.max_tokens
            )
        else:
            raise NotImplementedError(f"Provider {self.provider} not supported")

    async def generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate responses for a batch of prompts"""
        if not prompts:
            return []
        try:
            chain = ChatPromptTemplate.from_messages([("user", "{input}")]) | self.llm | StrOutputParser()
            # For simplicity, process one by one (can be optimized with async batch)
            results = []
            for prompt in prompts:
                result = await chain.ainvoke({"input": prompt})
                results.append(result.strip())
            return results
        except Exception as e:
            print(f"⚠️ Batch generation failed: {e}")
            return ["Context generation failed." for _ in prompts]

    def generate(self, prompt: str) -> str:
        """Generate response for a single prompt (synchronous)"""
        chain = ChatPromptTemplate.from_messages([("user", "{input}")]) | self.llm | StrOutputParser()
        return chain.invoke({"input": prompt}).strip()