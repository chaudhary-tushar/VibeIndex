"""
LLM client and generation utilities
"""

import os

import httpx  # Added import for httpx
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

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
                model=self.model, base_url=self.base_url, temperature=self.temperature, num_predict=self.max_tokens
            )
        elif self.provider == "openai":
            self.llm = ChatOpenAI(
                model=self.model,
                base_url=self.base_url,
                temperature=self.temperature,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                max_tokens=self.max_tokens,
            )
        else:
            # Default to Ollama if provider is not supported
            print(f"Warning: Provider {self.provider} not supported, defaulting to Ollama")
            self.provider = "ollama"
            self.llm = ChatOllama(
                model=self.model, base_url=self.base_url, temperature=self.temperature, num_predict=self.max_tokens
            )

    async def generate_batch(self, prompts: list[str]) -> list[str]:
        """Generate responses for a batch of prompts"""
        if not prompts:
            return []

        results = []
        chain = ChatPromptTemplate.from_messages([("user", "{input}")]) | self.llm | StrOutputParser()

        for prompt in prompts:
            try:
                result = await chain.ainvoke({"input": prompt})
                results.append(result.strip())
            except httpx.TimeoutException as e:  # Catch timeout specifically
                print(f"⚠️ Timeout during generation for prompt '{prompt[:50]}...': {e}")
                results.append("Context generation failed due to timeout.")
            except httpx.RequestError as e:  # Catch other network errors
                print(f"⚠️ Network error during generation for prompt '{prompt[:50]}...': {e}")
                results.append("Context generation failed due to network error.")
            except Exception as e:  # noqa: BLE001
                # Catching generic Exception for graceful degradation and batch continuation.
                # This ensures that a single prompt failure does not halt the entire batch.
                print(f"⚠️ Unexpected error during generation for prompt '{prompt[:50]}...': {e}")
                results.append("Context generation failed.")
        return results

    def generate(self, prompt: str) -> str:
        """Generate response for a single prompt (synchronous)"""
        chain = ChatPromptTemplate.from_messages([("user", "{input}")]) | self.llm | StrOutputParser()
        return chain.invoke({"input": prompt}).strip()
