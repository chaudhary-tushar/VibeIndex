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

from src.config import LLMConfig

# Load environment
load_dotenv()


class LLMClient:
    """Unified LLM client supporting multiple providers"""

    def __init__(self, llm_config: LLMConfig = None):
        # Use provided config or create a default one
        if llm_config is None:
            llm_config = LLMConfig()

        # Extract model information from the URL to determine provider
        model_url = llm_config.model_url
        model_name = llm_config.model_name
        self.provider = llm_config.model_provider
        self.model = model_name
        self.base_url = model_url
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
                openai_api_key="not needed",
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

        chain = ChatPromptTemplate.from_messages([("user", "{input}")]) | self.llm | StrOutputParser()
        results = []
        try:
            results = await chain.abatch(prompts)
        except httpx.TimeoutException as e:  # Catch timeout specifically
            print(f"⚠️ Batch generation failed: {e}")
            return ["Context generation failed." for _ in prompts]
        except httpx.RequestError as e:  # Catch other network errors
            print(f"⚠️ Batch generation failed: {e}")
            return ["Context generation failed." for _ in prompts]
        except Exception as e:
            print(f"⚠️ Batch generation failed: {e}")
            return ["Context generation failed." for _ in prompts]
        return results

    def generate(self, prompt: str) -> str:
        """Generate response for a single prompt (synchronous)"""
        chain = ChatPromptTemplate.from_messages([("user", "{input}")]) | self.llm | StrOutputParser()
        return chain.invoke({"input": prompt}).strip()
