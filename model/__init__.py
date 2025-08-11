"""
Model Package - AI Model Abstractions and Factory Patterns.

This package provides a unified interface for working with different AI models
and providers. It abstracts away the complexity of different APIs, authentication
methods, and configuration patterns to provide a consistent experience.

Modules:
    llm: Language model factory and custom implementations
    embeddings: Text embedding models for vector representations
    config: Configuration mappings for all supported models

Key Features:
    - Multi-provider support (OpenAI, Amazon Bedrock, Ollama)
    - Consistent API across different model types
    - Environment-based configuration management
    - Custom tool calling implementations
    - Automatic authentication handling

Example Usage:
    ```python
    from agent_rag.model.llm import ModelFactory
    from agent_rag.model.embeddings import Embedder

    # Create a language model
    factory = ModelFactory(model_name="gpt-4", temperature=0.3)
    llm = factory.create_model()

    # Create an embedding model
    embedder = Embedder(model_name="text-embedding-3-small")
    vectors = embedder.embed_query("Hello, world!")
    ```

Supported Providers:
    - OpenAI: GPT models and text embeddings
    - Amazon Bedrock: Claude, Nova, Titan models and embeddings
    - Ollama: Self-hosted open source models (Llama, Mistral, etc.)
"""

from .config import EMBEDDING_CONFIGS, MODEL_CONFIGS
from .embeddings import Embedder

# Import main classes for easy access
from .llm import CustomOllamaLLM, ModelFactory

__all__ = [
    "ModelFactory",
    "CustomOllamaLLM",
    "Embedder",
    "MODEL_CONFIGS",
    "EMBEDDING_CONFIGS",
]
