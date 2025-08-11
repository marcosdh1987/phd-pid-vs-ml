"""
Text Embedding Models Factory and Wrapper.

This module provides a unified interface for working with different text embedding
providers through the Embedder class. It supports both OpenAI and Amazon Bedrock
embedding models, handling authentication, configuration, and provider-specific
initialization automatically.

Key Features:
    - Multi-provider support (OpenAI, Amazon Bedrock)
    - Automatic configuration management
    - Consistent API across different providers
    - Environment-based credential management
    - Batch embedding support

Classes:
    Embedder: Main class for creating and using embedding models

Supported Providers:
    - OpenAI: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
    - Amazon Bedrock: amazon-titan-embed, cohere-embed

Example:
    ```python
    # Create embedder with OpenAI model
    embedder = Embedder(model_name="text-embedding-3-small")

    # Generate single embedding
    embedding = embedder.embed_query("Hello, world!")

    # Generate batch embeddings
    docs = ["Document 1", "Document 2", "Document 3"]
    embeddings = embedder.embed_documents(docs)
    ```
"""

import logging
import os

import boto3
from dotenv import load_dotenv
from langchain_community.embeddings import BedrockEmbeddings, OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from model.config import EMBEDDING_CONFIGS

logging.basicConfig(
    format="%(asctime)s - %(module)s - %(message)s",
    level=logging.INFO,
)

load_dotenv(override=True)


class Embedder:
    """
    A class that provides text embedding functionality using different providers.

    This class supports multiple embedding providers including OpenAI, Amazon Bedrock,
    and Google Gemini. It automatically handles the configuration and initialization
    of the appropriate embedding model based on the specified model name.

    Available Models:
        OpenAI:
            - text-embedding-3-small
            - text-embedding-3-large
            - text-embedding-ada-002

        Amazon Bedrock:
            - amazon-titan-embed
            - cohere-embed

        Google Gemini:
            - google-embedding (text-embedding-004)

    Usage Examples:
        # Using OpenAI embeddings
        >>> embedder = Embedder(model_name="text-embedding-3-small")
        >>> text = "Hello, world!"
        >>> embedding = embedder.embb.embed_query(text)

        # Using Bedrock embeddings
        >>> embedder = Embedder(
        ...     model_name="amazon-titan-embed",
        ...     region_name="us-east-1"
        ... )
        >>> embedding = embedder.embb.embed_query(text)

        # Using Google embeddings
        >>> embedder = Embedder(model_name="google-embedding")
        >>> embedding = embedder.embb.embed_query(text)

    Requirements:
        - For OpenAI models:
            - OPENAI_API_KEY environment variable must be set
        - For Bedrock models:
            - Valid AWS credentials configured
            - Appropriate permissions to access Bedrock service
        - For Google models:
            - GOOGLE_API_KEY environment variable must be set
    """

    def __init__(self, model_name="text-embedding-3-small", region_name="us-east-1"):
        """
        Initializes the embedder with the specified model.

        Args:
            model_name (str): Name of the embedding model to use.
                            See class docstring for available models.
            region_name (str): AWS region for Bedrock models.
                             Only required for Bedrock models.

        Raises:
            ValueError: If the specified model name is not supported
            ValueError: If the provider is not supported
            Exception: If required credentials are not properly configured
        """
        self.model_name = model_name
        self.region_name = region_name
        self.config = EMBEDDING_CONFIGS.get(model_name)
        if not self.config:
            raise ValueError(f"Unsupported embedding model: {model_name}")

        self.embb = self._build_embb()

    def _build_embb(self):
        """
        Builds the appropriate embedding model based on the configuration.

        Returns:
            The embedding model instance (OpenAIEmbeddings, BedrockEmbeddings, or GoogleGenerativeAIEmbeddings).

        Raises:
            ValueError: If the provider specified in config is not supported
        """
        if self.config["provider"] == "openai":
            return self._build_openai_embb()
        elif self.config["provider"] == "bedrock":
            return self._build_bedrock_embb()
        elif self.config["provider"] == "google":
            return self._build_google_embb()
        else:
            raise ValueError(f"Unsupported provider: {self.config['provider']}")

    def _build_openai_embb(self):
        """
        Builds the OpenAI embeddings model.

        Returns:
            OpenAIEmbeddings: The OpenAI embeddings model instance.

        Raises:
            Exception: If OPENAI_API_KEY is not set in environment variables
        """
        if not os.getenv("OPENAI_API_KEY"):
            raise Exception("OPENAI_API_KEY environment variable is not set")

        return OpenAIEmbeddings(
            model=self.config["model_id"], openai_api_key=os.getenv("OPENAI_API_KEY")
        )

    def _build_bedrock_embb(self):
        """
        Builds the Bedrock embeddings model.

        Returns:
            BedrockEmbeddings: The Bedrock embeddings model instance.

        Raises:
            Exception: If AWS credentials are not properly configured
        """
        try:
            bedrock_client = boto3.client(
                service_name="bedrock-runtime",
                region_name=self.region_name,
            )
        except Exception as e:
            raise Exception(f"Failed to initialize Bedrock client: {str(e)}")

        return BedrockEmbeddings(
            client=bedrock_client, model_id=self.config["model_id"]
        )

    def _build_google_embb(self):
        """
        Builds the Google Gemini embeddings model.

        Returns:
            GoogleGenerativeAIEmbeddings: The Google embeddings model instance.

        Raises:
            Exception: If GOOGLE_API_KEY is not set in environment variables
        """
        if not os.getenv("GOOGLE_API_KEY"):
            raise Exception("GOOGLE_API_KEY environment variable is not set")

        return GoogleGenerativeAIEmbeddings(
            model=self.config["model_id"], google_api_key=os.getenv("GOOGLE_API_KEY")
        )

    def embed_query(self, text: str) -> list[float]:
        """
        Generates embeddings for a given text.

        Args:
            text (str): The text to generate embeddings for.

        Returns:
            list[float]: The embedding vector.

        Example:
            >>> embedder = Embedder(model_name="text-embedding-3-small")
            >>> embeddings = embedder.embed_query("Hello, world!")
        """
        return self.embb.embed_query(text)

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """
        Generates embeddings for a list of documents.

        Args:
            documents (list[str]): List of texts to generate embeddings for.

        Returns:
            list[list[float]]: List of embedding vectors.

        Example:
            >>> embedder = Embedder(model_name="text-embedding-3-small")
            >>> docs = ["Hello, world!", "Another document"]
            >>> embeddings = embedder.embed_documents(docs)
        """
        return self.embb.embed_documents(documents)
