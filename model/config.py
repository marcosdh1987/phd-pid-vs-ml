"""
Configuration file for LLM models supported by the ModelFactory.

This module contains the mapping configuration between simplified model aliases
and their corresponding official model names and providers. It serves as a
central configuration point for all supported language models.

The MODEL_CONFIGS dictionary structure:
    - Key: The alias or shorthand name for the model (str)
    - Value: A dictionary containing:
        - provider: The service provider (e.g., "openai", "bedrock")
        - model_id: The official model identifier used by the provider

Example:
    To get the official model name for "gpt4":
    >>> MODEL_CONFIGS["gpt4"]["model_id"]
    'gpt-4'

    To check the provider:
    >>> MODEL_CONFIGS["claude-3-sonnet"]["provider"]
    'bedrock'
"""

import os

# Ollama host configuration - can be localhost or remote server
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")

# Comprehensive configuration dictionary for all supported language models.
#
# This configuration supports multiple AI providers and handles different
# authentication and deployment patterns:
#
# Provider Types:
#     1. OpenAI: Cloud-based models requiring API key authentication
#     2. Bedrock: AWS-managed models with IAM-based authentication
#     3. Ollama: Self-hosted open-source models
#
# Configuration Fields:
#     - provider: The AI service provider
#     - model_id: Official model identifier for the provider
#     - inference_profile_id: (Bedrock only) Optimized routing identifier
#     - endpoint: (Ollama only) Server endpoint URL
#
# Authentication Requirements:
#     - OpenAI: OPENAI_API_KEY environment variable
#     - Bedrock: AWS credentials (IAM role, access keys, or profile)
#     - Ollama: Network access to Ollama server (no auth by default)

MODEL_CONFIGS = {
    # === OpenAI Models ===
    # These models use the OpenAI API and require an OpenAI API key
    # Latest GPT-4 variants with improved capabilities
    "gpt5-mini": {
        "provider": "openai",
        "model_id": "gpt-5-mini",
    },
    "gpt5-nano": {
        "provider": "openai",
        "model_id": "gpt-5-nano",
    },
    "gpt4o": {
        "provider": "openai",
        "model_id": "gpt-4o-2024-08-06",
    },  # Optimized for speed and cost
    "gpt4omini": {
        "provider": "openai",
        "model_id": "gpt-4o-mini",
    },  # Smaller, faster GPT-4
    "gpt4": {"provider": "openai", "model_id": "gpt-4"},  # Standard GPT-4
    "gpt35": {
        "provider": "openai",
        "model_id": "gpt-3.5-turbo-0613",
    },  # Legacy but fast
    # Alternative naming conventions for consistency
    "gpt-4": {"provider": "openai", "model_id": "gpt-4"},
    "gpt-3.5-turbo-0613": {"provider": "openai", "model_id": "gpt-3.5-turbo-0613"},
    # === Amazon Bedrock Models with Inference Profiles ===
    # Inference profiles provide optimized routing and cost management
    # Claude 3.5 Sonnet - Latest and most capable Claude model
    "claude-3-5-sonnet-v2": {
        "provider": "bedrock",
        "model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "inference_profile_id": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    },
    "claude-3-5-sonnet": {
        "provider": "bedrock",
        "model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "inference_profile_id": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    },
    # Claude 3 Sonnet - Balanced performance and speed
    "claude-3-sonnet": {
        "provider": "bedrock",
        "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
        "inference_profile_id": "us.anthropic.claude-3-sonnet-20240229-v1:0",
    },
    # Amazon Nova Pro - Amazon's newest multimodal model
    "nova-pro": {
        "provider": "bedrock",
        "model_id": "amazon.nova-pro-v1:0",
        "inference_profile_id": "us.amazon.nova-pro-v1:0",
    },
    # === Amazon Bedrock Models without Inference Profiles ===
    # Direct model access for specialized use cases
    "claude-3-haiku": {
        "provider": "bedrock",
        "model_id": "anthropic.claude-3-haiku-20240307-v1:0",  # Fastest Claude model
    },
    "claude-2": {
        "provider": "bedrock",
        "model_id": "anthropic.claude-v2:1",
    },  # Legacy Claude
    # === Ollama Models (Self-hosted Open Source) ===
    # These models run on local or remote Ollama servers
    # Meta's Llama models - General purpose open source LLMs
    "llama2": {
        "provider": "ollama",
        "model_id": "llama2",
        "endpoint": f"http://{OLLAMA_HOST}:11434/api/generate",
    },
    "llama31": {
        "provider": "ollama",
        "model_id": "llama3.1",  # Latest Llama with improved capabilities
        "endpoint": f"http://{OLLAMA_HOST}:11434/api/generate",
    },
    # NVIDIA Nemotron - Optimized for instruction following
    "nemotron": {
        "provider": "ollama",
        "model_id": "nemotron",
        "endpoint": f"http://{OLLAMA_HOST}:11434/api/generate",
    },
    # DeepSeek R1 - Advanced reasoning model
    "deepseek-r1-32b": {
        "provider": "ollama",
        "model_id": "deepseek-r1:32b",
        "endpoint": f"http://{OLLAMA_HOST}:11434/api/generate",
    },
    # Mistral - Efficient French/multilingual model
    "mistral": {
        "provider": "ollama",
        "model_id": "mistral",
        "endpoint": f"http://{OLLAMA_HOST}:11434/api/generate",
    },
    # IBM Granite - Enterprise-focused model
    "granite3.2": {
        "provider": "ollama",
        "model_id": "granite3.2",
        "endpoint": f"http://{OLLAMA_HOST}:11434/api/generate",
    },
    # Gemma
    "gemma3:27b": {
        "provider": "ollama",
        "model_id": "gemma3:27b",
        "endpoint": f"http://{OLLAMA_HOST}:11434/api/generate",
    },
    # Google Gemini models
    # These models use Google's Generative AI API and require a GOOGLE_API_KEY
    "gemini-2.0-flash": {"provider": "google", "model_id": "gemini-2.0-flash"},
    "gemini-2.5-flash": {
        "provider": "google",
        "model_id": "gemini-2.5-flash-preview-04-17",
    },
    "gemini-2.5-pro": {
        "provider": "google",
        "model_id": "gemini-2.5-pro-preview-03-25",
    },
    "gemini-1.5-flash": {"provider": "google", "model_id": "gemini-1.5-flash"},
    "gemini-1.5-pro": {"provider": "google", "model_id": "gemini-1.5-pro"},
}

# Configuration dictionary for text embedding models.
#
# This dictionary maps friendly embedding model names to their provider-specific
# configurations. It supports both OpenAI and Amazon Bedrock embedding models,
# each with different capabilities and use cases.
#
# Structure:
#     - Key: Friendly model name (str)
#     - Value: Dictionary with provider and model_id
#
# OpenAI Models:
#     - text-embedding-3-small: Latest, efficient model for most use cases
#     - text-embedding-3-large: Larger model with higher accuracy
#     - text-embedding-ada-002: Legacy model, still widely supported
#
# Bedrock Models:
#     - amazon-titan-embed: Amazon's proprietary embedding model
#     - cohere-embed: Cohere's English-optimized embedding model
#
# Example Usage:
#     >>> config = EMBEDDING_CONFIGS["text-embedding-3-small"]
#     >>> print(config["provider"])  # "openai"
#     >>> print(config["model_id"])  # "text-embedding-3-small"

EMBEDDING_CONFIGS = {
    # OpenAI embeddings - Require OPENAI_API_KEY environment variable
    "text-embedding-3-small": {
        "provider": "openai",
        "model_id": "text-embedding-3-small",
    },
    "text-embedding-3-large": {
        "provider": "openai",
        "model_id": "text-embedding-3-large",
    },
    "text-embedding-ada-002": {
        "provider": "openai",
        "model_id": "text-embedding-ada-002",
    },
    # Bedrock embeddings - Require AWS credentials and proper IAM permissions
    "amazon-titan-embed": {
        "provider": "bedrock",
        "model_id": "amazon.titan-embed-text-v1",
    },
    "cohere-embed": {
        "provider": "bedrock",
        "model_id": "cohere.embed-english-v3",
    },
    # Google embeddings
    "google-embedding": {
        "provider": "google",
        "model_id": "text-embedding-004",
    },
}

# Provider display names for UI
PROVIDER_DISPLAY_NAMES = {
    "openai": "OpenAI",
    "bedrock": "AWS Bedrock",
    "ollama": "Ollama",
    "google": "Google",
}

# Model descriptions for API responses
MODEL_DESCRIPTIONS = {
    # OpenAI models
    "gpt-4o-2024-08-06": "Latest GPT-4 optimized for speed and cost",
    "gpt-4o-mini": "Fast, efficient OpenAI model",
    "gpt-4": "Full GPT-4 with enhanced reasoning",
    "gpt-3.5-turbo-0613": "Legacy but fast GPT model",
    # Bedrock models
    "anthropic.claude-3-5-sonnet-20241022-v2:0": "Latest Claude 3.5 Sonnet with enhanced capabilities",
    "anthropic.claude-3-sonnet-20240229-v1:0": "Anthropic's balanced model",
    "anthropic.claude-3-haiku-20240307-v1:0": "Fastest Claude model",
    "amazon.nova-pro-v1:0": "Amazon's newest multimodal model",
    "anthropic.claude-v2:1": "Legacy Claude model",
    # Ollama models
    "llama3.1": "Meta's latest open source model",
    "llama2": "Meta's general purpose open source model",
    "nemotron": "NVIDIA's instruction-following model",
    "deepseek-r1:32b": "Advanced reasoning model",
    "mistral": "Efficient multilingual model",
    "granite3.2": "IBM's enterprise-focused model",
    "gemma3:27b": "Google's efficient open model",
    # Google models
    "gemini-2.0-flash": "Google's latest fast model",
    "gemini-2.5-flash-preview-04-17": "Google's preview flash model",
    "gemini-2.5-pro-preview-03-25": "Google's preview pro model",
    "gemini-1.5-flash": "Google's efficient model",
    "gemini-1.5-pro": "Google's advanced model",
}

# Recommended models for API responses
RECOMMENDED_MODELS = {"gpt4omini", "claude-3-5-sonnet", "llama31"}


def get_provider_display_name(provider: str) -> str:
    """Get human-readable provider name."""
    return PROVIDER_DISPLAY_NAMES.get(provider, provider.title())


def get_model_description(model_id: str, provider: str) -> str:
    """Generate model description based on configuration."""
    return MODEL_DESCRIPTIONS.get(model_id, f"{provider.title()} model")


def is_recommended_model(alias: str) -> bool:
    """Determine if a model should be marked as recommended."""
    return alias in RECOMMENDED_MODELS


def get_model_info(alias: str, config: dict) -> dict:
    """
    Get complete model information for API responses.

    Args:
        alias: Model alias/key from MODEL_CONFIGS
        config: Model configuration dict

    Returns:
        dict: Complete model information for API response
    """
    provider = config["provider"]
    model_id = config["model_id"]

    model_info = {
        "id": alias,
        "name": alias.replace("-", " ").replace("_", " ").title(),
        "provider": get_provider_display_name(provider),
        "description": get_model_description(model_id, provider),
        "recommended": is_recommended_model(alias),
    }

    # Add additional info for specific providers
    if provider == "bedrock" and "inference_profile_id" in config:
        model_info["inference_profile"] = True
    elif provider == "ollama" and "endpoint" in config:
        model_info["self_hosted"] = True

    return model_info
