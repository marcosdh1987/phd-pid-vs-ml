"""
Utilities Package - Core Infrastructure and Helper Functions.

This package provides essential utility functions and infrastructure components
that support the RAG agent's operation. These utilities handle cross-cutting
concerns like AWS integration, configuration management, and prompt loading.

Modules:
    boto_session: AWS SDK session management and authentication
    prompt_loader: Dynamic prompt loading with fallback mechanisms

Key Features:
    - Environment-aware AWS authentication (local vs production)
    - Secure credential management through AWS services
    - Dynamic prompt loading with intelligent fallbacks
    - Comprehensive error handling and logging
    - Multi-environment deployment support

AWS Integration:
    The utils package provides robust AWS integration for:
    - Amazon Bedrock model access
    - SSM Parameter Store for configuration
    - Secrets Manager for sensitive data
    - IAM-based authentication and authorization

Example Usage:
    ```python
    from agent_rag.utils.boto_session import get_boto3_client, get_ssm_parameter
    from agent_rag.utils.prompt_loader import load_prompt

    # Create AWS service clients
    bedrock = get_boto3_client("bedrock-runtime", ENV="production")

    # Load configuration from Parameter Store
    model_name = get_ssm_parameter("/myapp/config/default_model")

    # Load appropriate prompts
    research_prompt = load_prompt("research", "gpt4", "agent_rag")
    ```

Configuration Philosophy:
    - Environment variables for deployment configuration
    - AWS Parameter Store for application configuration
    - AWS Secrets Manager for sensitive data
    - Fallback mechanisms for robustness
    - Centralized logging for observability
"""

from .boto_session import (
    get_boto3_client,
    get_boto3_session,
    get_secret,
    get_ssm_parameter,
)
from .prompt_loader import load_prompt

__all__ = [
    "get_boto3_session",
    "get_boto3_client",
    "get_ssm_parameter",
    "get_secret",
    "load_prompt",
]
