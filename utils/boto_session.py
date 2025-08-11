import json
import logging
import os

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load environment variables from .env if it exists
load_dotenv(override=True)

logger = logging.getLogger(__name__)


"""
AWS Boto3 Session Management and Configuration Utilities.

This module provides utilities for creating and managing AWS boto3 sessions across
different deployment environments. It handles authentication, credential management,
and AWS service client creation with environment-specific configurations.

Key Features:
    - Environment-aware session creation (local vs production)
    - Flexible authentication methods (profiles, explicit credentials, IAM roles)
    - AWS service client factory methods
    - SSM Parameter Store integration
    - AWS Secrets Manager integration
    - Comprehensive error handling and logging

Supported Environments:
    - local: Uses AWS profiles and credentials files for development
    - production: Uses explicit environment variables for deployment

Environment Variables:
    - ENV: Environment identifier ("local" or "production")
    - AWS_REGION: AWS region for all services (default: us-east-1)
    - AWS_PROFILE: AWS profile name for local development (default: "default")
    - AWS_ACCESS_KEY_ID: Access key ID for production authentication
    - AWS_SECRET_ACCESS_KEY: Secret access key for production authentication
    - AWS_SESSION_TOKEN: Optional session token for temporary credentials

Example Usage:
    ```python
    from agent_rag.utils.boto_session import get_boto3_client, get_ssm_parameter

    # Create a Bedrock client for production
    bedrock = get_boto3_client("bedrock-runtime", ENV="production")

    # Get configuration from SSM Parameter Store
    api_key = get_ssm_parameter("/myapp/openai/api_key", ENV="local")

    # Retrieve secrets from Secrets Manager
    db_password = get_secret("myapp/database", key_in_secret="password")
    ```

Security Considerations:
    - Never hardcode credentials in source code
    - Use IAM roles in production when possible
    - Rotate credentials regularly
    - Use least privilege principle for IAM permissions
    - Enable CloudTrail for audit logging
"""
import logging

from dotenv import load_dotenv

# Load environment variables from .env if it exists
load_dotenv(override=True)

logger = logging.getLogger(__name__)


def get_boto3_session(ENV="local"):
    """
    Create a boto3 session based on the deployment environment.

    This function creates an appropriately configured boto3 session for different
    deployment environments. It handles authentication through AWS profiles in
    local development and explicit credentials in production environments.

    Args:
        ENV (str): Environment identifier. Options:
                  - "local": Use AWS profiles for development (default)
                  - "production": Use explicit environment variable credentials

    Returns:
        boto3.Session: Configured boto3 session object ready for service client creation

    Raises:
        EnvironmentError: If required credentials are missing in production environment
        ValueError: If an unknown environment identifier is provided
        Exception: If session creation fails due to authentication or configuration issues

    Environment Variables (Production):
        - AWS_ACCESS_KEY_ID: Required AWS access key ID
        - AWS_SECRET_ACCESS_KEY: Required AWS secret access key
        - AWS_SESSION_TOKEN: Optional temporary session token
        - AWS_REGION: AWS region (default: us-east-1)

    Environment Variables (Local):
        - AWS_PROFILE: AWS profile name (default: "default")
        - AWS_REGION: AWS region (default: us-east-1)

    Example:
        ```python
        # Local development with default profile
        session = get_boto3_session(ENV="local")

        # Production with explicit credentials
        os.environ["AWS_ACCESS_KEY_ID"] = "your_key"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "your_secret"
        session = get_boto3_session(ENV="production")
        ```
    """
    ENV = os.getenv("ENV", "production")
    region = os.getenv("AWS_REGION", "us-east-1")  # Use consistent default
    logger.info(f"Creating boto3 session for environment: {ENV} in region: {region}")
    try:
        if ENV == "local":
            # Use default profile for local environment
            logger.info(
                f"Using profile {os.getenv('AWS_PROFILE', 'default')} for local environment"
            )
            return boto3.Session(
                profile_name=os.getenv("AWS_PROFILE", "default"), region_name=region
            )
        elif ENV == "production":
            # Use explicit credentials for production
            logger.info("Using explicit credentials for production environment")
            access_key = os.getenv("AWS_ACCESS_KEY_ID")
            secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            session_token = os.getenv("AWS_SESSION_TOKEN", None)

            # Validate that the required credentials are set
            if not access_key or not secret_key:
                raise OSError(
                    "Missing AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY in production"
                )

            # Create session explicitly with credentials
            session = boto3.Session(
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                aws_session_token=session_token,
                region_name=region,
            )
            logger.info("Successfully created boto3 session for production")
            return session
        else:
            raise ValueError(f"Unknown environment: {ENV}")
    except Exception as e:
        logger.error(f"Error creating boto3 session: {e}")
        raise


def get_boto3_client(service_name, ENV="local"):
    """
    Create a boto3 client for a specific AWS service.

    This function creates AWS service clients using the appropriate session for
    the deployment environment. It's a convenience wrapper around get_boto3_session
    that directly returns a configured client for immediate use.

    Args:
        service_name (str): Name of the AWS service for the client. Examples:
                           - "bedrock-runtime": For Amazon Bedrock model inference
                           - "s3": For Amazon S3 object storage
                           - "ssm": For Systems Manager Parameter Store
                           - "secretsmanager": For AWS Secrets Manager
                           - "dynamodb": For DynamoDB database operations
        ENV (str): Environment identifier ("local" or "production", default: "local")

    Returns:
        boto3.Client: Configured boto3 client object for the specified service

    Raises:
        ValueError: If service_name is empty or None
        Exception: If client creation fails due to authentication or service issues

    Example:
        ```python
        # Create a Bedrock client for model inference
        bedrock = get_boto3_client("bedrock-runtime", ENV="production")

        # Create an S3 client for file operations
        s3 = get_boto3_client("s3", ENV="local")

        # Create an SSM client for parameter management
        ssm = get_boto3_client("ssm")
        ```

    Note:
        This function inherits all authentication and configuration behavior
        from get_boto3_session. Ensure proper credentials are configured
        for the target environment.
    """
    logging.info(
        f"Creating boto3 client for service: {service_name} in environment: {ENV}"
    )
    if not service_name:
        raise ValueError("service_name is required")

    try:
        # Create a boto3 session and use it to create the client
        session = get_boto3_session(ENV)
        client = session.client(service_name)
        logging.info(f"Successfully created boto3 client for service: {service_name}")
        return client
    except Exception as e:
        logger.error(f"Error creating boto3 client for service {service_name}: {e}")
        raise


def get_ssm_parameter(parameter_name, ENV="local"):
    """
    Retrieve a parameter from AWS Systems Manager Parameter Store.

    This function provides a secure way to retrieve configuration values,
    API keys, and other sensitive data stored in AWS SSM Parameter Store.
    It supports both regular and encrypted (SecureString) parameters.

    Args:
        parameter_name (str): Full path/name of the parameter to retrieve.
                             Examples:
                             - "/myapp/database/password"
                             - "/myapp/openai/api_key"
                             - "/myapp/config/model_name"
        ENV (str): Environment identifier ("local" or "production", default: "local")

    Returns:
        str or None: The decrypted value of the SSM parameter, or None if retrieval fails

    Example:
        ```python
        # Get an API key from Parameter Store
        openai_key = get_ssm_parameter("/myapp/openai/api_key", ENV="production")

        # Get a configuration value
        model_name = get_ssm_parameter("/myapp/config/default_model")

        if openai_key:
            # Use the retrieved parameter
            os.environ["OPENAI_API_KEY"] = openai_key
        ```

    Note:
        - Parameters are automatically decrypted if they are SecureString type
        - Returns None on failure rather than raising exceptions
        - Errors are logged for debugging purposes
        - Requires appropriate IAM permissions for SSM parameter access
    """
    try:
        ssm = get_boto3_client("ssm", ENV)
        parameter = ssm.get_parameter(Name=parameter_name, WithDecryption=True)
        return parameter["Parameter"]["Value"]
    except Exception as e:
        logger.error(f"Error getting SSM parameter {parameter_name}: {e}")
        return None


def get_secret(secret_name, key_in_secret: str | None = None, ENV="local"):
    """
    Retrieve a secret from AWS Secrets Manager.

    This function provides secure access to sensitive data stored in AWS Secrets Manager.
    It can retrieve entire secrets (JSON strings) or extract specific keys from
    JSON-formatted secrets, making it ideal for database credentials, API keys,
    and other structured sensitive data.

    Args:
        secret_name (str): Name or ARN of the secret to retrieve. Examples:
                          - "myapp/database/credentials"
                          - "arn:aws:secretsmanager:us-east-1:123456789012:secret:myapp/api-keys-AbCdEf"
        key_in_secret (Optional[str]): If the secret is JSON, extract this specific key.
                                      Examples:
                                      - "password": Extract password from DB credentials
                                      - "api_key": Extract API key from credentials object
        ENV (str): Environment identifier ("local" or "production", default: "local")

    Returns:
        str: The secret value or extracted key value

    Raises:
        ClientError: If secret retrieval fails (permissions, non-existent secret, etc.)
        KeyError: If key_in_secret is specified but not found in the JSON secret
        json.JSONDecodeError: If key_in_secret is specified but secret is not valid JSON

    Example:
        ```python
        # Get entire secret (e.g., a simple API key)
        api_key = get_secret("myapp/openai/api_key", ENV="production")

        # Get specific key from JSON secret
        db_password = get_secret(
            "myapp/database/credentials",
            key_in_secret="password",
            ENV="production"
        )

        # Example JSON secret structure:
        # {
        #   "username": "admin",
        #   "password": "secure_password",
        #   "host": "db.example.com",
        #   "port": 5432
        # }
        ```

    Security Notes:
        - Secrets are automatically decrypted by AWS
        - Always use IAM policies to control access to secrets
        - Consider using VPC endpoints for enhanced security
        - Rotate secrets regularly using AWS automatic rotation
    """
    secrets = get_boto3_client("secretsmanager", ENV)
    try:
        response = secrets.get_secret_value(SecretId=secret_name)
        secret = response["SecretString"]
    except ClientError as e:
        logger.error(f"Error getting secret {secret_name}: {e}")
        raise e
    if key_in_secret:
        secret = json.loads(secret)[key_in_secret]
    return secret
