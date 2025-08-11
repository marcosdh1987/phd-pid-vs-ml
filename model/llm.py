"""
Large Language Model (LLM) factory and custom implementations for the RAG agent.

This module provides a flexible factory pattern for creating different types of LLM instances,
supporting multiple providers: OpenAI, Amazon Bedrock (Claude), and Ollama.
It includes custom implementations for tool calling and response parsing.

Classes:
    CustomOllamaLLM: Custom LLM implementation for Ollama with tool calling support
    ModelFactory: Factory class for creating LLM instances based on configuration
"""

import json
import logging
import os

import requests
from dotenv import load_dotenv
from langchain.llms.base import LLM
from langchain_aws import ChatBedrock
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import Field

from utils.boto_session import get_boto3_client
from model.config import MODEL_CONFIGS

load_dotenv(override=True)


class CustomOllamaLLM(LLM):
    """
    Custom LLM implementation for Ollama with tool calling capabilities.

    This class extends LangChain's base LLM to support Ollama models with
    custom tool calling functionality. It handles streaming responses,
    JSON parsing, and tool execution coordination.

    Attributes:
        model (str): Name of the Ollama model to use
        endpoint (str): Ollama API endpoint URL
        tools (list): List of available tools for this LLM instance
        _is_tool_calling (bool): Internal flag to prevent recursive tool calls

    Example:
        ```python
        llm = CustomOllamaLLM(
            model="llama2",
            endpoint="http://localhost:11434/api/generate"
        )
        response = llm.invoke("Generate medical consultation guidance")
        ```
    """

    model: str = Field(...)
    endpoint: str = Field(...)
    tools: list = []  # Store available tools for this LLM instance
    _is_tool_calling: bool = False

    def parse_response(self, raw_response: str) -> str:
        """
        Parse streaming JSON response from Ollama API.

        Ollama returns streaming JSON responses where each line contains
        a JSON object with partial response data. This method concatenates
        all response fragments into a complete response string.

        Args:
            raw_response (str): Raw response from Ollama API (JSON lines format)

        Returns:
            str: Complete parsed response text

        Example:
            Input: '{"response":"Hello"}\n{"response":" world"}\n{"done":true}'
            Output: "Hello world"
        """
        response_text = ""
        try:
            # Handle case where response is already a dict
            if isinstance(raw_response, dict):
                return raw_response.get("response", "")

            # Handle streaming response
            for line in raw_response.strip().splitlines():
                try:
                    data = json.loads(line)
                    # Concatenate the response text
                    if "response" in data:
                        response_text += data["response"]
                except json.JSONDecodeError:
                    logging.debug(f"Failed to decode JSON line: {line}")
                    continue
            return response_text
        except Exception as e:
            logging.error(f"Error parsing response: {str(e)}")
            return raw_response

    def __init__(self, **kwargs):
        """Initialize the CustomOllamaLLM with provided configuration."""
        super().__init__(**kwargs)

    @property
    def _llm_type(self) -> str:
        """Return the LLM type identifier for LangChain compatibility."""
        return "custom_ollama"

    def _call(self, prompt: str, stop: list[str] | None = None) -> str:
        """
        Call Ollama API with tool calling support.

        This method handles the core LLM interaction, including:
        1. Tool description injection into prompts
        2. API communication with Ollama
        3. Tool call detection and execution
        4. Response parsing and formatting

        Args:
            prompt (str): User prompt or system message
            stop (Optional[List[str]]): Stop sequences for generation

        Returns:
            str: LLM response or tool execution result

        Note:
            Tool calling uses a custom format:
            TOOL: <tool_name>
            ARGS: {"param": "value"}
        """
        # Add tool instructions if tools are available and not already calling
        if self.tools and not self._is_tool_calling:
            # Prepare tool definitions for prompt injection
            tool_descriptions = []
            for tool in self.tools:
                tool_desc = (
                    f"Tool Name: {tool.name}\nTool Description: {tool.description}\n"
                )
                tool_descriptions.append(tool_desc)

            tools_prompt = "\n".join(tool_descriptions)

            enhanced_prompt = (
                "You have access to the following tools:\n\n"
                f"{tools_prompt}\n\n"
                "To use a tool, respond with a format like this:\n"
                'TOOL: <tool_name>\nARGS: {"query": "what you want to search"}\n\n'
                "After getting the tool result, you can use it to respond to the user.\n\n"
                "Original user message:\n"
                f"{prompt}"
            )
            prompt = enhanced_prompt

        # Prepare API request
        data = {"model": self.model, "prompt": prompt}
        response = requests.post(self.endpoint, json=data)
        response.raise_for_status()

        # Process response for tool call detection
        parsed_response = self.parse_response(response.text)

        # Check if response contains a tool call
        if not self._is_tool_calling and "TOOL:" in parsed_response:
            self._is_tool_calling = True
            try:
                # Extract tool call information
                tool_line = parsed_response.split("TOOL:")[1].split("\n")[0].strip()
                args_line = parsed_response.split("ARGS:")[1].split("\n")[0].strip()

                # Find and execute the requested tool
                tool_name = tool_line
                for tool in self.tools:
                    if tool.name == tool_name:
                        # Parse arguments and execute tool
                        args = json.loads(args_line)
                        tool_result = tool.invoke(args)

                        # Build follow-up prompt with tool result
                        follow_up_prompt = (
                            f"{prompt}\n\n"
                            f"Tool {tool_name} result:\n{tool_result}\n\n"
                            "Based on this result, please provide a clear response to the user."
                        )

                        # Reset state and make new call with result
                        self._is_tool_calling = False
                        return self._call(follow_up_prompt, stop)

                # Tool not found
                self._is_tool_calling = False
                return f"Error: Could not find tool '{tool_name}'."

            except Exception as e:
                self._is_tool_calling = False
                logging.error(f"Error processing tool call: {str(e)}")
                return f"Error processing tool call: {str(e)}"

        # Reset state for future calls
        self._is_tool_calling = False

        return parsed_response

    def bind_tools(self, tools: list):
        """
        Bind tools to this LLM instance for tool calling support.

        Args:
            tools (list): List of LangChain Tool objects

        Returns:
            CustomOllamaLLM: This instance with tools attached

        Example:
            ```python
            llm_with_tools = llm.bind_tools([custom_tools])
            ```
        """
        self.tools = tools
        return self


class ModelFactory:
    """
    Factory class for creating different LLM instances based on configuration.

    This factory supports multiple LLM providers and handles the complexity of
    different initialization patterns, authentication methods, and model-specific
    configurations. It provides a unified interface for creating:

    - OpenAI models (GPT-3.5, GPT-4, etc.)
    - Amazon Bedrock models (Claude, Titan, etc.)
    - Google Gemini models (Gemini 2.0 Flash, Gemini 2.5 Pro, etc.)
    - Ollama models (local/self-hosted models)

    The factory uses the MODEL_CONFIGS configuration to determine provider-specific
    settings and translates simplified model names to official provider model IDs.

    Attributes:
        model_name (str): Simplified model name (e.g., "gpt-4", "claude-3", "gemini-2.0-flash")
        temperature (float): Sampling temperature for model responses
        max_tokens (int): Maximum tokens in model responses
        verbose (bool): Enable verbose logging for debugging
        region_name (str): AWS region for Bedrock models
        top_p (float): Nucleus sampling parameter
        top_k (int): Top-k sampling parameter

    Example:
        ```python
        # Create a GPT-4 model
        factory = ModelFactory(model_name="gpt-4", temperature=0.3)
        llm = factory.create_model()

        # Create a Claude model on Bedrock
        factory = ModelFactory(model_name="claude-3", region_name="us-west-2")
        llm = factory.create_model()

        # Create a Gemini model
        factory = ModelFactory(model_name="gemini-2.0-flash", temperature=0.7)
        llm = factory.create_model()

        # Create a local Ollama model
        factory = ModelFactory(model_name="llama2")
        llm = factory.create_model()
        ```
    """

    def __init__(
        self,
        model_name="gpt-4",
        temperature=0.5,
        max_tokens=None,
        verbose=False,
        region_name="us-east-1",
        top_p=0.9,
        top_k=20,
    ):
        """
        Initialize the ModelFactory with configuration parameters.

        Args:
            model_name (str): Model identifier from MODEL_CONFIGS
            temperature (float): Controls randomness in responses (0.0-1.0)
            max_tokens (int): Maximum tokens to generate (None for model default)
            verbose (bool): Enable detailed logging
            region_name (str): AWS region for Bedrock (ignored for other providers)
            top_p (float): Nucleus sampling threshold (0.0-1.0)
            top_k (int): Top-k sampling limit for token selection
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.region_name = os.getenv("AWS_REGION", "us-east-1")
        # OpenAI specific configuration
        self.openai_api_key = os.getenv("OPENAI_API_KEY", None)
        # Bedrock specific configuration
        self.bedrock_client = None
        self.top_p = top_p
        self.top_k = top_k
        self.environment = os.getenv("ENV", "local")

    def create_model(self):
        """
        Create and return the appropriate LLM instance based on model configuration.

        This method acts as the main factory method, routing to provider-specific
        creation methods based on the model's provider configuration. It handles
        all the complexity of different initialization patterns and authentication.

        Returns:
            LLM: Configured LLM instance ready for use

        Raises:
            ValueError: If the model name is not supported or misconfigured
            Exception: If there are authentication or connection issues

        Example:
            ```python
            factory = ModelFactory("gpt-4")
            llm = factory.create_model()
            response = llm.invoke("Hello, world!")
            ```
        """
        if self._is_openai_model():
            return self._create_openai_model()
        elif self._is_bedrock_model():
            return self._create_bedrock_model()
        elif self._is_google_model():
            return self._create_google_model()
        elif self._is_ollama_model():
            return self._create_ollama_model()
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    def _is_openai_model(self):
        """
        Check if the configured model is from OpenAI.

        Returns:
            bool: True if model provider is OpenAI
        """
        config = MODEL_CONFIGS.get(self.model_name, {})
        return config.get("provider") == "openai"

    def _is_bedrock_model(self):
        """
        Check if the configured model is from Amazon Bedrock.

        Returns:
            bool: True if model provider is Bedrock
        """
        config = MODEL_CONFIGS.get(self.model_name, {})
        return config.get("provider") == "bedrock"

    def _is_google_model(self):
        """
        Check if the configured model is from Google Gemini.

        Returns:
            bool: True if model provider is Google
        """
        config = MODEL_CONFIGS.get(self.model_name, {})
        return config.get("provider") == "google"

    def _is_ollama_model(self):
        """
        Check if the configured model is from Ollama.

        Returns:
            bool: True if model provider is Ollama
        """
        config = MODEL_CONFIGS.get(self.model_name, {})
        return config.get("provider") == "ollama"

    def _create_openai_model(self):
        """
        Create an OpenAI ChatGPT model instance.

        Configures a ChatOpenAI instance with the specified parameters,
        including API key authentication and model-specific settings.

        Returns:
            ChatOpenAI: Configured OpenAI model instance

        Raises:
            Exception: If API key is missing or invalid
        """

        return ChatOpenAI(
            model_name=self._translate_openai_model_name(),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            verbose=self.verbose,
            api_key=self.openai_api_key,
        )

    def _create_bedrock_model(self):
        """
        Create an Amazon Bedrock model instance.

        Initializes a ChatBedrock instance with AWS authentication,
        region configuration, and model-specific parameters. Handles
        both direct model IDs and inference profile IDs for optimized routing.

        Returns:
            ChatBedrock: Configured Bedrock model instance

        Raises:
            Exception: If AWS credentials are invalid or region is inaccessible

        Note:
            Requires proper AWS credentials configuration through environment
            variables, IAM roles, or AWS credentials file.
        """
        if not self.bedrock_client:
            logging.info("Creating Bedrock client")
            try:
                self.bedrock_client = get_boto3_client(
                    service_name="bedrock-runtime", ENV=self.environment
                )
                logging.info("Bedrock client created successfully")
            except Exception as e:
                logging.error(f"Error creating Bedrock client: {e}")
                raise

        config = MODEL_CONFIGS[self.model_name]
        model_kwargs = {
            "max_tokens": self.max_tokens or 2048,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "stop_sequences": ["\n\nHuman"],
            "anthropic_version": "bedrock-2023-05-31",
        }

        if "inference_profile_id" in config:
            return ChatBedrock(
                client=self.bedrock_client,
                model_id=config["inference_profile_id"],
                model_kwargs=model_kwargs,
            )
        else:
            return ChatBedrock(
                client=self.bedrock_client,
                model_id=config["model_id"],
                model_kwargs=model_kwargs,
            )

    def _create_google_model(self):
        """
        Create a Google Gemini model instance.

        Initializes a ChatGoogleGenerativeAI instance with Google API authentication
        and model-specific parameters. Supports various Gemini models including
        multimodal capabilities.

        Returns:
            ChatGoogleGenerativeAI: Configured Google Gemini model instance

        Raises:
            Exception: If GOOGLE_API_KEY is missing or invalid

        Note:
            Requires GOOGLE_API_KEY environment variable to be set.
            Get your API key from https://ai.google.dev/gemini-api/docs/api-key
        """
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise Exception(
                "GOOGLE_API_KEY environment variable is not set. "
                "Get your API key from https://ai.google.dev/gemini-api/docs/api-key"
            )

        config = MODEL_CONFIGS[self.model_name]

        return ChatGoogleGenerativeAI(
            model=config["model_id"],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            verbose=self.verbose,
            google_api_key=google_api_key,
            max_retries=2,
        )

    def _create_ollama_model(self):
        """
        Create a custom Ollama model instance.

        Initializes a CustomOllamaLLM instance configured to communicate
        with a local or remote Ollama server. Ollama provides access to
        open-source models like Llama, Mistral, and others.

        Returns:
            CustomOllamaLLM: Configured Ollama model instance

        Raises:
            Exception: If Ollama server is unreachable or model is unavailable

        Note:
            Requires Ollama server to be running and the specified model
            to be downloaded/available on the server.
        """
        config = MODEL_CONFIGS[self.model_name]
        return CustomOllamaLLM(model=config["model_id"], endpoint=config["endpoint"])

    def _translate_openai_model_name(self):
        """
        Translate simplified model names to official OpenAI model identifiers.

        Maps user-friendly model names (e.g., "gpt-4") to the exact model IDs
        required by the OpenAI API (e.g., "gpt-4-0613").

        Returns:
            str: Official OpenAI model identifier

        Example:
            "gpt-4" -> "gpt-4-0613"
            "gpt-3.5-turbo" -> "gpt-3.5-turbo-0613"
        """
        return MODEL_CONFIGS[self.model_name]["model_id"]

    def _translate_bedrock_model_name(self):
        """
        Translate simplified model names to official Bedrock model identifiers.

        Maps user-friendly model names to the exact model IDs or ARNs
        required by the Amazon Bedrock API.

        Returns:
            str: Official Bedrock model identifier

        Example:
            "claude-3" -> "anthropic.claude-3-sonnet-20240229-v1:0"
            "titan" -> "amazon.titan-text-express-v1"
        """
        return MODEL_CONFIGS[self.model_name]["model_id"]

    def _translate_google_model_name(self):
        """
        Translate simplified model names to official Google Gemini model identifiers.

        Maps user-friendly model names to the exact model IDs
        required by the Google Generative AI API.

        Returns:
            str: Official Google Gemini model identifier

        Example:
            "gemini-2.0-flash" -> "gemini-2.0-flash"
            "gemini-2.5-pro" -> "gemini-2.5-pro-preview-03-25"
        """
        return MODEL_CONFIGS[self.model_name]["model_id"]
