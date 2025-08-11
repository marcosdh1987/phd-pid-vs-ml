"""
Dynamic Prompt Loading Utility with Fallback Mechanisms.

This module provides intelligent prompt loading capabilities for AI agents,
supporting multiple fallback strategies to ensure robust operation across
different deployment scenarios and model configurations.

Key Features:
    - Dynamic module importing for flexible prompt organization
    - Multi-level fallback strategy (specific → default → agent-specific)
    - Model-specific prompt customization
    - Comprehensive error handling and logging
    - Agent-aware prompt loading for multi-agent systems

Prompt Organization Strategy:
    The loader follows a hierarchical approach to find the most appropriate prompt:
    1. Model-specific prompt: prompts.{chain}.{model_core}
    2. Default prompt: prompts.{chain}.default
    3. Agent-specific default: agent_{pkg}.prompts.{chain}.default

Example Directory Structure:
    ```
    prompts/
    ├── research/
    │   ├── gpt4.py          # GPT-4 specific research prompt
    │   ├── claude.py        # Claude specific research prompt
    │   └── default.py       # Default research prompt
    └── reflection/
        ├── default.py       # Default reflection prompt
        └── ...
    ```

Example Usage:
    ```python
    from agent_rag.utils.prompt_loader import load_prompt

    # Load model-specific prompt
    prompt = load_prompt(
        chain_name="research",
        model_core="gpt4",
        agent_name="agent_rag"
    )

    # Automatically falls back to default if gpt4 variant not found
    prompt = load_prompt("reflection", "unknown_model", "agent_rag")
    ```
"""

import importlib
import logging


def load_prompt(chain_name: str, model_core: str, agent_name: str = None) -> str:
    """
    Load the most appropriate prompt for a given chain, model, and agent configuration.

    This function implements a sophisticated fallback strategy to ensure that a prompt
    is always available, even when specific model variants don't exist. It attempts
    to load prompts in order of specificity, from most specific to most general.

    Enhanced with Langfuse integration: If available, it will first try to load
    the prompt from Langfuse before falling back to local files.

    Fallback Strategy:
        1. Langfuse prompt: Try to load from Langfuse using chain_name as prompt name
        2. Model-specific prompt: prompts.{chain_name}.{model_core}
        3. Default prompt: prompts.{chain_name}.default
        4. Agent-specific default: agent_{pkg}.prompts.{chain_name}.default

    Args:
        chain_name (str): The name of the processing chain/step that needs a prompt.
                         Examples: "research", "reflection", "summarization"
                         Also used as the prompt name when searching in Langfuse.
        model_core (str): The core model identifier to customize prompts for.
                         Examples: "gpt4", "claude", "llama", "default"
        agent_name (str): The agent identifier for agent-specific fallbacks.
                         Format: "agent_{package_name}"
                         Examples: "agent_rag", "agent_chat"

    Returns:
        str: The loaded prompt content ready for use with the LLM

    Raises:
        RuntimeError: If no prompt can be loaded after all fallback attempts

    Example:
        ```python
        # Load a research prompt optimized for GPT-4
        research_prompt = load_prompt("research", "gpt4", "agent_rag")

        # Load a reflection prompt with default fallback
        reflection_prompt = load_prompt("reflection", "unknown", "agent_rag")

        # Use the loaded prompt
        messages = [{"role": "system", "content": research_prompt}]
        ```

    Prompt Module Structure:
        Each prompt module should define a constant named {CHAIN_NAME}_PROMPT:
        ```python
        # prompts/research/gpt4.py
        RESEARCH_PROMPT = '''
        You are a research assistant specialized in...
        '''
        ```

    Logging:
        - INFO: Successful prompt loading
        - CRITICAL: Failed attempts and fallback usage
        - Helps debug prompt loading issues in production"""
    # STEP 1: Try to load from Langfuse first
    try:
        from langfuse import Langfuse

        langfuse = Langfuse()

        # Use chain_name as the prompt name in Langfuse
        langfuse_prompt = langfuse.get_prompt(chain_name)

        # Get the prompt content from Langfuse
        prompt_content = langfuse_prompt.get_langchain_prompt()

        logging.info(f"Successfully loaded prompt '{chain_name}' from Langfuse")
        return prompt_content

    except Exception as e:
        logging.info(f"Langfuse not available or prompt '{chain_name}' not found: {e}")
        logging.info("Falling back to loading local file prompt...")

    # STEP 2: Continue with existing file-based fallback strategy
    # Build the model name from the chain name and the model core
    # Extract agent package name from agent_name if provided
    if agent_name:
        pkg = agent_name.split("_")[1] if "_" in agent_name else agent_name
        base_module = f"agent_{pkg}.prompts"
    else:
        base_module = "prompts"

    module_name = f"{base_module}.{chain_name}.{model_core}"
    try:
        # Dynamically import the module
        module = importlib.import_module(module_name)
        logging.info("Loaded prompt: %s", module_name)
        # Get the prompt attribute from the module
        return getattr(module, f"{chain_name.upper()}_PROMPT")
    except (ModuleNotFoundError, AttributeError):
        logging.info("Error loading prompt: %s", module_name)
        logging.info("Attempting to load default prompt (default)")
        # Get the prompt attribute from the module
        try:
            default_module_name = f"{base_module}.{chain_name}.default"
            module = importlib.import_module(default_module_name)
            logging.info("Loaded default prompt: %s", default_module_name)
            return getattr(module, f"{chain_name.upper()}_PROMPT")
        except (ModuleNotFoundError, AttributeError) as e:
            logging.critical("Error loading default prompt: %s", e)
            # Fallback: try loading from agent_<suffix>.prompts
            try:
                # use agent name as suffix
                pkg = agent_name.split("_")[1]
                # Load the agent-specific default prompt
                fallback_module = f"agent_{pkg}.prompts.{chain_name}.default"
                module = importlib.import_module(fallback_module)
                logging.info(
                    "Loaded agent-specific default prompt: %s", fallback_module
                )
                return getattr(module, f"{chain_name.upper()}_PROMPT")
            except (ModuleNotFoundError, AttributeError) as e2:
                logging.critical("Error loading agent-specific default prompt: %s", e2)
                raise RuntimeError("Critical error: Unable to load any prompt")
