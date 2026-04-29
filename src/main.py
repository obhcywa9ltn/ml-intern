"""ML Intern - A machine learning assistant powered by Claude.

This module serves as the main entry point for the ML Intern application,
providing an interactive interface for ML-related tasks and code review.
"""

import os
import sys
import logging
from typing import Optional

import anthropic
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default model to use
DEFAULT_MODEL = "claude-opus-4-5"
MAX_TOKENS = 4096


def create_client() -> anthropic.Anthropic:
    """Create and return an Anthropic client.

    Returns:
        anthropic.Anthropic: Configured Anthropic client.

    Raises:
        ValueError: If ANTHROPIC_API_KEY is not set.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Please set it before running the application."
        )
    return anthropic.Anthropic(api_key=api_key)


def get_system_prompt() -> str:
    """Return the system prompt for the ML intern assistant."""
    return (
        "You are an ML Intern assistant — a helpful, knowledgeable AI specializing in "
        "machine learning, deep learning, and data science. You help developers with:\n"
        "- Reviewing and improving ML code\n"
        "- Explaining ML concepts and architectures\n"
        "- Debugging model training issues\n"
        "- Suggesting best practices for ML pipelines\n"
        "- Analyzing and interpreting model results\n\n"
        "Be concise, precise, and practical in your responses."
    )


def chat(
    client: anthropic.Anthropic,
    user_message: str,
    conversation_history: Optional[list] = None,
    model: str = DEFAULT_MODEL,
) -> tuple[str, list]:
    """Send a message to the ML intern and get a response.

    Args:
        client: Anthropic client instance.
        user_message: The user's input message.
        conversation_history: Previous messages in the conversation.
        model: The Claude model to use.

    Returns:
        Tuple of (assistant response text, updated conversation history).
    """
    if conversation_history is None:
        conversation_history = []

    # Add user message to history
    conversation_history.append({"role": "user", "content": user_message})

    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=get_system_prompt(),
        messages=conversation_history,
    )

    assistant_message = response.content[0].text

    # Add assistant response to history
    conversation_history.append({"role": "assistant", "content": assistant_message})

    return assistant_message, conversation_history


def run_interactive_session() -> None:
    """Run an interactive chat session with the ML intern."""
    logger.info("Starting ML Intern interactive session")

    try:
        client = create_client()
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    conversation_history: list = []

    print("\n🤖 ML Intern ready! Type 'exit' or 'quit' to end the session.\n")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nSession ended.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("\nGoodbye! Happy ML coding! 🚀")
            break

        try:
            response, conversation_history = chat(client, user_input, conversation_history)
            print(f"\nML Intern: {response}")
        except anthropic.APIError as e:
            logger.error("API error: %s", e)
            print(f"\nError communicating with Claude: {e}")


if __name__ == "__main__":
    run_interactive_session()
