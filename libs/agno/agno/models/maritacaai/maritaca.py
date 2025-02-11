from dataclasses import dataclass
from os import getenv
from typing import Optional

from agno.models.base import Metrics
from agno.models.message import Message
from agno.models.openai.like import OpenAILike
from agno.utils.log import logger

try:
    from openai.types.chat.chat_completion_message import ChatCompletionMessage
    from openai.types.completion_usage import CompletionUsage
except ModuleNotFoundError:
    raise ImportError("`openai` not installed. Please install using `pip install openai`")


@dataclass
class Maritaca(OpenAILike):
    """
    A class for interacting with the Maritaca chat model via API.
    """

    id: str = "maritaca-chat"
    name: str = "Maritaca"
    provider: str = "Maritaca"
    api_key: Optional[str] = None  # Set your API key here or use an environment variable
    base_url: str = "https://chat.maritaca.ai"

    def __post_init__(self):
        """
        Initializes the Maritaca chat model after the dataclass is created.
        """
        super().__post_init__()
        self.api_key = self.api_key or getenv("MARITACA_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required for Maritaca. Set it via the constructor or environment variable.")

    def create_assistant_message(
        self,
        response_message: ChatCompletionMessage,
        metrics: Metrics,
        response_usage: Optional[CompletionUsage],
    ) -> Message:
        """
        Creates an assistant message from the API response.

        :param response_message: The response message from the API.
        :param metrics: Metrics object to track usage.
        :param response_usage: Usage information from the API response.
        :return: A Message object representing the assistant's response.
        """
        assistant_message = Message(
            role=response_message.role or "assistant",
            content=response_message.content,
        )

        # Handle tool calls if present
        if hasattr(response_message, "tool_calls") and response_message.tool_calls:
            try:
                assistant_message.tool_calls = [t.model_dump() for t in response_message.tool_calls]
            except Exception as e:
                logger.warning(f"Error processing tool calls: {e}")

        # Update metrics
        self.update_usage_metrics(assistant_message, metrics, response_usage)

        return assistant_message
