"""Base provider interface for all LLM API providers.

Provides abstract classes for chat completion requests/responses and
a base class that concrete providers (Together, Fireworks) must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChatMessage:
    """A single chat message."""
    role: str  # "user", "assistant", "system"
    content: str


@dataclass
class CompletionRequest:
    """Parameters for a chat completion request."""
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 1024


@dataclass
class CompletionResponse:
    """Response from a chat completion."""
    content: str
    model: str
    usage: Optional[dict] = field(default=None, repr=False)


class BaseProvider(ABC):
    """Abstract base class for LLM API providers.

    Every provider must implement:
    - `chat_completion()`: send a request and return a CompletionResponse
    - `validate()`: check that the API key is valid
    - `name`: property returning the provider name (e.g. "together")
    """

    @abstractmethod
    def chat_completion(self, request: CompletionRequest) -> CompletionResponse:
        """Send a chat completion request and return the response."""
        ...

    @abstractmethod
    def validate(self) -> bool:
        """Check that required API keys are set. Return True if valid."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging/debugging."""
        ...
