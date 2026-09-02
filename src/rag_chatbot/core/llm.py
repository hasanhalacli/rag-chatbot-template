"""Multi-provider LLM client supporting OpenAI, Azure, Anthropic, xAI."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Union

from loguru import logger


@dataclass
class Message:
    """Chat message."""

    role: str  # system, user, assistant
    content: str


@dataclass
class LLMResponse:
    """LLM response container."""

    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    def generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> LLMResponse:
        """Generate response from messages."""
        pass

    @abstractmethod
    def stream(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> Generator[str, None, None]:
        """Stream response from messages."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider."""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key.
            model: Model name (gpt-4o, gpt-4-turbo, gpt-3.5-turbo).
        """
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> LLMResponse:
        """Generate response."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            finish_reason=response.choices[0].finish_reason,
        )

    def stream(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> Generator[str, None, None]:
        """Stream response."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AzureOpenAIProvider(BaseLLMProvider):
    """Azure OpenAI API provider."""

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        model: str = "gpt-4o",
        api_version: str = "2024-02-01",
    ):
        """Initialize Azure OpenAI provider.

        Args:
            api_key: Azure OpenAI API key.
            endpoint: Azure OpenAI endpoint URL.
            model: Deployment name.
            api_version: API version.
        """
        from openai import AzureOpenAI

        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        self.model = model

    def generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> LLMResponse:
        """Generate response."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            finish_reason=response.choices[0].finish_reason,
        )

    def stream(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> Generator[str, None, None]:
        """Stream response."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        """Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key.
            model: Model id (see the Anthropic models page).
        """
        from anthropic import Anthropic

        self.client = Anthropic(api_key=api_key)
        self.model = model

    def generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> LLMResponse:
        """Generate response."""
        # Extract system message if present
        system = None
        chat_messages = []
        for m in messages:
            if m.role == "system":
                system = m.content
            else:
                chat_messages.append({"role": m.role, "content": m.content})

        # The Anthropic SDK no longer accepts a temperature argument on messages.create().
        response = self.client.messages.create(
            model=self.model,
            messages=chat_messages,
            system=system,
            max_tokens=max_tokens,
            **kwargs,
        )

        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
            finish_reason=response.stop_reason,
        )

    def stream(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> Generator[str, None, None]:
        """Stream response."""
        system = None
        chat_messages = []
        for m in messages:
            if m.role == "system":
                system = m.content
            else:
                chat_messages.append({"role": m.role, "content": m.content})

        with self.client.messages.stream(
            model=self.model,
            messages=chat_messages,
            system=system,
            max_tokens=max_tokens,
            **kwargs,
        ) as stream:
            for text in stream.text_stream:
                yield text


class XAIProvider(BaseLLMProvider):
    """xAI Grok API provider."""

    def __init__(self, api_key: str, model: str = "grok-4.6"):
        """Initialize xAI provider.

        Args:
            api_key: xAI API key.
            model: Model id (see the xAI models page).
        """
        from openai import OpenAI

        # xAI uses OpenAI-compatible API
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )
        self.model = model

    def generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> LLMResponse:
        """Generate response."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            finish_reason=response.choices[0].finish_reason,
        )

    def stream(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> Generator[str, None, None]:
        """Stream response."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class LLMClient:
    """Unified LLM client supporting multiple providers."""

    PROVIDERS = {
        "openai": OpenAIProvider,
        "azure": AzureOpenAIProvider,
        "anthropic": AnthropicProvider,
        "xai": XAIProvider,
    }

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """Initialize LLM client.

        Args:
            provider: Provider name (openai, azure, anthropic, xai).
            model: Model name/deployment.
            api_key: API key (or set via environment).
            **kwargs: Provider-specific arguments.
        """
        if provider not in self.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}. Supported: {list(self.PROVIDERS)}")

        self.provider_name = provider

        # Get API key from environment if not provided
        if api_key is None:
            import os

            env_keys = {
                "openai": "OPENAI_API_KEY",
                "azure": "AZURE_OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "xai": "XAI_API_KEY",
            }
            api_key = os.getenv(env_keys[provider])

        if api_key is None:
            raise ValueError(f"API key required for {provider}")

        # Default models per provider
        default_models = {
            "openai": "gpt-4o",
            "azure": "gpt-4o",
            "anthropic": "claude-sonnet-4-6",
            "xai": "grok-4.6",
        }
        model = model or default_models[provider]

        # Initialize provider
        self.provider = self.PROVIDERS[provider](
            api_key=api_key,
            model=model,
            **kwargs,
        )

        logger.info(f"Initialized LLM client: {provider}/{model}")

    def generate(
        self,
        messages: Union[List[Message], List[Dict[str, str]], str],
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> LLMResponse:
        """Generate response.

        Args:
            messages: Messages (list of Message, list of dicts, or single string).
            system: System prompt (prepended if provided).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional arguments.

        Returns:
            LLMResponse with content and metadata.
        """
        # Normalize messages
        normalized = self._normalize_messages(messages, system)

        return self.provider.generate(
            messages=normalized,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def stream(
        self,
        messages: Union[List[Message], List[Dict[str, str]], str],
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> Generator[str, None, None]:
        """Stream response.

        Args:
            messages: Messages.
            system: System prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.
            **kwargs: Additional arguments.

        Yields:
            Response text chunks.
        """
        normalized = self._normalize_messages(messages, system)

        yield from self.provider.stream(
            messages=normalized,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def _normalize_messages(
        self,
        messages: Union[List[Message], List[Dict[str, str]], str],
        system: Optional[str] = None,
    ) -> List[Message]:
        """Normalize various message formats to List[Message]."""
        result = []

        if system:
            result.append(Message(role="system", content=system))

        if isinstance(messages, str):
            result.append(Message(role="user", content=messages))
        elif isinstance(messages, list):
            for m in messages:
                if isinstance(m, Message):
                    result.append(m)
                elif isinstance(m, dict):
                    result.append(Message(role=m["role"], content=m["content"]))
                else:
                    raise ValueError(f"Unknown message format: {type(m)}")

        return result
