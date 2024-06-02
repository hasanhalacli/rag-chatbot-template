"""Conversation memory management."""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..core.llm import Message


class ConversationMemory:
    """Simple in-memory conversation storage."""

    def __init__(self, max_messages: int = 20):
        """Initialize memory.

        Args:
            max_messages: Maximum messages per conversation.
        """
        self.max_messages = max_messages
        self._storage: Dict[str, List[Message]] = defaultdict(list)

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ) -> None:
        """Add message to conversation.

        Args:
            conversation_id: Conversation identifier.
            role: Message role (user/assistant).
            content: Message content.
        """
        messages = self._storage[conversation_id]
        messages.append(Message(role=role, content=content))

        # Trim if exceeds max
        if len(messages) > self.max_messages:
            self._storage[conversation_id] = messages[-self.max_messages :]

    def get_history(self, conversation_id: str) -> List[Message]:
        """Get conversation history.

        Args:
            conversation_id: Conversation identifier.

        Returns:
            List of messages.
        """
        return self._storage.get(conversation_id, [])

    def clear(self, conversation_id: str) -> None:
        """Clear conversation history.

        Args:
            conversation_id: Conversation to clear.
        """
        if conversation_id in self._storage:
            del self._storage[conversation_id]

    def clear_all(self) -> None:
        """Clear all conversations."""
        self._storage.clear()

    def list_conversations(self) -> List[str]:
        """List all conversation IDs.

        Returns:
            List of conversation IDs.
        """
        return list(self._storage.keys())


class SlidingWindowMemory(ConversationMemory):
    """Memory with sliding window over recent messages."""

    def __init__(self, window_size: int = 6, max_messages: int = 50):
        """Initialize sliding window memory.

        Args:
            window_size: Number of recent messages to include.
            max_messages: Maximum messages to store.
        """
        super().__init__(max_messages)
        self.window_size = window_size

    def get_history(self, conversation_id: str) -> List[Message]:
        """Get recent messages within window.

        Args:
            conversation_id: Conversation identifier.

        Returns:
            Recent messages.
        """
        messages = self._storage.get(conversation_id, [])
        return messages[-self.window_size :]


class SummaryMemory(ConversationMemory):
    """Memory that summarizes older messages."""

    def __init__(
        self,
        llm_client,
        summary_threshold: int = 10,
        max_messages: int = 50,
    ):
        """Initialize summary memory.

        Args:
            llm_client: LLM client for summarization.
            summary_threshold: Messages before summarizing.
            max_messages: Maximum messages to store.
        """
        super().__init__(max_messages)
        self.llm_client = llm_client
        self.summary_threshold = summary_threshold
        self._summaries: Dict[str, str] = {}

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ) -> None:
        """Add message and summarize if needed."""
        super().add_message(conversation_id, role, content)

        messages = self._storage[conversation_id]
        if len(messages) >= self.summary_threshold:
            self._summarize(conversation_id)

    def _summarize(self, conversation_id: str) -> None:
        """Summarize older messages."""
        messages = self._storage[conversation_id]
        to_summarize = messages[: -4]  # Keep last 2 turns

        if not to_summarize:
            return

        history_text = "\n".join(
            f"{m.role.capitalize()}: {m.content}" for m in to_summarize
        )

        prompt = f"Summarize this conversation in 2-3 sentences:\n\n{history_text}"
        response = self.llm_client.generate(prompt, max_tokens=150)

        self._summaries[conversation_id] = response.content
        self._storage[conversation_id] = messages[-4:]

    def get_history(self, conversation_id: str) -> List[Message]:
        """Get history with summary prefix."""
        messages = self._storage.get(conversation_id, [])
        summary = self._summaries.get(conversation_id)

        if summary:
            return [
                Message(role="system", content=f"Previous conversation summary: {summary}"),
                *messages,
            ]
        return messages
