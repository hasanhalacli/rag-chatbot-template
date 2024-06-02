"""RAG chain implementation."""

from dataclasses import dataclass
from typing import Generator, List, Optional

from loguru import logger

from ..core.llm import LLMClient, Message
from ..retrieval.qdrant_store import Document
from ..retrieval.retrievers import Retriever
from ..retrieval.reranker import CrossEncoderReranker
from .memory import ConversationMemory
from .prompts import RAG_PROMPT, CONDENSE_PROMPT


@dataclass
class RAGResponse:
    """RAG response with answer and sources."""

    answer: str
    sources: List[Document]
    conversation_id: Optional[str] = None


class RAGChain:
    """RAG chain combining retrieval and generation."""

    def __init__(
        self,
        retriever: Retriever,
        llm_client: LLMClient,
        reranker: Optional[CrossEncoderReranker] = None,
        memory: Optional[ConversationMemory] = None,
        system_prompt: Optional[str] = None,
        use_history_for_retrieval: bool = True,
    ):
        """Initialize RAG chain.

        Args:
            retriever: Document retriever.
            llm_client: LLM client for generation.
            reranker: Optional reranker for improved relevance.
            memory: Optional conversation memory.
            system_prompt: Custom system prompt.
            use_history_for_retrieval: Condense history into query.
        """
        self.retriever = retriever
        self.llm_client = llm_client
        self.reranker = reranker
        self.memory = memory or ConversationMemory()
        self.system_prompt = system_prompt
        self.use_history_for_retrieval = use_history_for_retrieval

    def query(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        top_k: Optional[int] = None,
        filter_conditions: Optional[dict] = None,
    ) -> RAGResponse:
        """Query the RAG chain.

        Args:
            question: User question.
            conversation_id: ID for conversation tracking.
            top_k: Number of documents to retrieve.
            filter_conditions: Metadata filters.

        Returns:
            RAGResponse with answer and sources.
        """
        # Get conversation history
        history = self.memory.get_history(conversation_id) if conversation_id else []

        # Condense history into standalone question if needed
        retrieval_query = question
        if history and self.use_history_for_retrieval:
            retrieval_query = self._condense_question(question, history)
            logger.debug(f"Condensed query: {retrieval_query}")

        # Retrieve relevant documents
        documents = self.retriever.retrieve(
            query=retrieval_query,
            top_k=top_k,
            filter_conditions=filter_conditions,
        )

        # Rerank if reranker available
        if self.reranker and documents:
            documents = self.reranker.rerank(retrieval_query, documents)

        # Generate answer
        answer = self._generate_answer(question, documents, history)

        # Save to memory
        if conversation_id:
            self.memory.add_message(conversation_id, "user", question)
            self.memory.add_message(conversation_id, "assistant", answer)

        return RAGResponse(
            answer=answer,
            sources=documents,
            conversation_id=conversation_id,
        )

    def stream(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        top_k: Optional[int] = None,
        filter_conditions: Optional[dict] = None,
    ) -> Generator[str, None, None]:
        """Stream RAG response.

        Args:
            question: User question.
            conversation_id: Conversation ID.
            top_k: Number of documents.
            filter_conditions: Metadata filters.

        Yields:
            Response text chunks.
        """
        history = self.memory.get_history(conversation_id) if conversation_id else []

        retrieval_query = question
        if history and self.use_history_for_retrieval:
            retrieval_query = self._condense_question(question, history)

        documents = self.retriever.retrieve(
            query=retrieval_query,
            top_k=top_k,
            filter_conditions=filter_conditions,
        )

        if self.reranker and documents:
            documents = self.reranker.rerank(retrieval_query, documents)

        # Build prompt
        context = self._format_context(documents)
        messages = self._build_messages(question, context, history)

        # Stream response
        full_response = ""
        for chunk in self.llm_client.stream(messages, system=self.system_prompt):
            full_response += chunk
            yield chunk

        # Save to memory after streaming
        if conversation_id:
            self.memory.add_message(conversation_id, "user", question)
            self.memory.add_message(conversation_id, "assistant", full_response)

    def _condense_question(self, question: str, history: List[Message]) -> str:
        """Condense chat history and question into standalone query."""
        history_text = "\n".join(
            f"{m.role.capitalize()}: {m.content}" for m in history[-4:]
        )

        prompt = CONDENSE_PROMPT.format(
            chat_history=history_text,
            question=question,
        )

        response = self.llm_client.generate(prompt, max_tokens=150)
        return response.content.strip()

    def _generate_answer(
        self,
        question: str,
        documents: List[Document],
        history: List[Message],
    ) -> str:
        """Generate answer from question and context."""
        context = self._format_context(documents)
        messages = self._build_messages(question, context, history)

        response = self.llm_client.generate(messages, system=self.system_prompt)
        return response.content

    def _format_context(self, documents: List[Document]) -> str:
        """Format documents into context string."""
        if not documents:
            return "No relevant documents found."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            context_parts.append(f"[{i}] Source: {source}\n{doc.content}")

        return "\n\n".join(context_parts)

    def _build_messages(
        self,
        question: str,
        context: str,
        history: List[Message],
    ) -> List[Message]:
        """Build message list for LLM."""
        messages = []

        # Add history
        for msg in history[-6:]:  # Last 3 turns
            messages.append(msg)

        # Add current question with context
        user_message = RAG_PROMPT.format(context=context, question=question)
        messages.append(Message(role="user", content=user_message))

        return messages

    def clear_memory(self, conversation_id: str) -> None:
        """Clear conversation memory.

        Args:
            conversation_id: Conversation to clear.
        """
        self.memory.clear(conversation_id)
