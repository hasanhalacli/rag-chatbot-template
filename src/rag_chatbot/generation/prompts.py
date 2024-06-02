"""Prompt templates for RAG generation."""

from dataclasses import dataclass
from typing import Dict


@dataclass
class PromptTemplate:
    """Simple prompt template with variable substitution."""

    template: str

    def format(self, **kwargs) -> str:
        """Format template with variables."""
        return self.template.format(**kwargs)


RAG_PROMPT = """Answer the question based on the following context.
If the context doesn't contain enough information, say so.
Be concise and accurate. Cite sources using [1], [2], etc.

Context:
{context}

Question: {question}

Answer:"""


CONDENSE_PROMPT = """Given the conversation history and a follow-up question,
rephrase the follow-up question to be a standalone question.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""


SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on
provided context. You are accurate, concise, and cite your sources.
If you don't know the answer, say so honestly."""


QA_PROMPT = """Based on the context below, answer the question accurately.

Context: {context}

Question: {question}

If the context doesn't contain the answer, respond with "I don't have enough information to answer this question."

Answer:"""
