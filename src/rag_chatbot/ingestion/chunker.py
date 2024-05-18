"""Advanced text chunking strategies for RAG."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
from loguru import logger


@dataclass
class Chunk:
    """Text chunk with metadata."""

    text: str
    start_idx: int
    end_idx: int
    metadata: dict


class BaseChunker(ABC):
    """Base class for text chunking."""

    @abstractmethod
    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        """Split text into chunks.

        Args:
            text: Text to chunk.
            metadata: Optional metadata to attach.

        Returns:
            List of chunks.
        """
        pass


class RecursiveChunker(BaseChunker):
    """Recursive character text splitter.

    Splits on separators in order of priority, falling back to smaller
    separators when chunks are too large.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
        length_function: Callable[[str], int] = len,
    ):
        """Initialize recursive chunker.

        Args:
            chunk_size: Target chunk size.
            chunk_overlap: Overlap between chunks.
            separators: Separator priority list.
            length_function: Function to measure text length.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        self.length_function = length_function

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        """Split text recursively."""
        metadata = metadata or {}
        chunks = self._split_text(text, self.separators)

        return [
            Chunk(
                text=chunk_text,
                start_idx=0,  # Simplified for now
                end_idx=len(chunk_text),
                metadata=metadata,
            )
            for chunk_text in chunks
        ]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text."""
        if not separators:
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            splits = list(text)
        else:
            splits = text.split(separator)

        chunks = []
        current_chunk = []
        current_length = 0

        for split in splits:
            split_length = self.length_function(split)

            if current_length + split_length > self.chunk_size:
                if current_chunk:
                    chunk_text = separator.join(current_chunk)
                    if self.length_function(chunk_text) > self.chunk_size:
                        # Recursively split with next separator
                        chunks.extend(self._split_text(chunk_text, remaining_separators))
                    else:
                        chunks.append(chunk_text)

                current_chunk = [split]
                current_length = split_length
            else:
                current_chunk.append(split)
                current_length += split_length + len(separator)

        if current_chunk:
            chunks.append(separator.join(current_chunk))

        # Add overlap
        return self._add_overlap(chunks)

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between chunks."""
        if self.chunk_overlap == 0 or len(chunks) <= 1:
            return chunks

        overlapped = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                # Get end of previous chunk for overlap
                prev_words = chunks[i - 1].split()[-self.chunk_overlap :]
                chunk = " ".join(prev_words) + " " + chunk
            overlapped.append(chunk)

        return overlapped


class SentenceChunker(BaseChunker):
    """Sentence-based chunker using NLTK or spaCy."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 1,  # Number of sentences to overlap
        min_sentences: int = 1,
    ):
        """Initialize sentence chunker.

        Args:
            chunk_size: Target chunk size in characters.
            chunk_overlap: Number of sentences to overlap.
            min_sentences: Minimum sentences per chunk.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_sentences = min_sentences

        try:
            import nltk

            nltk.download("punkt", quiet=True)
            self.sent_tokenize = nltk.sent_tokenize
        except ImportError:
            # Fallback to simple splitting
            self.sent_tokenize = lambda x: x.split(". ")

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        """Split text by sentences."""
        metadata = metadata or {}
        sentences = self.sent_tokenize(text)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            if current_length + len(sentence) > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        start_idx=0,
                        end_idx=len(chunk_text),
                        metadata=metadata,
                    )
                )

                # Keep overlap sentences
                current_chunk = current_chunk[-self.chunk_overlap :]
                current_length = sum(len(s) for s in current_chunk)

            current_chunk.append(sentence)
            current_length += len(sentence)

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    start_idx=0,
                    end_idx=len(chunk_text),
                    metadata=metadata,
                )
            )

        return chunks


class SemanticChunker(BaseChunker):
    """Semantic chunking based on embedding similarity.

    Splits text where semantic similarity between adjacent segments
    drops below a threshold.
    """

    def __init__(
        self,
        embedding_model,
        threshold: float = 0.7,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
    ):
        """Initialize semantic chunker.

        Args:
            embedding_model: Embedding model for similarity.
            threshold: Similarity threshold for splitting.
            min_chunk_size: Minimum chunk size.
            max_chunk_size: Maximum chunk size.
        """
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        """Split text at semantic boundaries."""
        metadata = metadata or {}

        # Split into sentences first
        sentences = text.replace("\n", " ").split(". ")
        sentences = [s.strip() + "." for s in sentences if s.strip()]

        if len(sentences) <= 1:
            return [Chunk(text=text, start_idx=0, end_idx=len(text), metadata=metadata)]

        # Get embeddings for each sentence
        embeddings = self.embedding_model.embed(sentences)

        # Calculate similarities between adjacent sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # Find split points where similarity drops
        split_points = [0]
        for i, sim in enumerate(similarities):
            if sim < self.threshold:
                split_points.append(i + 1)
        split_points.append(len(sentences))

        # Create chunks
        chunks = []
        for i in range(len(split_points) - 1):
            start = split_points[i]
            end = split_points[i + 1]
            chunk_text = " ".join(sentences[start:end])

            # Enforce size limits
            if len(chunk_text) < self.min_chunk_size and chunks:
                # Merge with previous
                prev = chunks.pop()
                chunk_text = prev.text + " " + chunk_text
            elif len(chunk_text) > self.max_chunk_size:
                # Split further with recursive chunker
                sub_chunker = RecursiveChunker(
                    chunk_size=self.max_chunk_size,
                    chunk_overlap=50,
                )
                sub_chunks = sub_chunker.chunk(chunk_text, metadata)
                chunks.extend(sub_chunks)
                continue

            chunks.append(
                Chunk(
                    text=chunk_text,
                    start_idx=0,
                    end_idx=len(chunk_text),
                    metadata=metadata,
                )
            )

        return chunks


class LLMChunker(BaseChunker):
    """LLM-based chunking for complex documents.

    Uses an LLM to identify logical section boundaries.
    """

    def __init__(
        self,
        llm_client,
        max_chunk_size: int = 1000,
        fallback_chunker: Optional[BaseChunker] = None,
    ):
        """Initialize LLM chunker.

        Args:
            llm_client: LLM client for boundary detection.
            max_chunk_size: Maximum chunk size.
            fallback_chunker: Chunker to use if LLM fails.
        """
        self.llm_client = llm_client
        self.max_chunk_size = max_chunk_size
        self.fallback_chunker = fallback_chunker or RecursiveChunker(
            chunk_size=max_chunk_size
        )

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        """Use LLM to identify chunk boundaries."""
        metadata = metadata or {}

        # For very long texts, use fallback
        if len(text) > 10000:
            logger.info("Text too long for LLM chunking, using fallback")
            return self.fallback_chunker.chunk(text, metadata)

        prompt = f"""Analyze this text and identify logical section boundaries.
Output line numbers where new sections begin (comma-separated).
Only output numbers, nothing else.

Text:
{text[:5000]}

Section boundaries (line numbers):"""

        try:
            response = self.llm_client.generate(prompt, max_tokens=100)
            boundaries = [int(x.strip()) for x in response.content.split(",")]
        except Exception as e:
            logger.warning(f"LLM chunking failed: {e}, using fallback")
            return self.fallback_chunker.chunk(text, metadata)

        # Split at boundaries
        lines = text.split("\n")
        chunks = []
        prev_boundary = 0

        for boundary in boundaries + [len(lines)]:
            if boundary <= prev_boundary:
                continue

            chunk_text = "\n".join(lines[prev_boundary:boundary])

            if chunk_text.strip():
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        start_idx=prev_boundary,
                        end_idx=boundary,
                        metadata=metadata,
                    )
                )

            prev_boundary = boundary

        return chunks if chunks else self.fallback_chunker.chunk(text, metadata)


def get_chunker(
    strategy: str = "recursive",
    **kwargs,
) -> BaseChunker:
    """Get chunker by strategy name.

    Args:
        strategy: Chunking strategy name.
        **kwargs: Strategy-specific arguments.

    Returns:
        Chunker instance.
    """
    chunkers = {
        "recursive": RecursiveChunker,
        "sentence": SentenceChunker,
        "semantic": SemanticChunker,
        "llm": LLMChunker,
    }

    if strategy not in chunkers:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    return chunkers[strategy](**kwargs)
