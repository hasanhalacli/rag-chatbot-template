"""Chunking behaviour that must hold regardless of which LLM or embedding model is plugged in."""

import numpy as np
import pytest

from rag_chatbot.ingestion.chunker import (
    LLMChunker,
    RecursiveChunker,
    SemanticChunker,
    get_chunker,
)

LONG_PARAGRAPH = " ".join(f"word{i}" for i in range(400))  # one paragraph, no newlines, ~2.7k chars


def test_recursive_chunker_respects_chunk_size_without_overlap():
    chunks = RecursiveChunker(chunk_size=200, chunk_overlap=0).chunk(LONG_PARAGRAPH)
    assert len(chunks) > 1, "a paragraph far longer than chunk_size must be split"
    assert all(len(c.text) <= 200 for c in chunks), [len(c.text) for c in chunks]


def test_recursive_chunker_loses_no_words_without_overlap():
    chunks = RecursiveChunker(chunk_size=200, chunk_overlap=0).chunk(LONG_PARAGRAPH)
    assert " ".join(c.text for c in chunks).split() == LONG_PARAGRAPH.split()


def test_recursive_chunker_overlap_repeats_tail_of_previous_chunk():
    chunks = RecursiveChunker(chunk_size=200, chunk_overlap=3).chunk(LONG_PARAGRAPH)
    for prev, cur in zip(chunks, chunks[1:]):
        assert cur.text.split()[:3] == prev.text.split()[-3:]


def test_recursive_chunker_attaches_metadata_to_every_chunk():
    chunks = RecursiveChunker(chunk_size=200, chunk_overlap=0).chunk(LONG_PARAGRAPH, {"source": "x"})
    assert all(c.metadata == {"source": "x"} for c in chunks)


def test_short_text_is_a_single_chunk():
    chunks = RecursiveChunker(chunk_size=512).chunk("short text")
    assert [c.text for c in chunks] == ["short text"]


class _UnitEmbedder:
    """Two sentence groups that point in different directions."""

    def embed(self, sentences):
        out = []
        for s in sentences:
            out.append(np.array([1.0, 0.0]) if "alpha" in s else np.array([0.0, 1.0]))
        return out


def test_semantic_chunker_splits_where_similarity_drops():
    text = "alpha one. alpha two. beta one. beta two."
    chunks = SemanticChunker(_UnitEmbedder(), threshold=0.5, min_chunk_size=0, max_chunk_size=10_000).chunk(text)
    assert [c.text for c in chunks] == ["alpha one. alpha two.", "beta one. beta two."]


class _BrokenLLM:
    def generate(self, prompt, max_tokens=100):
        raise RuntimeError("provider down")


def test_llm_chunker_falls_back_when_the_model_fails():
    fallback = RecursiveChunker(chunk_size=50, chunk_overlap=0)
    chunks = LLMChunker(_BrokenLLM(), fallback_chunker=fallback).chunk(LONG_PARAGRAPH)
    assert chunks == fallback.chunk(LONG_PARAGRAPH)


def test_get_chunker_rejects_unknown_strategy():
    assert isinstance(get_chunker("recursive"), RecursiveChunker)
    with pytest.raises(ValueError):
        get_chunker("does-not-exist")
