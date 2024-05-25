"""Retrieval strategies for RAG."""

from dataclasses import dataclass
from typing import List, Optional

from loguru import logger

from ..core.embeddings import EmbeddingModel
from .qdrant_store import Document, QdrantStore


class Retriever:
    """Basic retriever using vector similarity."""

    def __init__(
        self,
        store: QdrantStore,
        embedding_model: EmbeddingModel,
        collection: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ):
        """Initialize retriever.

        Args:
            store: Qdrant store instance.
            embedding_model: Embedding model for queries.
            collection: Collection to search.
            top_k: Number of documents to retrieve.
            score_threshold: Minimum similarity score.
        """
        self.store = store
        self.embedding_model = embedding_model
        self.collection = collection
        self.top_k = top_k
        self.score_threshold = score_threshold

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_conditions: Optional[dict] = None,
    ) -> List[Document]:
        """Retrieve relevant documents.

        Args:
            query: Query text.
            top_k: Override default top_k.
            filter_conditions: Metadata filters.

        Returns:
            List of relevant documents.
        """
        query_embedding = self.embedding_model.embed_query(query)

        documents = self.store.search(
            collection=self.collection,
            query_embedding=query_embedding.tolist(),
            top_k=top_k or self.top_k,
            score_threshold=self.score_threshold,
            filter_conditions=filter_conditions,
        )

        logger.debug(f"Retrieved {len(documents)} documents for query: {query[:50]}...")
        return documents


class MultiQueryRetriever(Retriever):
    """Retriever that generates multiple query variations."""

    def __init__(
        self,
        store: QdrantStore,
        embedding_model: EmbeddingModel,
        collection: str,
        llm_client,
        top_k: int = 5,
        num_queries: int = 3,
    ):
        """Initialize multi-query retriever.

        Args:
            store: Qdrant store instance.
            embedding_model: Embedding model.
            collection: Collection to search.
            llm_client: LLM client for query generation.
            top_k: Documents per query.
            num_queries: Number of query variations.
        """
        super().__init__(store, embedding_model, collection, top_k)
        self.llm_client = llm_client
        self.num_queries = num_queries

    def _generate_queries(self, query: str) -> List[str]:
        """Generate query variations using LLM."""
        prompt = f"""Generate {self.num_queries} different versions of this question
to help find relevant documents. Return only the questions, one per line.

Original question: {query}

Alternative questions:"""

        response = self.llm_client.generate(prompt, max_tokens=200)
        queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
        return [query] + queries[: self.num_queries - 1]

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_conditions: Optional[dict] = None,
    ) -> List[Document]:
        """Retrieve using multiple query variations.

        Args:
            query: Original query.
            top_k: Documents to return.
            filter_conditions: Metadata filters.

        Returns:
            Deduplicated relevant documents.
        """
        queries = self._generate_queries(query)
        logger.debug(f"Generated {len(queries)} query variations")

        # Collect unique documents
        seen_ids = set()
        all_docs = []

        for q in queries:
            docs = super().retrieve(q, top_k=top_k, filter_conditions=filter_conditions)
            for doc in docs:
                if doc.id not in seen_ids:
                    seen_ids.add(doc.id)
                    all_docs.append(doc)

        # Sort by score and return top_k
        all_docs.sort(key=lambda d: d.score or 0, reverse=True)
        return all_docs[: top_k or self.top_k]


class HybridRetriever:
    """Hybrid retriever combining dense and sparse (keyword) search."""

    def __init__(
        self,
        dense_retriever: Retriever,
        dense_weight: float = 0.7,
        top_k: int = 5,
    ):
        """Initialize hybrid retriever.

        Args:
            dense_retriever: Dense vector retriever.
            dense_weight: Weight for dense results (0-1).
            top_k: Final number of documents.
        """
        self.dense_retriever = dense_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = 1.0 - dense_weight
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_conditions: Optional[dict] = None,
    ) -> List[Document]:
        """Retrieve using hybrid approach.

        Args:
            query: Query text.
            top_k: Number of results.
            filter_conditions: Metadata filters.

        Returns:
            Ranked documents.
        """
        # Get dense results
        dense_docs = self.dense_retriever.retrieve(
            query,
            top_k=(top_k or self.top_k) * 2,  # Over-fetch for fusion
            filter_conditions=filter_conditions,
        )

        # Re-score with weighted combination
        # (In production, add BM25 or keyword search here)
        for doc in dense_docs:
            doc.score = doc.score * self.dense_weight

        # Sort and return top_k
        dense_docs.sort(key=lambda d: d.score or 0, reverse=True)
        return dense_docs[: top_k or self.top_k]
