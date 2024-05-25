"""Cross-encoder reranking for improved relevance."""

from typing import List, Optional

from loguru import logger

from .qdrant_store import Document


class CrossEncoderReranker:
    """Rerank documents using cross-encoder model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "auto",
        top_k: int = 3,
    ):
        """Initialize reranker.

        Args:
            model_name: Cross-encoder model name.
            device: Device for inference.
            top_k: Number of documents to return after reranking.
        """
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.top_k = top_k

        # Determine device
        if device == "auto":
            import torch

            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.model = CrossEncoder(model_name, device=device)
        logger.info(f"Loaded reranker: {model_name} on {device}")

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """Rerank documents by relevance to query.

        Args:
            query: Query text.
            documents: Documents to rerank.
            top_k: Number of documents to return.

        Returns:
            Reranked documents with updated scores.
        """
        if not documents:
            return []

        top_k = top_k or self.top_k

        # Prepare pairs for cross-encoder
        pairs = [(query, doc.content) for doc in documents]

        # Get scores
        scores = self.model.predict(pairs)

        # Update document scores
        for doc, score in zip(documents, scores):
            doc.score = float(score)

        # Sort by score and return top_k
        documents.sort(key=lambda d: d.score, reverse=True)

        logger.debug(f"Reranked {len(documents)} docs, returning top {top_k}")
        return documents[:top_k]
