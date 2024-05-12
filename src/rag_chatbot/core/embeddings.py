"""Embedding model wrapper for text vectorization."""

from typing import List, Optional, Union

import numpy as np
from loguru import logger


class EmbeddingModel:
    """Embedding model for text vectorization."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        batch_size: int = 32,
    ):
        """Initialize embedding model.

        Args:
            model_name: HuggingFace model name or path.
            device: Device to use ('auto', 'cpu', 'cuda').
            batch_size: Batch size for encoding.
        """
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size

        # Determine device
        if device == "auto":
            import torch

            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        logger.info(f"Loaded embedding model: {model_name} on {device}")
        logger.info(f"Embedding dimension: {self.embedding_dim}")

    def embed(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Embed texts into vectors.

        Args:
            texts: Single text or list of texts.
            normalize: Whether to L2-normalize embeddings.
            show_progress: Show progress bar for batches.

        Returns:
            Numpy array of embeddings (n_texts, embedding_dim).
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query.

        Args:
            query: Query text.

        Returns:
            1D numpy array of embedding.
        """
        return self.embed(query, normalize=True)[0]

    def embed_documents(
        self,
        documents: List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """Embed multiple documents.

        Args:
            documents: List of document texts.
            show_progress: Show progress bar.

        Returns:
            2D numpy array of embeddings.
        """
        return self.embed(documents, normalize=True, show_progress=show_progress)

    def similarity(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: np.ndarray,
    ) -> np.ndarray:
        """Compute cosine similarity between query and documents.

        Args:
            query_embedding: Query embedding (1D or 2D).
            doc_embeddings: Document embeddings (2D).

        Returns:
            Similarity scores.
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Cosine similarity (embeddings are already normalized)
        return np.dot(doc_embeddings, query_embedding.T).flatten()

    def get_dimension(self) -> int:
        """Get embedding dimension.

        Returns:
            Embedding dimension.
        """
        return self.embedding_dim
