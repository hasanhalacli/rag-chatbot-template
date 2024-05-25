"""Qdrant vector store implementation."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import uuid4

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams


@dataclass
class Document:
    """Document with content and metadata."""

    content: str
    metadata: Dict[str, Any]
    id: Optional[str] = None
    score: Optional[float] = None


class QdrantStore:
    """Qdrant vector store for document storage and retrieval."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        https: bool = False,
        embedding_dim: int = 384,
    ):
        """Initialize Qdrant store.

        Args:
            host: Qdrant host.
            port: Qdrant port.
            api_key: API key for Qdrant Cloud.
            https: Use HTTPS.
            embedding_dim: Dimension of embeddings.
        """
        if api_key:
            self.client = QdrantClient(
                url=f"https://{host}" if https else f"http://{host}:{port}",
                api_key=api_key,
            )
        else:
            self.client = QdrantClient(host=host, port=port)

        self.embedding_dim = embedding_dim
        logger.info(f"Connected to Qdrant at {host}:{port}")

    def create_collection(
        self,
        name: str,
        distance: str = "cosine",
        recreate: bool = False,
    ) -> None:
        """Create a collection.

        Args:
            name: Collection name.
            distance: Distance metric (cosine, dot, euclidean).
            recreate: Delete existing collection if exists.
        """
        distance_map = {
            "cosine": Distance.COSINE,
            "dot": Distance.DOT,
            "euclidean": Distance.EUCLID,
        }

        if recreate:
            try:
                self.client.delete_collection(name)
                logger.info(f"Deleted existing collection: {name}")
            except Exception:
                pass

        try:
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=distance_map.get(distance, Distance.COSINE),
                ),
            )
            logger.info(f"Created collection: {name}")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(f"Collection already exists: {name}")
            else:
                raise

    def add_documents(
        self,
        collection: str,
        documents: List[Document],
        embeddings: List[List[float]],
        batch_size: int = 100,
    ) -> List[str]:
        """Add documents to collection.

        Args:
            collection: Collection name.
            documents: List of documents.
            embeddings: List of embedding vectors.
            batch_size: Batch size for upsert.

        Returns:
            List of document IDs.
        """
        ids = []

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]

            points = []
            for doc, embedding in zip(batch_docs, batch_embeddings):
                doc_id = doc.id or str(uuid4())
                ids.append(doc_id)

                points.append(
                    models.PointStruct(
                        id=doc_id,
                        vector=embedding,
                        payload={
                            "content": doc.content,
                            **doc.metadata,
                        },
                    )
                )

            self.client.upsert(collection_name=collection, points=points)

        logger.info(f"Added {len(documents)} documents to {collection}")
        return ids

    def search(
        self,
        collection: str,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Search for similar documents.

        Args:
            collection: Collection name.
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            score_threshold: Minimum score threshold.
            filter_conditions: Qdrant filter conditions.

        Returns:
            List of matching documents with scores.
        """
        # Build filter if provided
        query_filter = None
        if filter_conditions:
            must_conditions = []
            for key, value in filter_conditions.items():
                if isinstance(value, list):
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchAny(any=value),
                        )
                    )
                else:
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value),
                        )
                    )
            query_filter = models.Filter(must=must_conditions)

        results = self.client.search(
            collection_name=collection,
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=query_filter,
        )

        documents = []
        for hit in results:
            payload = hit.payload or {}
            content = payload.pop("content", "")
            documents.append(
                Document(
                    id=str(hit.id),
                    content=content,
                    metadata=payload,
                    score=hit.score,
                )
            )

        return documents

    def delete_collection(self, name: str) -> None:
        """Delete a collection.

        Args:
            name: Collection name.
        """
        self.client.delete_collection(name)
        logger.info(f"Deleted collection: {name}")

    def list_collections(self) -> List[str]:
        """List all collections.

        Returns:
            List of collection names.
        """
        collections = self.client.get_collections()
        return [c.name for c in collections.collections]

    def get_collection_info(self, name: str) -> Dict[str, Any]:
        """Get collection info.

        Args:
            name: Collection name.

        Returns:
            Collection info dictionary.
        """
        info = self.client.get_collection(name)
        return {
            "name": name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status,
        }
