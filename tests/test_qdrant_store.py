"""QdrantStore against qdrant-client's in-memory mode — the real client code path, no server."""

from qdrant_client import QdrantClient

from rag_chatbot.retrieval.qdrant_store import Document, QdrantStore


def _memory_store(dim: int = 4) -> QdrantStore:
    store = QdrantStore.__new__(QdrantStore)   # skip the network constructor
    store.client = QdrantClient(":memory:")
    store.embedding_dim = dim
    return store


def test_add_then_search_returns_nearest_document_with_payload():
    store = _memory_store()
    store.create_collection("t")
    docs = [
        Document(content="apples", metadata={"topic": "fruit"}),
        Document(content="engines", metadata={"topic": "cars"}),
    ]
    store.add_documents("t", docs, [[1, 0, 0, 0], [0, 1, 0, 0]])

    hits = store.search("t", [0.9, 0.1, 0, 0], top_k=1)
    assert [h.content for h in hits] == ["apples"]
    assert hits[0].metadata == {"topic": "fruit"}
    assert hits[0].score is not None


def test_filter_conditions_restrict_results():
    store = _memory_store()
    store.create_collection("t")
    store.add_documents(
        "t",
        [Document(content="a", metadata={"lang": "de"}), Document(content="b", metadata={"lang": "en"})],
        [[1, 0, 0, 0], [0.9, 0.1, 0, 0]],
    )
    hits = store.search("t", [1, 0, 0, 0], top_k=5, filter_conditions={"lang": "en"})
    assert [h.content for h in hits] == ["b"]
