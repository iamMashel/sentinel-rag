import pytest
from unittest.mock import MagicMock, patch
from sentinel_rag.vector_db.store import VectorStore

@pytest.fixture
def mock_chroma_client():
    with patch("chromadb.PersistentClient") as mock:
        yield mock

@pytest.fixture
def vector_store(mock_chroma_client):
    return VectorStore(collection_name="test_collection", persistent_path="./test_db")

def test_initialization(vector_store, mock_chroma_client):
    mock_chroma_client.return_value.get_or_create_collection.assert_called_with(name="test_collection")

def test_add_documents(vector_store):
    documents = ["doc1", "doc2"]
    metadatas = [{"source": "1"}, {"source": "2"}]
    ids = ["1", "2"]
    embeddings = [[0.1, 0.2], [0.3, 0.4]]
    
    vector_store.add_documents(documents, metadatas, ids, embeddings)
    
    vector_store.collection.add.assert_called_with(
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
        ids=ids
    )

def test_add_documents_no_embeddings(vector_store):
    documents = ["doc1"]
    metadatas = [{"source": "1"}]
    ids = ["1"]
    
    vector_store.add_documents(documents, metadatas, ids)
    
    vector_store.collection.add.assert_called_with(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

def test_add_documents_error(vector_store):
    vector_store.collection.add.side_effect = Exception("DB Error")
    with pytest.raises(Exception):
        vector_store.add_documents(["doc"], [{}], ["1"])

def test_query(vector_store):
    query_embeddings = [[0.1, 0.2]]
    vector_store.query(query_embeddings, n_results=2)
    
    vector_store.collection.query.assert_called_with(
        query_embeddings=query_embeddings,
        n_results=2
    )

def test_count(vector_store):
    vector_store.collection.count.return_value = 5
    assert vector_store.count() == 5
