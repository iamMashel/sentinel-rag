import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, collection_name: str = "sentinel_rag_kb", persistent_path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persistent_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        logger.info(f"Initialized ChromaDB collection: {collection_name} at {persistent_path}")

    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str], embeddings: Optional[List[List[float]]] = None):
        """
        Adds documents to the collection.
        If embeddings are provided, they are used. Otherwise, ChromaDB's default embedding function (if configured) or the caller should handle it.
        Current design assumes embeddings are generated externally and passed in for better control, 
        or we can rely on Chroma's default if we don't pass them.
        """
        try:
            if embeddings:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings,
                    ids=ids
                )
            else:
                 self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            logger.info(f"Added {len(documents)} documents to collection.")
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    def query(self, query_embeddings: List[List[float]], n_results: int = 1) -> Dict[str, Any]:
        """
        Queries the collection using embeddings.
        """
        try:
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results
            )
            return results
        except Exception as e:
            logger.error(f"Failed to query collection: {e}")
            raise

    def count(self) -> int:
        return self.collection.count()
