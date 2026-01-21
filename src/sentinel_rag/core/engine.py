import logging
import json
import os
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
import google.generativeai as genai
from dotenv import load_dotenv

from sentinel_rag.vector_db.store import VectorStore

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SupportBot:
    """
    A Retrieval-Augmented Generation (RAG) system for automated customer support using Google Gemini.
    """
    def __init__(self, model_name: str = "gemini-pro", embedding_model: str = "models/text-embedding-004", collection_name: str = "sentinel_rag_kb"):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found in environment variables.")
        else:
            genai.configure(api_key=self.api_key)
        
        self.model = genai.GenerativeModel(model_name)
        self.embedding_model = embedding_model
        
        # Initialize Vector Store
        self.vector_store = VectorStore(collection_name=collection_name)
        
    def load_knowledge_base(self, file_path: str):
        """Loads and prepares the local knowledge base."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Search for 'responses' key or assume list/dict
            items = data.get('responses', data) if isinstance(data, dict) else data
            
            documents = []
            metadatas = []
            ids = []
            
            for idx, item in enumerate(items):
                text = item.get('response_text', str(item)) if isinstance(item, dict) else str(item)
                documents.append(text)
                metadatas.append({"source": "json_file", "original_index": idx})
                ids.append(f"doc_{idx}")

            if not documents:
                logger.warning("No documents found to load.")
                return

            logger.info(f"Generating embeddings for {len(documents)} items...")
            embeddings = self._get_embeddings(documents)
            
            if embeddings:
                # Store in ChromaDB
                self.vector_store.add_documents(documents=documents, metadatas=metadatas, ids=ids, embeddings=embeddings)
                logger.info("Knowledge base initialized and stored in ChromaDB.")
            else:
                logger.error("Failed to generate embeddings, KB not initialized.")
            
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Helper to fetch embeddings using Gemini."""
        embeddings = []
        try:
            # Gemini embedding API often processes batch, but let's be safe.
            # "models/embedding-001" or "models/text-embedding-004" accepts 'content' as list or string.
            # However, output format might differ. 
            result = genai.embed_content(
                model=self.embedding_model,
                content=texts,
                task_type="retrieval_document",
                title="Knowledge Base Document" # Optional but good for 004
            )
            # Result is usually a dict with 'embedding' key which is list of list (if batch)
            # Verify structure: {'embedding': [[...], [...]]}
            if 'embedding' in result:
                return result['embedding']
            else:
                # Provide fallback or specific handling if response format varies
                logger.error(f"Unexpected embedding response format: {result.keys()}")
                return []
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return []

    def get_response(self, query: str, threshold: float = 0.5) -> Dict[str, Any]:
        """Core logic: Retrieve from KB or Fallback to LLM."""
        try:
            # Embed query
            query_embedding_result = genai.embed_content(
                model=self.embedding_model,
                content=query,
                task_type="retrieval_query"
            )
            
            query_vector = query_embedding_result['embedding']
            if not query_vector:
                raise ValueError("Failed to embed query")
                
            # Vector search
            # Note: ChromaDB queries usually require list of lists for embedding
            results = self.vector_store.query(query_embeddings=[query_vector], n_results=1)
            
            confidence = 0.0
            answer = None
            source = "unknown"

            if results['distances'] and len(results['distances'][0]) > 0:
                distance = results['distances'][0][0]
                # Approximation of similarity for thresholding logic
                similarity = 1 - (distance**2) / 2
                confidence = float(similarity)
                
                # Gemini embeddings might behave differently with L2 distance, so may need tuning.
                # Adjust threshold or logic if needed. 
                if confidence > threshold:
                    source = "knowledge_base"
                    answer = results['documents'][0][0]
                else:
                    source = "llm_generation"
                    answer = self._generate_fallback(query)
            else:
                 source = "llm_generation"
                 answer = self._generate_fallback(query)
                
            return {
                "query": query,
                "answer": answer,
                "confidence": round(confidence, 4),
                "source": source,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting response: {e}")
            return {
                "query": query,
                "answer": "Sorry, I encountered an error processing your request.",
                "confidence": 0.0,
                "source": "error"
            }

    def _generate_fallback(self, query: str) -> str:
        try:
            # Generate content
            response = self.model.generate_content(
                f"You are a professional support agent for ChatSolveAI. User Query: {query}"
            )
            return response.text
        except Exception as e:
            logger.error(f"LLM fallback failed: {e}")
            return "I am currently unable to generate a response. Please try again later."
