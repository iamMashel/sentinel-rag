import pytest
import json
from unittest.mock import MagicMock, patch
from sentinel_rag.core.engine import SupportBot

@pytest.fixture
def mock_genai():
    with patch("sentinel_rag.core.engine.genai") as mock:
        yield mock

@pytest.fixture
def mock_vector_store():
    with patch("sentinel_rag.core.engine.VectorStore") as mock:
        yield mock

@pytest.fixture
def support_bot(mock_genai, mock_vector_store):
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test_key"}):
        return SupportBot()

def test_load_knowledge_base(support_bot, mock_genai):
    # Mock file loading
    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = '{"responses": [{"id": "1", "response_text": "Test"}]}'
        with patch("json.load", return_value={"responses": [{"id": "1", "response_text": "Test"}]}):
             # Mock embed_content to return correct structure
             mock_genai.embed_content.return_value = {'embedding': [[0.1]*768]}
             support_bot.load_knowledge_base("dummy.json")
    
    # Check if embeddings were generated using genai.embed_content
    mock_genai.embed_content.assert_called()
    # Check if documents were added to vector store
    support_bot.vector_store.add_documents.assert_called()

def test_get_response_knowledge_base(support_bot, mock_genai):
    # Mock vector store query result (high confidence match)
    support_bot.vector_store.query.return_value = {
        'ids': [['doc_1']],
        'distances': [[0.1]], # Low distance -> High similarity
        'documents': [['Test Answer']]
    }
    
    # Mock embedding generation for query
    # engine.py expects call to embed_content return dict with 'embedding'
    mock_genai.embed_content.return_value = {'embedding': [0.1]*768}
    
    response = support_bot.get_response("test query", threshold=0.8)
    
    assert response['source'] == 'knowledge_base'
    assert response['answer'] == 'Test Answer'

def test_get_response_fallback(support_bot, mock_genai):
    # Mock vector store query result (low confidence match)
    support_bot.vector_store.query.return_value = {
        'ids': [['doc_1']],
        'distances': [[10.0]], # High distance -> Low similarity
        'documents': [['Irrelevant']]
    }
    
    # Mock embedding generation
    mock_genai.embed_content.return_value = {'embedding': [0.1]*768}
    
    # Mock LLM completion
    mock_response = MagicMock()
    mock_response.text = "LLM Answer"
    support_bot.model.generate_content.return_value = mock_response
    
    response = support_bot.get_response("test query", threshold=0.8)
    
    assert response['source'] == 'llm_generation'
    assert response['answer'] == "LLM Answer"

def test_load_knowledge_base_file_not_found(support_bot):
    with patch("builtins.open", side_effect=FileNotFoundError):
        # Should catch exception and log error, not raise
        support_bot.load_knowledge_base("non_existent.json")
    
    # Verify no embeddings were generated
    support_bot.vector_store.add_documents.assert_not_called()

def test_load_knowledge_base_invalid_json(support_bot):
    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = 'INVALID JSON'
        # json.load would raise JSONDecodeError when reading from file object if we didn't mock it to return the string directly?
        # Actually json.load takes a file-like object. 
        with patch("json.load", side_effect=json.JSONDecodeError("msg", "doc", 0)):
             support_bot.load_knowledge_base("bad.json")
             
    support_bot.vector_store.add_documents.assert_not_called()

def test_get_response_api_error(support_bot, mock_genai):
    # Mock embedding failure
    mock_genai.embed_content.side_effect = Exception("API Down")
    
    response = support_bot.get_response("query")
    
    assert response['source'] == 'error'
    assert "error" in response['answer']

def test_generate_fallback_error(support_bot):
    # Mock LLM failure
    support_bot.model.generate_content.side_effect = Exception("LLM Error")
    
    result = support_bot._generate_fallback("query")
    assert "unable to generate" in result
