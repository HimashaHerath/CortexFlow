"""
Tests for the CortexFlow LLM Client module.

Covers the factory function, OllamaClient URL construction, VertexAIClient
model remapping, and mocked API calls.
"""

from unittest.mock import MagicMock, patch

from cortexflow.config import ConfigBuilder, CortexFlowConfig, LLMConfig
from cortexflow.llm_client import (
    LLMClient,
    OllamaClient,
    VertexAIClient,
    create_llm_client,
)

# ---------------------------------------------------------------------------
# create_llm_client factory
# ---------------------------------------------------------------------------

class TestCreateLLMClient:
    """Test the factory function that returns the correct client."""

    def test_default_config_returns_ollama_client(self):
        config = CortexFlowConfig()
        client = create_llm_client(config)
        assert isinstance(client, OllamaClient)

    def test_ollama_backend_returns_ollama_client(self):
        config = ConfigBuilder().with_llm(backend="ollama").build()
        client = create_llm_client(config)
        assert isinstance(client, OllamaClient)

    @patch("cortexflow.llm_client.VertexAIClient")
    def test_vertex_ai_backend_returns_vertex_client(self, mock_vertex_cls):
        mock_vertex_cls.return_value = MagicMock(spec=LLMClient)
        config = ConfigBuilder().with_vertex_ai(project_id="test-project").build()
        create_llm_client(config)
        mock_vertex_cls.assert_called_once_with(config)

    def test_accepts_llm_sub_config_directly(self):
        llm_config = LLMConfig(backend="ollama", ollama_host="http://myhost:1234")
        client = create_llm_client(llm_config)
        assert isinstance(client, OllamaClient)
        assert client.ollama_host == "http://myhost:1234"


# ---------------------------------------------------------------------------
# OllamaClient
# ---------------------------------------------------------------------------

class TestOllamaClient:
    """Test OllamaClient behavior."""

    def test_url_construction_for_generate(self):
        client = OllamaClient(ollama_host="http://localhost:11434", default_model="gemma3:1b")
        # Verify the host is stored without trailing slash
        assert client.ollama_host == "http://localhost:11434"

    def test_trailing_slash_is_stripped(self):
        client = OllamaClient(ollama_host="http://localhost:11434/", default_model="test")
        assert client.ollama_host == "http://localhost:11434"

    def test_default_model_is_stored(self):
        client = OllamaClient(ollama_host="http://localhost:11434", default_model="llama3:8b")
        assert client.default_model == "llama3:8b"

    @patch("cortexflow.llm_client.requests.post")
    def test_generate_from_prompt_sends_correct_payload(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Hello world"}
        mock_post.return_value = mock_response

        client = OllamaClient(ollama_host="http://localhost:11434", default_model="gemma3:1b")
        result = client.generate_from_prompt("Say hello")

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]["json"]["prompt"] == "Say hello"
        assert call_args[1]["json"]["model"] == "gemma3:1b"
        assert "http://localhost:11434/api/generate" in call_args[0][0]
        assert result == "Hello world"

    @patch("cortexflow.llm_client.requests.post")
    def test_generate_sends_to_chat_endpoint(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": {"content": "Hi there"}}
        mock_post.return_value = mock_response

        client = OllamaClient(ollama_host="http://localhost:11434", default_model="gemma3:1b")
        result = client.generate([{"role": "user", "content": "Hello"}])

        mock_post.assert_called_once()
        call_url = mock_post.call_args[0][0]
        assert "/api/chat" in call_url
        assert result == "Hi there"

    @patch("cortexflow.llm_client.requests.post")
    def test_generate_from_prompt_handles_error_status(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        client = OllamaClient(ollama_host="http://localhost:11434", default_model="gemma3:1b")
        result = client.generate_from_prompt("test")

        assert "Error" in result

    @patch("cortexflow.llm_client.requests.post")
    def test_generate_from_prompt_handles_exception(self, mock_post):
        mock_post.side_effect = ConnectionError("Connection refused")

        client = OllamaClient(ollama_host="http://localhost:11434", default_model="gemma3:1b")
        result = client.generate_from_prompt("test")

        assert "Error" in result

    @patch("cortexflow.llm_client.requests.post")
    def test_model_override_in_generate_from_prompt(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "OK"}
        mock_post.return_value = mock_response

        client = OllamaClient(ollama_host="http://localhost:11434", default_model="gemma3:1b")
        client.generate_from_prompt("test", model="llama3:8b")

        sent_model = mock_post.call_args[1]["json"]["model"]
        assert sent_model == "llama3:8b"


# ---------------------------------------------------------------------------
# VertexAIClient
# ---------------------------------------------------------------------------

class TestVertexAIClient:
    """Test VertexAIClient configuration and model remapping."""

    @patch("cortexflow.llm_client.VertexAIClient._build_client")
    def test_gemini_15_flash_remapped_to_20_flash(self, mock_build):
        mock_build.return_value = MagicMock()
        config = ConfigBuilder().with_vertex_ai(
            project_id="test",
            default_model="gemini-1.5-flash",
        ).build()

        client = VertexAIClient(config)
        assert client.default_model == "gemini-2.0-flash"

    @patch("cortexflow.llm_client.VertexAIClient._build_client")
    def test_gemini_15_pro_remapped_to_20_flash(self, mock_build):
        mock_build.return_value = MagicMock()
        config = ConfigBuilder().with_vertex_ai(
            project_id="test",
            default_model="gemini-1.5-pro",
        ).build()

        client = VertexAIClient(config)
        assert client.default_model == "gemini-2.0-flash"

    @patch("cortexflow.llm_client.VertexAIClient._build_client")
    def test_global_location_remapped_to_us_central1(self, mock_build):
        mock_build.return_value = MagicMock()
        config = ConfigBuilder().with_vertex_ai(
            project_id="test",
            location="global",
        ).build()

        client = VertexAIClient(config)
        assert client.location == "us-central1"

    @patch("cortexflow.llm_client.VertexAIClient._build_client")
    def test_valid_location_not_remapped(self, mock_build):
        mock_build.return_value = MagicMock()
        config = ConfigBuilder().with_vertex_ai(
            project_id="test",
            location="us-east1",
        ).build()

        client = VertexAIClient(config)
        assert client.location == "us-east1"

    @patch("cortexflow.llm_client.VertexAIClient._build_client")
    def test_project_id_from_config(self, mock_build):
        mock_build.return_value = MagicMock()
        config = ConfigBuilder().with_vertex_ai(project_id="my-gcp-project").build()

        client = VertexAIClient(config)
        assert client.project_id == "my-gcp-project"

    @patch("cortexflow.llm_client.VertexAIClient._build_client")
    def test_messages_to_prompt_conversion(self, mock_build):
        mock_build.return_value = MagicMock()
        config = ConfigBuilder().with_vertex_ai(project_id="test").build()

        client = VertexAIClient(config)
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        prompt = client._messages_to_prompt(messages)
        assert "[System]: Be helpful." in prompt
        assert "[User]: Hello!" in prompt
        assert "[Assistant]: Hi there!" in prompt

    @patch("cortexflow.llm_client.VertexAIClient._build_client")
    def test_generate_delegates_to_generate_from_prompt(self, mock_build):
        mock_genai_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Generated response"
        mock_genai_client.models.generate_content.return_value = mock_response
        mock_build.return_value = mock_genai_client

        config = ConfigBuilder().with_vertex_ai(project_id="test").build()
        client = VertexAIClient(config)

        result = client.generate([{"role": "user", "content": "Hello"}])
        assert result == "Generated response"
