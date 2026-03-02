"""
CortexFlow LLM Client module.

Provides a unified interface for interacting with LLM backends (Ollama, Vertex AI/Gemini).
"""

import json
import logging
import os
from abc import abstractmethod
from typing import Dict, Iterator, List, Optional

import requests

from cortexflow.interfaces import LLMProviderInterface

logger = logging.getLogger("cortexflow")


class LLMClient(LLMProviderInterface):
    """Abstract base for LLM clients."""

    @abstractmethod
    def generate(self, messages: List[Dict], model: str = None, **kwargs) -> str:
        """Generate a response from a list of role/content messages."""

    @abstractmethod
    def generate_from_prompt(self, prompt: str, model: str = None, **kwargs) -> str:
        """Generate a response from a raw string prompt."""

    @abstractmethod
    def generate_stream(self, messages: List[Dict], model: str = None, **kwargs) -> Iterator[str]:
        """Stream a response from a list of role/content messages."""

    def batch_generate_from_prompts(self, prompts: List[str], model: str = None) -> List[str]:
        """Generate responses for multiple prompts sequentially."""
        return [self.generate_from_prompt(p, model=model) for p in prompts]


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------

class OllamaClient(LLMClient):
    """LLM client that talks to a local Ollama instance (preserves existing behavior)."""

    def __init__(self, ollama_host: str, default_model: str):
        self.ollama_host = ollama_host.rstrip("/")
        self.default_model = default_model

    def generate(self, messages: List[Dict], model: str = None, **kwargs) -> str:
        model = model or self.default_model
        url = f"{self.ollama_host}/api/chat"
        payload = {"model": model, "messages": messages, "stream": False}
        try:
            resp = requests.post(url, json=payload, timeout=kwargs.get("timeout", 30))
            if resp.status_code == 200:
                return resp.json()["message"]["content"]
            return f"Error: {resp.status_code} - {resp.text}"
        except Exception as exc:
            logger.error(f"OllamaClient.generate error: {exc}")
            return f"Error: {exc}"

    def generate_from_prompt(self, prompt: str, model: str = None, **kwargs) -> str:
        model = model or self.default_model
        url = f"{self.ollama_host}/api/generate"
        payload = {"model": model, "prompt": prompt, "stream": False}
        try:
            resp = requests.post(url, json=payload, timeout=kwargs.get("timeout", 30))
            if resp.status_code == 200:
                return resp.json().get("response", "")
            return f"Error: {resp.status_code} - {resp.text}"
        except Exception as exc:
            logger.error(f"OllamaClient.generate_from_prompt error: {exc}")
            return f"Error: {exc}"

    def generate_stream(self, messages: List[Dict], model: str = None, **kwargs) -> Iterator[str]:
        model = model or self.default_model
        url = f"{self.ollama_host}/api/chat"
        payload = {"model": model, "messages": messages, "stream": True}
        try:
            resp = requests.post(url, json=payload, timeout=kwargs.get("timeout", 30), stream=True)
            if resp.status_code == 200:
                for line in resp.iter_lines():
                    if line:
                        chunk_data = json.loads(line)
                        chunk = (
                            chunk_data.get("message", {}).get("content", "")
                            or chunk_data.get("response", "")
                        )
                        if chunk:
                            yield chunk
            else:
                yield f"Error: {resp.status_code} - {resp.text}"
        except Exception as exc:
            logger.error(f"OllamaClient.generate_stream error: {exc}")
            yield f"Error: {exc}"

    def batch_generate_from_prompts(self, prompts: List[str], model: str = None) -> List[str]:
        model = model or self.default_model
        results = []
        url = f"{self.ollama_host}/api/generate"
        try:
            with requests.Session() as session:
                for prompt in prompts:
                    resp = session.post(
                        url,
                        json={"model": model, "prompt": prompt, "stream": False},
                        timeout=30,
                    )
                    if resp.status_code == 200:
                        results.append(resp.json().get("response", ""))
                    else:
                        results.append(f"Error: {resp.status_code}")
        except Exception as exc:
            logger.error(f"OllamaClient.batch_generate error: {exc}")
            results.extend([""] * (len(prompts) - len(results)))
        return results


# ---------------------------------------------------------------------------
# Vertex AI / Gemini client
# ---------------------------------------------------------------------------

class VertexAIClient(LLMClient):
    """LLM client backed by Vertex AI / Gemini models via google-genai SDK.

    Auth priority:
      1. GOOGLE_APPLICATION_CREDENTIALS / config.vertex_credentials_path → service account
         (uses google.genai.Client with vertexai=True and scoped credentials)
      2. Application Default Credentials (vertexai=True, ADC)
    """

    _SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

    def __init__(self, config):
        """
        Args:
            config: CortexFlowConfig (or just config.llm).
                    Accepts either the full config or the llm sub-config.
        """
        llm = getattr(config, "llm", config)

        self.project_id = (
            getattr(llm, "vertex_project_id", None)
            or os.environ.get("VERTEX_PROJECT_ID", "")
        )
        raw_location = (
            getattr(llm, "vertex_location", None)
            or os.environ.get("VERTEX_LOCATION", "us-central1")
        )
        # "global" is not a valid Vertex AI region — map to us-central1
        self.location = raw_location if raw_location != "global" else "us-central1"

        credentials_path = (
            getattr(llm, "vertex_credentials_path", None)
            or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        )
        self.default_model = (
            getattr(llm, "vertex_model", None)
            or getattr(llm, "default_model", "gemini-2.0-flash")
        )
        # Remap legacy model names that aren't available
        if self.default_model in ("gemini-1.5-flash", "gemini-1.5-pro"):
            self.default_model = "gemini-2.0-flash"

        self._genai_client = self._build_client(credentials_path)

    def _build_client(self, credentials_path: str):
        """Build and return a google.genai.Client configured for Vertex AI."""
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "google-genai package not installed. Run: pip install google-genai"
            ) from exc

        if credentials_path and os.path.isfile(credentials_path):
            try:
                from google.oauth2 import service_account
                creds = service_account.Credentials.from_service_account_file(
                    credentials_path, scopes=self._SCOPES
                )
                logger.info(
                    f"VertexAIClient: service account auth "
                    f"(project={self.project_id}, location={self.location})"
                )
                return genai.Client(
                    vertexai=True,
                    project=self.project_id or None,
                    location=self.location,
                    credentials=creds,
                )
            except Exception as exc:
                logger.warning(f"VertexAIClient: SA auth failed ({exc}), falling back to ADC")

        # Application Default Credentials
        logger.info(
            f"VertexAIClient: ADC auth "
            f"(project={self.project_id}, location={self.location})"
        )
        from google import genai
        return genai.Client(
            vertexai=True,
            project=self.project_id or None,
            location=self.location,
        )

    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """Convert OpenAI-style messages list to a single prompt string."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"[System]: {content}")
            elif role == "assistant":
                parts.append(f"[Assistant]: {content}")
            else:
                parts.append(f"[User]: {content}")
        return "\n\n".join(parts)

    def generate(self, messages: List[Dict], model: str = None, **kwargs) -> str:
        prompt = self._messages_to_prompt(messages)
        return self.generate_from_prompt(prompt, model=model, **kwargs)

    def generate_from_prompt(self, prompt: str, model: str = None, **kwargs) -> str:
        model = model or self.default_model
        try:
            response = self._genai_client.models.generate_content(
                model=model, contents=prompt
            )
            return response.text
        except Exception as exc:
            logger.error(f"VertexAIClient.generate_from_prompt error: {exc}")
            return f"Error: {exc}"

    def generate_stream(self, messages: List[Dict], model: str = None, **kwargs) -> Iterator[str]:
        model = model or self.default_model
        prompt = self._messages_to_prompt(messages)
        try:
            for chunk in self._genai_client.models.generate_content_stream(
                model=model, contents=prompt
            ):
                try:
                    text = chunk.text
                    if text:
                        yield text
                except Exception:
                    pass
        except Exception as exc:
            logger.error(f"VertexAIClient.generate_stream error: {exc}")
            yield f"Error: {exc}"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_llm_client(config) -> LLMClient:
    """Create and return the appropriate LLM client based on config/env."""
    # Support passing either the full config or just config.llm
    llm = getattr(config, "llm", config)

    backend = (
        getattr(llm, "backend", None)
        or os.environ.get("CORTEXFLOW_LLM_BACKEND", "ollama")
    )

    if backend == "vertex_ai":
        logger.info("LLM backend: Vertex AI / Gemini")
        return VertexAIClient(config)

    # Default: Ollama
    ollama_host = getattr(llm, "ollama_host", "http://localhost:11434")
    default_model = getattr(llm, "default_model", "gemma3:1b")
    logger.info(f"LLM backend: Ollama ({ollama_host}, model={default_model})")
    return OllamaClient(ollama_host=ollama_host, default_model=default_model)
