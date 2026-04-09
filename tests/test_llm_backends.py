from __future__ import annotations
import os
import pytest
import requests
from unittest.mock import patch, MagicMock
from src.agents.llm_backends import LLMBackend, OllamaBackend, build_llm_backend


def test_llm_backend_is_abstract():
    """LLMBackend는 직접 인스턴스화할 수 없다."""
    with pytest.raises(TypeError):
        LLMBackend()


def test_ollama_backend_name():
    b = OllamaBackend(model="qwen3:4b", host="http://localhost:11434", timeout=5)
    assert b.name() == "ollama/qwen3:4b"


def test_ollama_backend_generate_success():
    """Ollama API 정상 응답 시 response 텍스트를 반환한다."""
    b = OllamaBackend(model="qwen3:4b", host="http://localhost:11434", timeout=5)
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": "KEEP: 이상 확인됨"}
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_resp) as mock_post:
        result = b.generate("테스트 프롬프트")

    assert result == "KEEP: 이상 확인됨"
    mock_post.assert_called_once_with(
        "http://localhost:11434/api/generate",
        json={"model": "qwen3:4b", "prompt": "테스트 프롬프트", "stream": False},
        timeout=5,
    )


def test_ollama_backend_generate_raises_on_error():
    """Ollama 호출 실패 시 예외를 그대로 전파한다."""
    b = OllamaBackend(model="qwen3:4b", host="http://localhost:11434", timeout=5)
    with patch("requests.post", side_effect=ConnectionError("refused")):
        with pytest.raises(ConnectionError):
            b.generate("프롬프트")


def test_ollama_backend_generate_raises_on_http_error():
    """HTTP 에러(4xx/5xx) 시 HTTPError를 전파한다."""
    b = OllamaBackend(model="qwen3:4b", host="http://localhost:11434", timeout=5)
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
    with patch("requests.post", return_value=mock_resp):
        with pytest.raises(requests.HTTPError):
            b.generate("프롬프트")


def test_openai_backend_name():
    pytest.importorskip("openai")
    from src.agents.llm_backends import OpenAIBackend
    b = OpenAIBackend(model="gpt-4o", api_key="sk-test")
    assert b.name() == "openai/gpt-4o"


def test_openai_backend_generate_success():
    """OpenAI API 정상 응답 시 텍스트를 반환한다."""
    pytest.importorskip("openai")
    from src.agents.llm_backends import OpenAIBackend
    b = OpenAIBackend(model="gpt-4o", api_key="sk-test")

    mock_message = MagicMock()
    mock_message.content = "KEEP: 복수 신호 동시 급변"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch("openai.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_cls.return_value = mock_client

        result = b.generate("테스트 프롬프트")

    assert result == "KEEP: 복수 신호 동시 급변"
    mock_client.chat.completions.create.assert_called_once_with(
        model="gpt-4o",
        max_tokens=256,
        messages=[{"role": "user", "content": "테스트 프롬프트"}],
    )
    mock_cls.assert_called_once_with(api_key="sk-test", timeout=60)


def test_openai_backend_generate_raises_on_error():
    """OpenAI API 호출 실패 시 예외를 그대로 전파한다."""
    pytest.importorskip("openai")
    from src.agents.llm_backends import OpenAIBackend
    b = OpenAIBackend(model="gpt-4o", api_key="sk-test")
    with patch("openai.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        mock_cls.return_value = mock_client

        with pytest.raises(Exception, match="API error"):
            b.generate("프롬프트")


# ── build_llm_backend() factory 테스트 ──────────────────────────────────────

def test_build_llm_backend_disabled():
    cfg = {"backend": "disabled"}
    assert build_llm_backend(cfg) is None


def test_build_llm_backend_ollama():
    cfg = {
        "backend": "ollama",
        "ollama": {"model": "qwen3:4b", "host": "http://localhost:11434", "timeout": 10},
    }
    b = build_llm_backend(cfg)
    assert isinstance(b, OllamaBackend)
    assert b.name() == "ollama/qwen3:4b"


def test_build_llm_backend_openai():
    from src.agents.llm_backends import OpenAIBackend
    cfg = {
        "backend": "openai",
        "openai": {"model": "gpt-4o", "api_key_env": "OPENAI_API_KEY", "timeout": 60},
    }
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
        b = build_llm_backend(cfg)
    assert isinstance(b, OpenAIBackend)
    assert b.name() == "openai/gpt-4o"
    assert b.api_key == "sk-test-key"


def test_build_llm_backend_unknown_falls_back_to_none():
    cfg = {"backend": "unknown_backend"}
    assert build_llm_backend(cfg) is None
