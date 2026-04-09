"""LLM 백엔드 추상화 — 교체 가능한 LLM 연결 인터페이스."""
from __future__ import annotations
import os
from abc import ABC, abstractmethod

import requests


class LLMBackend(ABC):
    """LLM 호출 추상 인터페이스. 모든 백엔드가 구현해야 한다."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """프롬프트를 보내고 텍스트 응답을 반환한다. 오류 시 예외를 발생시킨다."""

    @abstractmethod
    def name(self) -> str:
        """'backend/model' 형식의 식별자를 반환한다."""


class OllamaBackend(LLMBackend):
    """로컬 Ollama HTTP API 백엔드."""

    def __init__(
        self,
        model: str = "qwen3:4b",
        host: str = "http://localhost:11434",
        timeout: int = 30,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        resp = requests.post(
            f"{self.host}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if "response" not in data:
            raise ValueError(f"Ollama 응답에 'response' 키가 없음: {data}")
        return data["response"]

    def name(self) -> str:
        return f"ollama/{self.model}"


class OpenAIBackend(LLMBackend):
    """OpenAI GPT API 백엔드 (사내망/클라우드 사용)."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        timeout: int = 60,
        max_tokens: int = 256,
    ):
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        import openai
        client = openai.OpenAI(api_key=self.api_key, timeout=self.timeout)
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if not response.choices:
            raise ValueError(f"OpenAI 응답에 choices가 없음: {response}")
        content = response.choices[0].message.content
        if content is None:
            raise ValueError(
                f"OpenAI 응답 content가 None입니다 (finish_reason={response.choices[0].finish_reason!r})"
            )
        return content

    def name(self) -> str:
        return f"openai/{self.model}"


def build_llm_backend(llm_cfg: dict) -> "LLMBackend | None":
    """settings.yaml llm 섹션 dict로 적합한 LLMBackend를 생성한다.

    반환값이 None이면 LLMFilter가 비활성화된다.
    """
    backend_type = (llm_cfg.get("backend") or "disabled").lower()

    if backend_type == "ollama":
        cfg = llm_cfg.get("ollama") or {}
        return OllamaBackend(
            model=cfg.get("model", "qwen3:4b"),
            host=cfg.get("host", "http://localhost:11434"),
            timeout=int(cfg.get("timeout", 30)),
        )

    if backend_type == "openai":
        cfg = llm_cfg.get("openai") or {}
        api_key_env = cfg.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.environ.get(api_key_env)
        return OpenAIBackend(
            model=cfg.get("model", "gpt-4o"),
            api_key=api_key,
            timeout=int(cfg.get("timeout", 60)),
            max_tokens=int(cfg.get("max_tokens", 256)),
        )

    return None  # "disabled" 및 알 수 없는 값
