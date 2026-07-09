from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence, Mapping


@dataclass(frozen=True)
class DeepSeekConfig:
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    timeout_seconds: int = 60


def load_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def config_from_env(env_path: str | Path = ".env") -> DeepSeekConfig:
    load_dotenv(env_path)
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set. Please add it to .env or the environment.")
    return DeepSeekConfig(
        api_key=api_key,
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip(),
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip(),
    )


class DeepSeekClient:
    """Minimal OpenAI-compatible DeepSeek chat client using only stdlib."""

    def __init__(self, config: DeepSeekConfig):
        self.config = config

    def _debug_payload_path(self) -> Path:
        root = Path(__file__).resolve().parent.parent
        debug_dir = root / "logs"
        debug_dir.mkdir(parents=True, exist_ok=True)
        return debug_dir / "last_llm_payload.json"

    def _write_debug_payload(self, payload: dict) -> None:
        """
        保存最近一次真实发给 LLM 的请求体。
        注意：这里不保存 Authorization 请求头，所以不会暴露 API Key。
        """
        debug_payload = {
            "captured_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "base_url": self.config.base_url,
            "model": self.config.model,
            "payload": payload,
        }

        path = self._debug_payload_path()
        path.write_text(
            json.dumps(debug_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        payload = {
            "model": self.config.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        self._write_debug_payload(payload)

        request = urllib.request.Request(
            url=f"{self.config.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DeepSeek request failed with HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"DeepSeek request failed: {exc.reason}") from exc

        data = json.loads(body)
        try:
            return str(data["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected DeepSeek response shape: {body}") from exc