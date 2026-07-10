from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence, Mapping


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


def _assert_no_confidential_prompt(messages: Sequence[Mapping[str, str]]) -> None:
    """P0 安全断言：如果 prompt 中包含保密库标记，直接中断请求。

    阻断标记（完整列表）：
    - SEC-（保密事实 ID 前缀）
    - 保密内容= / 保密关键词= / 保密类别= / 保密摘要= / 保密事实库
    - 密级= / 机密★ / 绝密★ / 秘密★
    - 【内部资产台账】（保密数据段的标题）
    """
    joined = "\n".join(m.get("content", "") for m in messages)
    forbidden = [
        "SEC-",
        "保密内容=", "保密关键词=", "保密类别=", "保密摘要=",
        "保密事实库",
        "密级=", "机密★", "绝密★", "秘密★",
        "【内部资产台账】",
    ]
    hit = [x for x in forbidden if x in joined]
    if hit:
        raise RuntimeError(f"Blocked confidential prompt leakage: {hit}")


class DeepSeekClient:
    """Minimal OpenAI-compatible DeepSeek chat client using only stdlib."""

    def __init__(self, config: DeepSeekConfig):
        self.config = config

    def _debug_payload_path(self) -> Path:
        root = Path(__file__).resolve().parent.parent
        debug_dir = root / "logs"
        debug_dir.mkdir(parents=True, exist_ok=True)
        return debug_dir / "last_llm_payload.json"

    def _debug_payload_dir(self, request_id: str) -> Path:
        root = Path(__file__).resolve().parent.parent
        safe_request_id = _safe_debug_name(request_id or "unknown")
        debug_dir = root / "logs" / "llm_payloads" / safe_request_id
        debug_dir.mkdir(parents=True, exist_ok=True)
        return debug_dir

    def _write_debug_payload(
        self,
        payload: dict,
        debug_metadata: Mapping[str, Any] | None = None,
    ) -> dict:
        """
        保存真实发给 LLM 的请求体。
        注意：这里不保存 Authorization 请求头，所以不会暴露 API Key。
        """
        metadata = dict(debug_metadata or {})
        metadata.pop("trace", None)
        request_id = str(metadata.get("request_id") or "")
        purpose = str(metadata.get("purpose") or "llm_call")
        captured_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        debug_payload = {
            "captured_at": captured_at,
            "request_id": request_id,
            "purpose": purpose,
            "base_url": self.config.base_url,
            "model": self.config.model,
            "payload": payload,
        }
        for key, value in metadata.items():
            if key not in debug_payload:
                debug_payload[key] = value

        if request_id:
            debug_dir = self._debug_payload_dir(request_id)
            debug_dir.mkdir(parents=True, exist_ok=True)
            existing = sorted(debug_dir.glob("*.json"))
            call_index = len(existing) + 1
            call_id = str(metadata.get("call_id") or f"{request_id}_{purpose}_{call_index:03d}")
            debug_payload["call_id"] = call_id
            path = debug_dir / f"{_safe_debug_name(purpose)}_{call_index:03d}.json"
        else:
            debug_payload["call_id"] = str(metadata.get("call_id") or purpose)
            path = self._debug_payload_path()

        path.write_text(
            json.dumps(debug_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 兼容旧调试入口，同时新的前端会按 request_id 精确读取。
        self._debug_payload_path().write_text(
            json.dumps(debug_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return debug_payload

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 512,
        debug_metadata: Mapping[str, Any] | None = None,
    ) -> str:
        debug_metadata = debug_metadata or {}
        trace = debug_metadata.get("trace") if isinstance(debug_metadata, Mapping) else None
        purpose = str(debug_metadata.get("purpose") or "llm_call")
        call_id = str(debug_metadata.get("call_id") or purpose)

        # P0 安全断言：在发送外部 LLM 请求前，拦截 prompt 中的泄密内容
        _assert_no_confidential_prompt(messages)

        payload = {
            "model": self.config.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if trace is not None:
            clean_metadata = {
                key: value
                for key, value in dict(debug_metadata).items()
                if key != "trace"
            }
            trace.snapshot(
                f"{call_id}_request",
                {
                    "call_id": call_id,
                    "purpose": purpose,
                    "model": self.config.model,
                    "base_url": self.config.base_url,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "messages": list(messages),
                    "debug_metadata": clean_metadata,
                },
                component="DeepSeekClient",
                stage="llm_request",
                directory="llm",
                sensitivity="confidential",
            )
            trace.emit(
                component="DeepSeekClient",
                stage="llm_request",
                event_type="start",
                payload={
                    "call_id": call_id,
                    "purpose": purpose,
                    "model": self.config.model,
                },
            )

        if os.environ.get("CFA_DEBUG_LLM_PAYLOAD", "0") == "1":
            self._write_debug_payload(payload, debug_metadata=debug_metadata)

        request = urllib.request.Request(
            url=f"{self.config.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                http_status = response.status
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            detail = exc.read().decode("utf-8", errors="replace")
            if trace is not None:
                trace.snapshot(
                    f"{call_id}_error",
                    {
                        "call_id": call_id,
                        "purpose": purpose,
                        "http_status": exc.code,
                        "latency_ms": latency_ms,
                        "error_type": "HTTPError",
                        "error_message": detail,
                    },
                    component="DeepSeekClient",
                    stage="llm_error",
                    directory="llm",
                    sensitivity="restricted",
                )
                trace.emit(
                    component="DeepSeekClient",
                    stage="llm_request",
                    event_type="error",
                    status="failed",
                    elapsed_ms=latency_ms,
                    payload={
                        "call_id": call_id,
                        "http_status": exc.code,
                        "error_type": "HTTPError",
                    },
                )
            raise RuntimeError(f"DeepSeek request failed with HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            if trace is not None:
                trace.snapshot(
                    f"{call_id}_error",
                    {
                        "call_id": call_id,
                        "purpose": purpose,
                        "latency_ms": latency_ms,
                        "error_type": "URLError",
                        "error_message": str(exc.reason),
                    },
                    component="DeepSeekClient",
                    stage="llm_error",
                    directory="llm",
                    sensitivity="restricted",
                )
                trace.emit(
                    component="DeepSeekClient",
                    stage="llm_request",
                    event_type="error",
                    status="failed",
                    elapsed_ms=latency_ms,
                    payload={
                        "call_id": call_id,
                        "error_type": "URLError",
                        "error_message": str(exc.reason),
                    },
                )
            raise RuntimeError(f"DeepSeek request failed: {exc.reason}") from exc

        latency_ms = (time.perf_counter() - started) * 1000
        data = json.loads(body)
        try:
            content = str(data["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            if trace is not None:
                trace.snapshot(
                    f"{call_id}_unexpected_response",
                    {
                        "call_id": call_id,
                        "purpose": purpose,
                        "http_status": http_status,
                        "latency_ms": latency_ms,
                        "raw_response": data,
                    },
                    component="DeepSeekClient",
                    stage="llm_response_parse",
                    directory="llm",
                    sensitivity="restricted",
                )
            raise RuntimeError(f"Unexpected DeepSeek response shape: {body}") from exc

        if trace is not None:
            trace.snapshot(
                f"{call_id}_response",
                {
                    "call_id": call_id,
                    "purpose": purpose,
                    "http_status": http_status,
                    "latency_ms": latency_ms,
                    "raw_response": data,
                    "content": content,
                },
                component="DeepSeekClient",
                stage="llm_response",
                directory="llm",
                sensitivity="confidential",
            )
            trace.emit(
                component="DeepSeekClient",
                stage="llm_request",
                event_type="end",
                elapsed_ms=latency_ms,
                payload={
                    "call_id": call_id,
                    "http_status": http_status,
                    "content_length": len(content),
                },
            )

        return content

def _safe_debug_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in str(value))
    return cleaned[:80] or "llm_call"

