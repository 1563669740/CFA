from __future__ import annotations

import json
import os
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


TRACE_LEVEL_OFF = "OFF"
TRACE_LEVEL_BASIC = "BASIC"
TRACE_LEVEL_FULL = "FULL"
TRACE_LEVEL_FORENSIC = "FORENSIC"

_TRACE_LEVELS = {
    TRACE_LEVEL_OFF,
    TRACE_LEVEL_BASIC,
    TRACE_LEVEL_FULL,
    TRACE_LEVEL_FORENSIC,
}


@dataclass(frozen=True)
class TraceConfig:
    enabled: bool = False
    level: str = TRACE_LEVEL_OFF
    root: Path = Path("logs/traces")
    include_sensitive: bool = False
    flush_immediately: bool = True

    @classmethod
    def from_env(cls, base_dir: str | Path | None = None) -> "TraceConfig":
        enabled = os.environ.get("CFA_TRACE_ENABLED", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        level = os.environ.get("CFA_TRACE_LEVEL", TRACE_LEVEL_FULL).strip().upper()
        if level not in _TRACE_LEVELS:
            level = TRACE_LEVEL_FULL
        if not enabled or level == TRACE_LEVEL_OFF:
            return cls(enabled=False, level=TRACE_LEVEL_OFF)

        raw_root = os.environ.get("CFA_TRACE_ROOT", "logs/traces").strip() or "logs/traces"
        root = Path(raw_root)
        if not root.is_absolute() and base_dir is not None:
            root = Path(base_dir) / root

        include_sensitive = os.environ.get(
            "CFA_TRACE_INCLUDE_SENSITIVE",
            "0",
        ).strip().lower() in {"1", "true", "yes", "on"}

        flush_immediately = os.environ.get(
            "CFA_TRACE_FLUSH_IMMEDIATELY",
            "1",
        ).strip().lower() not in {"0", "false", "no", "off"}

        return cls(
            enabled=True,
            level=level,
            root=root,
            include_sensitive=include_sensitive,
            flush_immediately=flush_immediately,
        )


class TraceRecorder:
    """Append-only per-request trace recorder.

    The recorder writes small events to ``events.jsonl`` and larger payloads to
    separate JSON snapshots referenced by event ``artifact_ref`` fields.
    """

    def __init__(
        self,
        *,
        request_id: str,
        config: TraceConfig,
        metadata: dict[str, Any] | None = None,
    ):
        self.request_id = request_id
        self.trace_id = request_id
        self.config = config
        self.metadata = dict(metadata or {})
        self._lock = threading.Lock()
        self._sequence = 0
        self._started_at = self._now()
        date_part = self._started_at[:10]
        safe_request_id = _safe_path_name(request_id)
        self.trace_dir = config.root / date_part / safe_request_id
        self.snapshot_dir = self.trace_dir / "snapshots"
        self.llm_dir = self.trace_dir / "llm"
        self.retrieval_dir = self.trace_dir / "retrieval"
        self.resources_dir = self.trace_dir / "resources"
        self.artifact_dir = self.trace_dir / "artifacts"
        self.manifest_path = self.trace_dir / "manifest.json"
        self.events_path = self.trace_dir / "events.jsonl"
        self.errors_path = self.trace_dir / "errors.jsonl"

        for path in (
            self.snapshot_dir,
            self.llm_dir,
            self.retrieval_dir,
            self.resources_dir,
            self.artifact_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

        self.emit(
            component="TraceRecorder",
            stage="trace_started",
            event_type="start",
            payload=self.metadata,
            sensitivity="internal",
        )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")

    @staticmethod
    def _to_jsonable(value: Any) -> Any:
        if is_dataclass(value):
            return TraceRecorder._to_jsonable(asdict(value))
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {
                str(key): TraceRecorder._to_jsonable(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [TraceRecorder._to_jsonable(item) for item in value]
        if hasattr(value, "to_dict"):
            return TraceRecorder._to_jsonable(value.to_dict())
        if hasattr(value, "model_dump"):
            return TraceRecorder._to_jsonable(value.model_dump())
        return value

    @staticmethod
    def _write_json_atomic(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(
                TraceRecorder._to_jsonable(payload),
                file,
                ensure_ascii=False,
                indent=2,
            )
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)

    def emit(
        self,
        *,
        component: str,
        stage: str,
        event_type: str,
        status: str = "success",
        span_id: str | None = None,
        parent_span_id: str | None = None,
        elapsed_ms: float | None = None,
        payload: dict[str, Any] | None = None,
        artifact_ref: str | None = None,
        sensitivity: str = "internal",
    ) -> None:
        if not self.config.enabled:
            return

        with self._lock:
            self._sequence += 1
            event = {
                "schema_version": "1.0",
                "trace_id": self.trace_id,
                "request_id": self.request_id,
                "seq": self._sequence,
                "timestamp": self._now(),
                "component": component,
                "stage": stage,
                "event_type": event_type,
                "status": status,
                "span_id": span_id,
                "parent_span_id": parent_span_id,
                "elapsed_ms": elapsed_ms,
                "payload": self._to_jsonable(payload or {}),
                "artifact_ref": artifact_ref,
                "sensitivity": sensitivity,
            }

            with self.events_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(event, ensure_ascii=False) + "\n")
                if self.config.flush_immediately:
                    file.flush()
                    os.fsync(file.fileno())

    def snapshot(
        self,
        name: str,
        payload: Any,
        *,
        component: str,
        stage: str,
        directory: str = "snapshots",
        sensitivity: str = "restricted",
    ) -> str:
        if not self.config.enabled:
            return ""
        base_dir = {
            "snapshots": self.snapshot_dir,
            "llm": self.llm_dir,
            "retrieval": self.retrieval_dir,
            "resources": self.resources_dir,
            "artifacts": self.artifact_dir,
        }.get(directory, self.snapshot_dir)
        artifact_path = base_dir / f"{_safe_path_name(name)}.json"
        self._write_json_atomic(artifact_path, payload)
        relative_path = str(artifact_path.relative_to(self.trace_dir))
        self.emit(
            component=component,
            stage=stage,
            event_type="snapshot",
            artifact_ref=relative_path,
            sensitivity=sensitivity,
        )
        return relative_path

    @contextmanager
    def span(
        self,
        *,
        component: str,
        stage: str,
        parent_span_id: str | None = None,
        input_payload: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        span_id = str(uuid.uuid4())
        started = time.perf_counter()
        self.emit(
            component=component,
            stage=stage,
            event_type="start",
            span_id=span_id,
            parent_span_id=parent_span_id,
            payload=input_payload,
        )
        try:
            yield span_id
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            error_payload = {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "stack_trace": traceback.format_exc(),
            }
            self.emit(
                component=component,
                stage=stage,
                event_type="error",
                status="failed",
                span_id=span_id,
                parent_span_id=parent_span_id,
                elapsed_ms=elapsed_ms,
                payload=error_payload,
                sensitivity="restricted",
            )
            with self.errors_path.open("a", encoding="utf-8") as file:
                file.write(
                    json.dumps(
                        {
                            "timestamp": self._now(),
                            "component": component,
                            "stage": stage,
                            **error_payload,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            raise
        else:
            elapsed_ms = (time.perf_counter() - started) * 1000
            self.emit(
                component=component,
                stage=stage,
                event_type="end",
                span_id=span_id,
                parent_span_id=parent_span_id,
                elapsed_ms=elapsed_ms,
            )

    def fallback(
        self,
        *,
        component: str,
        stage: str,
        requested_strategy: str,
        effective_strategy: str,
        reason_code: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.emit(
            component=component,
            stage=stage,
            event_type="fallback",
            status="degraded",
            payload={
                "requested_strategy": requested_strategy,
                "effective_strategy": effective_strategy,
                "reason_code": reason_code,
                "details": details or {},
            },
        )

    def finalize(
        self,
        *,
        status: str,
        summary: dict[str, Any] | None = None,
    ) -> None:
        if not self.config.enabled:
            return
        self.emit(
            component="TraceRecorder",
            stage="trace_finalized",
            event_type="end",
            status=status,
            payload=summary or {},
            sensitivity="internal",
        )
        manifest = {
            "schema_version": "1.0",
            "trace_id": self.trace_id,
            "request_id": self.request_id,
            "started_at": self._started_at,
            "ended_at": self._now(),
            "status": status,
            "event_count": self._sequence,
            "trace_level": self.config.level,
            "trace_dir": str(self.trace_dir),
            "metadata": self.metadata,
            "summary": summary or {},
        }
        self._write_json_atomic(self.manifest_path, manifest)


class NullTraceRecorder:
    trace_id = ""
    request_id = ""
    trace_dir = Path("")

    def emit(self, **_: Any) -> None:
        return None

    def snapshot(self, *_: Any, **__: Any) -> str:
        return ""

    def fallback(self, **_: Any) -> None:
        return None

    def finalize(self, **_: Any) -> None:
        return None

    @contextmanager
    def span(self, **_: Any) -> Iterator[str]:
        yield "null-span"


def create_trace_recorder(
    *,
    request_id: str,
    base_dir: str | Path,
    metadata: dict[str, Any] | None = None,
) -> TraceRecorder | NullTraceRecorder:
    config = TraceConfig.from_env(base_dir)
    if not config.enabled:
        return NullTraceRecorder()
    return TraceRecorder(
        request_id=request_id,
        config=config,
        metadata=metadata,
    )


def _safe_path_name(value: str) -> str:
    cleaned = "".join(
        ch if ch.isalnum() or ch in "._-" else "_"
        for ch in str(value or "")
    )
    return cleaned[:120] or "trace"
