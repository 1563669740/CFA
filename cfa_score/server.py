"""
CFA-Score HTTP API Server.

Zero-dependency (Python stdlib only) server that exposes the CFA Gateway
as REST endpoints.

Endpoints
---------
POST /api/cfa-chat     — Full pipeline: LLM + CFA + safe answer
POST /api/cfa-analyze  — CFA only: analyze given model_output
GET  /api/health       — Health check
GET  /api/scenarios    — List available scenarios

Start::

    python -m cfa_score.server               # default: 0.0.0.0:8080
    python -m cfa_score.server --port 9000    # custom port
    python -m cfa_score.server --host 127.0.0.1 --port 8080  # custom host

Or programmatically::

    from cfa_score.server import CFAServer
    CFAServer(host="0.0.0.0", port=8080).serve()
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse, parse_qs

from .gateway import CFAGateway


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"


# ---------------------------------------------------------------------------
# Request Handler
# ---------------------------------------------------------------------------

class _CFAHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the CFA-Score API."""

    gateway: CFAGateway = None  # type: ignore[assignment]  # set by CFAServer

    # Suppress default request logging (we do our own)
    def log_message(self, format: str, *args: Any) -> None:
        pass

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/api/health":
            self._health()
        elif path == "/api/scenarios":
            self._list_scenarios()
        elif path == "/api/fact-schema":
            self._fact_schema(parsed)
        elif path == "/api/protected-facts":
            self._list_protected_facts(parsed)
        elif path == "/api/debug/last-llm-payload":
            self._debug_last_llm_payload(parsed)
        else:
            self._not_found()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/api/cfa-chat":
            self._cfa_chat()
        elif path == "/api/cfa-analyze":
            self._cfa_analyze()
        elif path == "/api/protected-facts":
            self._add_protected_fact()
        elif path == "/api/protected-facts/import-jsonl":
            self._import_confidential_jsonl()
        else:
            self._not_found()

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    def _health(self) -> None:
        self._respond(200, {"status": "ok", "service": "cfa-gateway"})

    def _list_scenarios(self) -> None:
        self._respond(200, {
            "scenarios": [
                {"id": "healthcare", "label": "医疗健康 · 住院患者诊疗信息泄露检测"},
                {"id": "finance", "label": "金融信贷 · 银行信贷审批敏感信息泄露检测"},
                {"id": "aerospace", "label": "航天测控 · CVE-2024-3094 xz-utils 组合事实还原"},
                {"id": "meetings", "label": "会议室 · 企业内部涉密会议信息还原检测"},
            ]
        })

    def _cfa_chat(self) -> None:
        """POST /api/cfa-chat"""
        body = self._read_json()
        if body is None:
            return  # error already sent

        # Validate required fields
        if "user_input" not in body:
            self._respond(400, {"error": "Missing required field: user_input"})
            return

        scenario = body.get("scenario", "auto")
        mode = body.get("mode", "rule_only")
        secondary_check = body.get("secondary_check", False)

        try:
            resp = self.gateway.handle_chat(
                user_input=body["user_input"],
                scenario=scenario,
                mode=mode,
                secondary_check=secondary_check,
            )
            self._respond(200, resp.to_dict(debug=False))
        except ValueError as exc:
            self._respond(400, {"error": str(exc)})
        except Exception as exc:
            self._respond(500, {"error": f"CFA gateway error: {exc}"})

    def _cfa_analyze(self) -> None:
        """POST /api/cfa-analyze"""
        body = self._read_json()
        if body is None:
            return

        if "user_input" not in body:
            self._respond(400, {"error": "Missing required field: user_input"})
            return
        if "model_output" not in body:
            self._respond(400, {"error": "Missing required field: model_output"})
            return

        scenario = body.get("scenario", "healthcare")
        mode = body.get("mode", "rule_only")
        secondary_check = body.get("secondary_check", False)

        try:
            resp = self.gateway.handle_analyze(
                user_input=body["user_input"],
                model_output=body["model_output"],
                scenario=scenario,
                mode=mode,
                secondary_check=secondary_check,
            )
            self._respond(200, resp.to_dict(debug=False))
        except ValueError as exc:
            self._respond(400, {"error": str(exc)})
        except Exception as exc:
            self._respond(500, {"error": f"CFA analyze error: {exc}"})

    # ------------------------------------------------------------------
    # Admin: fact management endpoints
    # ------------------------------------------------------------------

    def _get_query_scenario(self, parsed, default: str = "healthcare") -> str:
        query = parse_qs(parsed.query)
        return query.get("scenario", [default])[0]

    def _fact_schema(self, parsed) -> None:
        scenario = self._get_query_scenario(parsed)

        try:
            data = self.gateway.get_fact_schema(scenario)
            self._respond(200, data)
        except ValueError as exc:
            self._respond(400, {"error": str(exc)})
        except Exception as exc:
            self._respond(500, {"error": f"fact schema error: {exc}"})

    def _list_protected_facts(self, parsed) -> None:
        scenario = self._get_query_scenario(parsed)

        try:
            data = self.gateway.list_protected_facts(scenario)
            self._respond(200, data)
        except ValueError as exc:
            self._respond(400, {"error": str(exc)})
        except Exception as exc:
            self._respond(500, {"error": f"list facts error: {exc}"})

    def _add_protected_fact(self) -> None:
        body = self._read_json()
        if body is None:
            return

        scenario = body.get("scenario", "healthcare")
        fact = body.get("fact")

        if not isinstance(fact, dict):
            self._respond(400, {"error": "Missing required field: fact object"})
            return

        try:
            data = self.gateway.add_protected_fact(scenario, fact)
            self._respond(200, data)
        except ValueError as exc:
            self._respond(400, {"error": str(exc)})
        except Exception as exc:
            self._respond(500, {"error": f"add fact error: {exc}"})

    def _import_confidential_jsonl(self) -> None:
        body = self._read_json()
        if body is None:
            return

        content = body.get("content", "")
        filename = body.get("filename", "")
        replace = bool(body.get("replace", False))

        if not isinstance(content, str) or not content.strip():
            self._respond(400, {"error": "Missing required field: content"})
            return

        try:
            data = self.gateway.import_confidential_jsonl(
                content=content,
                filename=filename,
                replace=replace,
            )
            self._respond(200, data)
        except ValueError as exc:
            self._respond(400, {"error": str(exc)})
        except Exception as exc:
            self._respond(500, {"error": f"import jsonl error: {exc}"})

    # ------------------------------------------------------------------
    # Debug endpoints
    # ------------------------------------------------------------------

    def _debug_last_llm_payload(self, parsed=None) -> None:
        """
        调试接口：返回真实发给 LLM 的 payload。
        支持 request_id + purpose 精确查询当前对话对应的 payload。
        """
        root = Path(__file__).resolve().parent.parent
        parsed = parsed or urlparse(self.path)
        query = parse_qs(parsed.query)
        request_id = query.get("request_id", [""])[0]
        purpose = query.get("purpose", [""])[0]

        if request_id:
            path = self._find_request_debug_payload(root, request_id, purpose)
            if path is None:
                self._respond(404, {
                    "error": "当前请求没有捕获到对应的 LLM 请求。可能未开启 CFA_DEBUG_LLM_PAYLOAD，或该请求未调用外部 LLM。"
                })
                return
        else:
            path = root / "logs" / "last_llm_payload.json"
            if not path.exists():
                self._respond(404, {
                    "error": "还没有捕获到 LLM 请求。请先在对话模式发送一次请求。"
                })
                return

        try:
            raw = path.read_text(encoding="utf-8-sig")
            data = json.loads(raw)

            # P0 安全：payload 中不得包含保密库敏感标记
            confidential_markers = [
                "SEC-", "保密内容=", "保密关键词=", "保密类别=", "保密摘要=",
                "保密事实库", "密级=", "机密★", "绝密★", "秘密★",
                "【内部资产台账】",
            ]
            raw_lower = raw.lower()
            if any(m in raw for m in confidential_markers) or any(m.lower() in raw_lower for m in confidential_markers):
                try:
                    path.unlink()
                except OSError:
                    pass
                self._respond(403, {
                    "error": "debug payload 包含保密库标记，已自动删除。"
                            "请重新发起一次安全的对话请求以生成新的 debug payload。"
                })
                return

            self._respond(200, data)
        except json.JSONDecodeError:
            self._respond(500, {"error": "debug payload JSON 解析失败"})
        except Exception as exc:
            self._respond(500, {
                "error": f"读取 LLM payload 失败: {exc}"
            })

    def _find_request_debug_payload(self, root: Path, request_id: str, purpose: str = "") -> Optional[Path]:
        safe_request_id = _safe_debug_name(request_id)
        debug_dir = root / "logs" / "llm_payloads" / safe_request_id
        if not debug_dir.exists():
            return None

        files = sorted(debug_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            return None

        if purpose:
            for path in files:
                try:
                    data = json.loads(path.read_text(encoding="utf-8-sig"))
                except Exception:
                    continue
                if data.get("request_id") == request_id and data.get("purpose") == purpose:
                    return path
            return None

        return files[0]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_json(self) -> Optional[Dict[str, Any]]:
        """Read and parse JSON request body.  Sends 400 on failure."""
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._respond(400, {"error": "Invalid Content-Length"})
            return None
        if content_length == 0:
            self._respond(400, {"error": "Empty request body"})
            return None

        raw = self.rfile.read(content_length)
        if isinstance(raw, bytes):
            text = raw.decode("utf-8")
        else:
            text = str(raw)
        if not text or text.strip() == "":
            self._respond(400, {"error": "Empty request body"})
            return None
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            self._respond(400, {"error": f"Invalid JSON: {exc}"})
            return None
        if not isinstance(data, dict):
            self._respond(400, {"error": "Request body must be a JSON object"})
            return None
        return data

    def _respond(self, status_code: int, data: Dict[str, Any]) -> None:
        """Send JSON response."""
        response = json.dumps(data, ensure_ascii=False, indent=2)
        body = response.encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Request-ID", data.get("request_id", ""))
        self.end_headers()
        self.wfile.write(body)

    def _not_found(self) -> None:
        self._respond(404, {"error": "Not found"})


def _safe_debug_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in str(value))
    return cleaned[:80] or "llm_call"


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class CFAServer:
    """HTTP server wrapping CFAGateway."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        env_path: str | Path = ENV_PATH,
        base_dir: str | Path | None = None,
    ):
        self.host = host
        self.port = port
        self._env_path = Path(env_path)
        self._base_dir = base_dir

    def serve(self) -> None:
        """Start the HTTP server.  Blocking call."""
        gateway = CFAGateway(
            env_path=self._env_path,
            base_dir=self._base_dir,
        )

        # Inject gateway into handler class
        _CFAHandler.gateway = gateway

        server = HTTPServer((self.host, self.port), _CFAHandler)
        print(f"CFA-Score Gateway listening on http://{self.host}:{self.port}")
        print(f"  POST /api/cfa-chat     — Full pipeline (LLM + CFA + safe answer)")
        print(f"  POST /api/cfa-analyze  — CFA only (analyze model_output)")
        print(f"  GET  /api/health       — Health check")
        print(f"  GET  /api/scenarios    — List available scenarios")

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down CFA-Score Gateway.")
            server.shutdown()

    def serve_non_blocking(self) -> HTTPServer:
        """Start the HTTP server in a new thread.  Returns the server object."""
        import threading

        gateway = CFAGateway(
            env_path=self._env_path,
            base_dir=self._base_dir,
        )
        _CFAHandler.gateway = gateway

        server = HTTPServer((self.host, self.port), _CFAHandler)
        print(f"CFA-Score Gateway listening on http://{self.host}:{self.port}")

        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CFA-Score LLM Safety Gateway HTTP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m cfa_score.server                          # default: 0.0.0.0:8080
  python -m cfa_score.server --port 9000               # custom port
  python -m cfa_score.server --host 127.0.0.1 --port 8080  # localhost only
        """.strip(),
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Bind host (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Bind port (default: {DEFAULT_PORT})")
    parser.add_argument("--env", default=str(ENV_PATH), help="Path to .env file")
    args = parser.parse_args()

    CFAServer(
        host=args.host,
        port=args.port,
        env_path=args.env,
    ).serve()


if __name__ == "__main__":
    main()