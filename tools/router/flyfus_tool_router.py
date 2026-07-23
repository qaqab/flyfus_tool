from __future__ import annotations

import base64
import json
import time
import traceback
import uuid
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests
from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

from tools._sls_logging import write_tool_log


class FlyfusToolRouter(Tool):
    """Load Geo's Dify tool catalog and reverse-invoke selected tools."""

    _MAX_CALLS = 8
    _REQUEST_TIMEOUT = (10, 60)
    _SUPPORTED_PROVIDER_TYPES = frozenset({"builtin", "api", "workflow"})

    def _invoke(self, tool_parameters: dict) -> Generator[ToolInvokeMessage, None, None]:
        batch_log_id = str(uuid.uuid4())
        started_at_ms = self._epoch_ms()
        started_at = time.monotonic()
        method = str(tool_parameters.get("method") or "").strip()
        self._write_log(
            batch_log_id,
            "router_request_received",
            method=method,
            started_at_ms=started_at_ms,
            input_json=self._json_text(tool_parameters),
        )
        if method not in {"list_tools", "invoke_tools"}:
            error = "method must be list_tools or invoke_tools."
            self._write_log(
                batch_log_id,
                "router_request_finished",
                method=method,
                status="error",
                duration_ms=self._duration_ms(started_at),
                output_json=self._json_text({"error": error}),
            )
            yield self.create_text_message(f"Error: {error}")
            return
        try:
            catalog = self._fetch_catalog()
            if method == "list_tools":
                self._write_log(
                    batch_log_id,
                    "router_catalog_loaded",
                    method=method,
                    started_at_ms=started_at_ms,
                    duration_ms=self._duration_ms(started_at),
                    tool_count=catalog["tool_count"],
                    output_json=self._json_text(catalog),
                )
                yield self.create_json_message(catalog)
                return

            calls = self._parse_calls(tool_parameters.get("tool_calls"), catalog["tools"])
        except (RuntimeError, ValueError) as error:
            output = {"error": str(error)}
            self._write_log(
                batch_log_id,
                "router_request_finished",
                method=method,
                status="error",
                duration_ms=self._duration_ms(started_at),
                output_json=self._json_text(output),
            )
            yield self.create_text_message(f"Error: {error}")
            return

        with ThreadPoolExecutor(max_workers=min(len(calls), self._MAX_CALLS)) as executor:
            futures = [
                executor.submit(self._invoke_one, call, batch_log_id, index)
                for index, call in enumerate(calls)
            ]
            results = [future.result() for future in futures]
        duration_ms = self._duration_ms(started_at)
        self._write_log(
            batch_log_id,
            "router_batch_finished",
            method=method,
            started_at_ms=started_at_ms,
            duration_ms=duration_ms,
            tool_count=len(calls),
            success_count=sum(result["status"] == "success" for result in results),
            error_count=sum(result["status"] == "error" for result in results),
            result_json=self._json_text(results),
        )
        yield self.create_json_message(
            {
                "batch_log_id": batch_log_id,
                "started_at_ms": started_at_ms,
                "duration_ms": duration_ms,
                "results": results,
            }
        )

    def _fetch_catalog(self) -> dict[str, Any]:
        url = f"{self._credential('geo_url').rstrip('/')}/dify_admin/tools/available"
        try:
            response = requests.post(
                url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._credential('geo_key')}",
                },
                json={},
                timeout=self._REQUEST_TIMEOUT,
            )
        except requests.RequestException as error:
            raise RuntimeError(f"Tool catalog request failed: {error}") from error

        if response.status_code != 200:
            raise RuntimeError(f"Tool catalog request failed with status {response.status_code}: {response.text}")
        try:
            payload = response.json()
        except ValueError as error:
            raise RuntimeError("Tool catalog response is not valid JSON.") from error

        data = payload.get("data") if isinstance(payload, dict) else None
        tools = data.get("tools") if isinstance(data, dict) else None
        if not isinstance(tools, list):
            raise RuntimeError("Tool catalog response is missing data.tools.")

        normalized_tools: list[dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            provider_type = tool.get("provider_type")
            provider = tool.get("provider")
            tool_name = tool.get("tool_name")
            if not all(isinstance(value, str) and value for value in (provider_type, provider, tool_name)):
                continue
            if provider_type not in self._SUPPORTED_PROVIDER_TYPES:
                continue
            parameters = tool.get("parameters")
            normalized_tools.append(
                {
                    "provider_type": provider_type,
                    "provider": provider,
                    "tool_name": tool_name,
                    "name": str(tool.get("name") or f"{provider_type}.{provider}.{tool_name}"),
                    "description": str(tool.get("description") or ""),
                    "parameters": parameters if isinstance(parameters, dict) else {"type": "object", "properties": {}},
                }
            )

        return {"tool_count": len(normalized_tools), "tools": normalized_tools}

    def _invoke_one(self, call: dict[str, Any], batch_log_id: str, call_index: int) -> dict[str, Any]:
        call_log_id = str(uuid.uuid4())
        started_at_ms = self._epoch_ms()
        started_at = time.monotonic()
        session = self.session
        session_id = getattr(session, "session_id", None)
        invocation_started_at_ms: int | None = None
        first_response_at_ms: int | None = None
        self._write_log(
            call_log_id,
            "router_call_started",
            batch_log_id=batch_log_id,
            call_index=call_index,
            started_at_ms=started_at_ms,
            name=call["name"],
            provider_type=call["provider_type"],
            provider=call["provider"],
            tool_name=call["tool_name"],
            input_json=self._json_text(
                {
                    "name": call["name"],
                    "provider_type": call["provider_type"],
                    "provider": call["provider"],
                    "tool_name": call["tool_name"],
                    "parameters": call["parameters"],
                }
            ),
        )
        try:
            invocation = session.tool
            provider_type = call["provider_type"]
            if provider_type == "builtin":
                response_stream = invocation.invoke_builtin_tool(
                    provider=call["provider"],
                    tool_name=call["tool_name"],
                    parameters=call["parameters"],
                )
            elif provider_type == "workflow":
                response_stream = invocation.invoke_workflow_tool(
                    provider=call["provider"],
                    tool_name=call["tool_name"],
                    parameters=call["parameters"],
                )
            else:
                response_stream = invocation.invoke_api_tool(
                    provider=call["provider"],
                    tool_name=call["tool_name"],
                    parameters=call["parameters"],
                )
            messages: list[ToolInvokeMessage] = []
            invocation_started_at_ms = self._epoch_ms()
            for message in response_stream:
                if first_response_at_ms is None:
                    first_response_at_ms = self._epoch_ms()
                messages.append(message)
        except Exception as error:
            error_details = self._error_details(error)
            result = {
                "name": call["name"],
                "status": "error",
                "error": str(error),
                "call_log_id": call_log_id,
                "started_at_ms": started_at_ms,
                "invocation_started_at_ms": invocation_started_at_ms,
                "first_response_at_ms": first_response_at_ms,
                "first_response_delay_ms": (
                    first_response_at_ms - invocation_started_at_ms
                    if first_response_at_ms is not None and invocation_started_at_ms is not None
                    else None
                ),
                "duration_ms": self._duration_ms(started_at),
            }
            self._write_log(
                call_log_id,
                "router_call_finished",
                batch_log_id=batch_log_id,
                call_index=call_index,
                provider_type=call["provider_type"],
                provider=call["provider"],
                tool_name=call["tool_name"],
                status="error",
                error=str(error),
                error_json=self._json_text(error_details),
                session_id=session_id,
                invocation_started_at_ms=invocation_started_at_ms,
                first_response_at_ms=first_response_at_ms,
                first_response_delay_ms=result["first_response_delay_ms"],
                duration_ms=result["duration_ms"],
                output_json=self._json_text({**result, "error_details": error_details}),
            )
            return result

        result = {
            "name": call["name"],
            "status": "success",
            "messages": [self._message_to_dict(message) for message in messages],
            "call_log_id": call_log_id,
            "started_at_ms": started_at_ms,
            "invocation_started_at_ms": invocation_started_at_ms,
            "first_response_at_ms": first_response_at_ms,
            "first_response_delay_ms": (
                first_response_at_ms - invocation_started_at_ms if first_response_at_ms is not None else None
            ),
            "last_response_at_ms": self._epoch_ms() if messages else None,
            "message_count": len(messages),
            "duration_ms": self._duration_ms(started_at),
        }
        self._write_log(
            call_log_id,
            "router_call_finished",
            batch_log_id=batch_log_id,
            call_index=call_index,
            provider_type=call["provider_type"],
            provider=call["provider"],
            tool_name=call["tool_name"],
            status="success",
            session_id=session_id,
            invocation_started_at_ms=invocation_started_at_ms,
            first_response_at_ms=first_response_at_ms,
            first_response_delay_ms=result["first_response_delay_ms"],
            last_response_at_ms=result["last_response_at_ms"],
            message_count=result["message_count"],
            duration_ms=result["duration_ms"],
            output_json=self._json_text(result["messages"]),
        )
        return result

    @classmethod
    def _parse_calls(cls, value: object, catalog_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("tool_calls must be a JSON array.")
        try:
            calls = json.loads(value)
        except ValueError as error:
            raise ValueError("tool_calls must be valid JSON.") from error
        if not isinstance(calls, list) or not calls:
            raise ValueError("tool_calls must contain at least one call.")
        if len(calls) > cls._MAX_CALLS:
            raise ValueError(f"tool_calls supports at most {cls._MAX_CALLS} calls.")

        tool_by_name = {tool["name"]: tool for tool in catalog_tools}
        parsed_calls: list[dict[str, Any]] = []
        for call in calls:
            if not isinstance(call, dict):
                raise ValueError("Each tool call must be an object.")
            name = call.get("name")
            parameters = call.get("parameters", {})
            if not isinstance(name, str) or name not in tool_by_name:
                raise ValueError("Each tool call name must be a tool returned by list_tools.")
            if not isinstance(parameters, dict):
                raise ValueError("Each tool call parameters field must be an object.")
            parsed_calls.append({**tool_by_name[name], "parameters": parameters})
        return parsed_calls

    def _credential(self, name: str) -> str:
        value = str(self.runtime.credentials.get(name) or "").strip()
        if not value:
            raise RuntimeError(f"Missing required Tool Router credential: {name}.")
        return value

    def _write_log(self, log_id: str, event: str, **fields: object) -> None:
        write_tool_log(self.runtime.credentials, log_id, event, **fields)

    @staticmethod
    def _epoch_ms() -> int:
        return int(time.time() * 1000)

    @staticmethod
    def _duration_ms(started_at: float) -> int:
        return round((time.monotonic() - started_at) * 1000)

    @staticmethod
    def _json_text(value: object) -> str:
        return json.dumps(value, ensure_ascii=False, default=FlyfusToolRouter._json_default, separators=(",", ":"))

    @staticmethod
    def _json_default(value: object) -> object:
        if isinstance(value, bytes):
            return {"encoding": "base64", "data": base64.b64encode(value).decode("ascii")}
        return str(value)

    @classmethod
    def _error_details(cls, error: Exception) -> dict[str, Any]:
        chain: list[dict[str, Any]] = []
        current: BaseException | None = error
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            detail: dict[str, Any] = {
                "type": type(current).__name__,
                "message": str(current),
                "repr": repr(current),
            }
            errors = getattr(current, "errors", None)
            if callable(errors):
                try:
                    detail["validation_errors"] = errors()
                except Exception as nested_error:
                    detail["validation_errors_error"] = repr(nested_error)
            for name in ("execution_id", "workflow_run_id", "task_id", "request_id", "backwards_request_id"):
                found, value = cls._safe_getattr(current, name)
                if found:
                    detail[name] = value
            found, response = cls._safe_getattr(current, "response")
            if found:
                detail["http_response"] = cls._http_response_details(response)
            chain.append(detail)
            current = current.__cause__ or current.__context__
        return {
            "exception_chain": chain,
            "traceback": "".join(traceback.format_exception(error)),
            "sdk_exposes_backwards_request_id": False,
        }

    @staticmethod
    def _http_response_details(response: object) -> dict[str, Any]:
        details: dict[str, Any] = {}
        for name in ("status_code", "reason_phrase", "url"):
            found, value = FlyfusToolRouter._safe_getattr(response, name)
            if found:
                details[name] = value
        found, headers = FlyfusToolRouter._safe_getattr(response, "headers")
        if found:
            try:
                details["headers"] = dict(headers)
            except Exception as error:
                details["headers_read_error"] = repr(error)
        for name in ("text", "content"):
            found, value = FlyfusToolRouter._safe_getattr(response, name)
            if found:
                details["body" if name == "text" else "body_bytes"] = value
        return details

    @staticmethod
    def _safe_getattr(value: object, name: str) -> tuple[bool, object | None]:
        try:
            result = getattr(value, name, None)
        except Exception as error:
            return True, {"read_error": repr(error)}
        return result is not None, result

    @staticmethod
    def _message_to_dict(message: ToolInvokeMessage) -> dict[str, Any]:
        # Preserve every SDK message field (including meta and Base64-encoded blobs).
        return message.model_dump(mode="json")
