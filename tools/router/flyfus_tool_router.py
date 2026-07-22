from __future__ import annotations

import json
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests
from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage


class FlyfusToolRouter(Tool):
    """Load Geo's Dify tool catalog and reverse-invoke selected tools."""

    _MAX_CALLS = 8
    _REQUEST_TIMEOUT = (10, 60)
    _SUPPORTED_PROVIDER_TYPES = frozenset({"builtin", "api", "workflow"})

    def _invoke(self, tool_parameters: dict) -> Generator[ToolInvokeMessage, None, None]:
        method = str(tool_parameters.get("method") or "").strip()
        if method not in {"list_tools", "invoke_tools"}:
            yield self.create_text_message("Error: method must be list_tools or invoke_tools.")
            return

        try:
            catalog = self._fetch_catalog()
            if method == "list_tools":
                yield self.create_json_message(catalog)
                return

            calls = self._parse_calls(tool_parameters.get("tool_calls"), catalog["tools"])
        except RuntimeError as error:
            yield self.create_text_message(f"Error: {error}")
            return
        except ValueError as error:
            yield self.create_text_message(f"Error: {error}")
            return

        with ThreadPoolExecutor(max_workers=min(len(calls), self._MAX_CALLS)) as executor:
            futures = [executor.submit(self._invoke_one, call) for call in calls]
            results = [future.result() for future in futures]
        yield self.create_json_message({"results": results})

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

    def _invoke_one(self, call: dict[str, Any]) -> dict[str, Any]:
        try:
            invocation = self.session.tool
            provider_type = call["provider_type"]
            if provider_type == "builtin":
                messages = list(
                    invocation.invoke_builtin_tool(
                        provider=call["provider"],
                        tool_name=call["tool_name"],
                        parameters=call["parameters"],
                    )
                )
            elif provider_type == "workflow":
                messages = list(
                    invocation.invoke_workflow_tool(
                        provider=call["provider"],
                        tool_name=call["tool_name"],
                        parameters=call["parameters"],
                    )
                )
            else:
                messages = list(
                    invocation.invoke_api_tool(
                        provider=call["provider"],
                        tool_name=call["tool_name"],
                        parameters=call["parameters"],
                    )
                )
        except Exception as error:
            return {"name": call["name"], "status": "error", "error": str(error)}

        return {
            "name": call["name"],
            "status": "success",
            "messages": [self._message_to_dict(message) for message in messages],
        }

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

    @staticmethod
    def _message_to_dict(message: ToolInvokeMessage) -> dict[str, Any]:
        result: dict[str, Any] = {"type": message.type.value}
        if message.type == ToolInvokeMessage.MessageType.JSON:
            result["data"] = message.message.json_object
        elif hasattr(message.message, "text"):
            result["text"] = message.message.text
        else:
            result["data"] = repr(message.message)
        return result
