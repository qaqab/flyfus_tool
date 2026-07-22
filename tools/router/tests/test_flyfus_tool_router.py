from __future__ import annotations

import json
import threading
import time

from dify_plugin.entities.tool import ToolInvokeMessage

from tools.router.flyfus_tool_router import FlyfusToolRouter


CATALOG_TOOLS = [
    {
        "provider_type": "workflow",
        "provider": "workflow-provider",
        "tool_name": "skill_tool",
        "name": "workflow.workflow-provider.skill_tool",
        "description": "Run a skill workflow.",
        "parameters": {"type": "object", "properties": {"input": {"type": "string"}}},
    },
    {
        "provider_type": "builtin",
        "provider": "builtin-provider",
        "tool_name": "read_tool",
        "name": "builtin.builtin-provider.read_tool",
        "description": "Read a file.",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}},
    },
    {
        "provider_type": "api",
        "provider": "api-provider",
        "tool_name": "search_tool",
        "name": "api.api-provider.search_tool",
        "description": "Search data.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
    },
]


class FakeResponse:
    status_code = 200
    text = ""

    def json(self) -> dict:
        return {"data": {"tool_count": len(CATALOG_TOOLS), "tools": CATALOG_TOOLS}}


class FakeToolInvocation:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, dict]] = []
        self._lock = threading.Lock()

    def invoke_builtin_tool(self, provider: str, tool_name: str, parameters: dict):
        return self._invoke("builtin", provider, tool_name, parameters)

    def invoke_workflow_tool(self, provider: str, tool_name: str, parameters: dict):
        return self._invoke("workflow", provider, tool_name, parameters)

    def invoke_api_tool(self, provider: str, tool_name: str, parameters: dict):
        return self._invoke("api", provider, tool_name, parameters)

    def _invoke(self, provider_type: str, provider: str, tool_name: str, parameters: dict):
        with self._lock:
            self.calls.append((provider_type, provider, tool_name, parameters))
        time.sleep(0.1)
        yield ToolInvokeMessage(
            type=ToolInvokeMessage.MessageType.TEXT,
            message=ToolInvokeMessage.TextMessage(text=f"{provider_type}:{tool_name}"),
        )


def _tool() -> FlyfusToolRouter:
    tool = FlyfusToolRouter.from_credentials({"geo_url": "https://geo.example/api/geo/v2", "geo_key": "test-key"})
    tool.session.tool = FakeToolInvocation()
    return tool


def test_router_lists_geo_catalog(monkeypatch) -> None:
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse()

    monkeypatch.setattr("tools.router.flyfus_tool_router.requests.post", post)

    messages = list(_tool().invoke({"method": "list_tools"}))

    assert messages[0].message.json_object["tool_count"] == 3
    assert [tool["name"] for tool in messages[0].message.json_object["tools"]] == [
        tool["name"] for tool in CATALOG_TOOLS
    ]
    assert calls == [
        (
            "https://geo.example/api/geo/v2/dify_admin/tools/available",
            {
                "headers": {"Content-Type": "application/json", "Authorization": "Bearer test-key"},
                "json": {},
                "timeout": (10, 60),
            },
        )
    ]


def test_router_invokes_catalog_tools_in_parallel(monkeypatch) -> None:
    monkeypatch.setattr("tools.router.flyfus_tool_router.requests.post", lambda url, **kwargs: FakeResponse())
    tool = _tool()
    started_at = time.monotonic()
    messages = list(
        tool.invoke(
            {
                "method": "invoke_tools",
                "tool_calls": json.dumps(
                    [
                        {"name": CATALOG_TOOLS[0]["name"], "parameters": {"input": "one"}},
                        {"name": CATALOG_TOOLS[1]["name"], "parameters": {"url": "two"}},
                        {"name": CATALOG_TOOLS[2]["name"], "parameters": {"query": "three"}},
                    ]
                ),
            }
        )
    )

    assert time.monotonic() - started_at < 0.18
    assert [item["status"] for item in messages[0].message.json_object["results"]] == [
        "success",
        "success",
        "success",
    ]
    assert {call[0] for call in tool.session.tool.calls} == {"builtin", "workflow", "api"}


def test_router_rejects_tool_not_returned_by_geo(monkeypatch) -> None:
    monkeypatch.setattr("tools.router.flyfus_tool_router.requests.post", lambda url, **kwargs: FakeResponse())

    messages = list(
        _tool().invoke({"method": "invoke_tools", "tool_calls": '[{"name":"unknown","parameters":{}}]'})
    )

    assert messages[0].message.text == "Error: Each tool call name must be a tool returned by list_tools."


def test_router_writes_input_output_and_timing_logs(monkeypatch) -> None:
    events = []
    monkeypatch.setattr("tools.router.flyfus_tool_router.requests.post", lambda url, **kwargs: FakeResponse())
    monkeypatch.setattr(
        "tools.router.flyfus_tool_router.write_tool_log",
        lambda credentials, log_id, event, **fields: events.append((log_id, event, fields)),
    )

    messages = list(
        _tool().invoke(
            {
                "method": "invoke_tools",
                "tool_calls": json.dumps(
                    [{"name": CATALOG_TOOLS[0]["name"], "parameters": {"input": "audit me"}}]
                ),
            }
        )
    )

    result = messages[0].message.json_object
    call_started = next(fields for _, event, fields in events if event == "router_call_started")
    call_finished = next(fields for _, event, fields in events if event == "router_call_finished")
    batch_finished = next(fields for _, event, fields in events if event == "router_batch_finished")
    assert json.loads(call_started["input_json"]) == {
        "name": CATALOG_TOOLS[0]["name"],
        "provider_type": "workflow",
        "provider": "workflow-provider",
        "tool_name": "skill_tool",
        "parameters": {"input": "audit me"},
    }
    assert call_finished["status"] == "success"
    assert json.loads(call_finished["output_json"])[0]["message"]["text"] == "workflow:skill_tool"
    assert call_finished["duration_ms"] >= 100
    assert batch_finished["result_json"]
    assert result["results"][0]["call_log_id"]
    assert result["duration_ms"] >= 100


def test_router_logs_complete_error_response(monkeypatch) -> None:
    events = []
    monkeypatch.setattr("tools.router.flyfus_tool_router.requests.post", lambda url, **kwargs: FakeResponse())
    monkeypatch.setattr(
        "tools.router.flyfus_tool_router.write_tool_log",
        lambda credentials, log_id, event, **fields: events.append((log_id, event, fields)),
    )

    messages = list(
        _tool().invoke({"method": "invoke_tools", "tool_calls": '[{"name":"unknown","parameters":{}}]'})
    )

    assert messages[0].message.text == "Error: Each tool call name must be a tool returned by list_tools."
    finished = next(fields for _, event, fields in events if event == "router_request_finished")
    assert json.loads(finished["output_json"]) == {
        "error": "Each tool call name must be a tool returned by list_tools."
    }
