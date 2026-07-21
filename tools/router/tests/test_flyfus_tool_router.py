from __future__ import annotations

import json
import threading
import time

from dify_plugin.entities.tool import ToolInvokeMessage

from tools.router.flyfus_tool_router import FlyfusToolRouter


class FakeToolInvocation:
    def __init__(self) -> None:
        self.started: list[str] = []
        self._lock = threading.Lock()

    def invoke_builtin_tool(self, provider: str, tool_name: str, parameters: dict):
        assert provider == "qaqab/flyfus_tool/flyfus_tool"
        with self._lock:
            self.started.append(tool_name)
        time.sleep(0.1)
        yield ToolInvokeMessage(
            type=ToolInvokeMessage.MessageType.TEXT,
            message=ToolInvokeMessage.TextMessage(
                text=f"{tool_name}:{parameters['value']}"
            ),
        )


def _tool() -> FlyfusToolRouter:
    tool = FlyfusToolRouter.from_credentials({})
    tool.session.tool = FakeToolInvocation()
    return tool


def test_router_lists_the_two_fixed_tools() -> None:
    messages = list(_tool().invoke({"method": "list_tools"}))

    assert set(messages[0].message.json_object["tools"]) == {
        "set_next_step",
        "generate_image",
    }


def test_router_invokes_multiple_fixed_tools_in_parallel() -> None:
    tool = _tool()
    started_at = time.monotonic()
    messages = list(
        tool.invoke(
            {
                "method": "invoke_tools",
                "tool_calls": json.dumps(
                    [
                        {"tool": "set_next_step", "parameters": {"value": 1}},
                        {"tool": "generate_image", "parameters": {"value": 2}},
                    ]
                ),
            }
        )
    )

    assert time.monotonic() - started_at < 0.16
    assert [item["status"] for item in messages[0].message.json_object["results"]] == [
        "success",
        "success",
    ]


def test_router_rejects_unknown_tool() -> None:
    messages = list(
        _tool().invoke({"method": "invoke_tools", "tool_calls": '[{"tool":"unknown"}]'})
    )

    assert messages[0].message.text == "Error: Unsupported tool: unknown."
