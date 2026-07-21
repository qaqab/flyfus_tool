from __future__ import annotations

import json
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage


class FlyfusToolRouter(Tool):
    """Expose and dispatch the fixed Flyfus tool set."""

    _MAX_CALLS = 8
    _PROVIDER = "qaqab/flyfus_tool/flyfus_tool"
    _TOOLS: dict[str, dict[str, Any]] = {
        "set_next_step": {
            "tool_name": "set_next_step",
            "description": "设置下一轮 Agent 的状态、目标和推理强度。",
            "parameters": {
                "next_state": "plan | act | observe | verify | write_output",
                "next_effort": "low | medium | high | xhigh",
                "next_objective": "下一轮的具体目标",
                "effort_reason": "选择该推理强度的原因",
            },
        },
        "generate_image": {
            "tool_name": "flyfus_image_generate",
            "description": "根据提示词生成图片；也可以提供参考图片 URL 来编辑图片。",
            "parameters": {
                "prompt": "必填，图片生成或编辑提示词",
                "reference_image_urls": "可选，参考图 URL",
                "mask_url": "可选，局部编辑蒙版 URL",
                "model": "可选，默认 gpt-image-2",
                "size": "可选，例如 1024x1024 或 auto",
            },
        },
    }

    def _invoke(
        self, tool_parameters: dict
    ) -> Generator[ToolInvokeMessage, None, None]:
        method = str(tool_parameters.get("method") or "").strip()
        if method == "list_tools":
            yield self.create_json_message({"tools": self._TOOLS})
            return
        if method != "invoke_tools":
            yield self.create_text_message(
                "Error: method must be list_tools or invoke_tools."
            )
            return

        try:
            calls = self._parse_calls(tool_parameters.get("tool_calls"))
        except ValueError as error:
            yield self.create_text_message(f"Error: {error}")
            return

        with ThreadPoolExecutor(
            max_workers=min(len(calls), self._MAX_CALLS)
        ) as executor:
            futures = [executor.submit(self._invoke_one, call) for call in calls]
            results = [future.result() for future in futures]
        yield self.create_json_message({"results": results})

    def _invoke_one(self, call: dict[str, Any]) -> dict[str, Any]:
        tool = self._TOOLS[call["tool"]]
        try:
            messages = list(
                self.session.tool.invoke_builtin_tool(
                    provider=self._PROVIDER,
                    tool_name=tool["tool_name"],
                    parameters=call["parameters"],
                )
            )
        except Exception as error:
            return {"tool": call["tool"], "status": "error", "error": str(error)}

        return {
            "tool": call["tool"],
            "status": "success",
            "messages": [self._message_to_dict(message) for message in messages],
        }

    @classmethod
    def _parse_calls(cls, value: object) -> list[dict[str, Any]]:
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

        parsed_calls: list[dict[str, Any]] = []
        for call in calls:
            if not isinstance(call, dict):
                raise ValueError("Each tool call must be an object.")
            tool = call.get("tool")
            parameters = call.get("parameters", {})
            if tool not in cls._TOOLS:
                raise ValueError(f"Unsupported tool: {tool}.")
            if not isinstance(parameters, dict):
                raise ValueError("Each tool call parameters field must be an object.")
            parsed_calls.append({"tool": tool, "parameters": parameters})
        return parsed_calls

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
