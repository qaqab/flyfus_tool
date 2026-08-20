from __future__ import annotations

import re
from collections.abc import Generator

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
import requests


class FlyfusMemoryTool(Tool):
    _USER_ID_PATTERN = re.compile(
        r"^(?P<user_id>[1-9]\d*):web_chat:"
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
    )
    _REQUEST_TIMEOUT = (10, 60)

    def _invoke(self, tool_parameters: dict) -> Generator[ToolInvokeMessage, None, None]:
        method = str(tool_parameters.get("method") or "").strip()
        if method not in {"list_scenarios", "read_scenario"}:
            yield self.create_text_message("Error: method 不正确")
            return

        user_id = str(tool_parameters.get("user_id") or "").strip()
        match = self._USER_ID_PATTERN.fullmatch(user_id)
        if match is None:
            yield self.create_text_message("Error: user_id 不正确")
            return

        payload: dict[str, object] = {"user_id": int(match.group("user_id"))}
        if method == "list_scenarios":
            path_prefix = str(tool_parameters.get("path_prefix") or "").strip()
            if path_prefix:
                payload["path_prefix"] = path_prefix
            path = "/chat/memory/scenarios"
        else:
            scenario_path = str(tool_parameters.get("path") or "").strip()
            if not self._valid_scenario_path(scenario_path):
                yield self.create_text_message("Error: path 不正确")
                return
            payload["path"] = scenario_path
            path = "/chat/memory/scenarios/read"

        try:
            data = self._post(path, payload)
        except RuntimeError as error:
            yield self.create_text_message(f"Error: {error}")
            return
        yield self.create_json_message(data)

    def _post(self, path: str, payload: dict[str, object]) -> dict:
        base_url = self._credential("geo_url").rstrip("/")
        try:
            response = requests.post(
                f"{base_url}{path}",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._credential('geo_key')}",
                },
                json=payload,
                timeout=self._REQUEST_TIMEOUT,
            )
        except requests.RequestException as error:
            raise RuntimeError(f"场景记忆接口请求失败：{type(error).__name__}") from error

        if response.status_code != 200:
            raise RuntimeError(f"场景记忆接口 HTTP {response.status_code}")
        try:
            response_body = response.json()
        except ValueError:
            raise RuntimeError("场景记忆接口返回了无效 JSON") from None
        if not isinstance(response_body, dict):
            raise RuntimeError("场景记忆接口返回格式不正确")

        code = response_body.get("code")
        if code != 200:
            message = str(response_body.get("message") or "unknown error")
            raise RuntimeError(f"场景记忆接口业务码 {code}：{message}")
        data = response_body.get("data")
        if not isinstance(data, dict):
            raise RuntimeError("场景记忆接口未返回有效 data")
        return data

    @staticmethod
    def _valid_scenario_path(path: str) -> bool:
        return bool(path) and not path.startswith("/") and "\\" not in path and ".." not in path

    def _credential(self, name: str) -> str:
        value = str(self.runtime.credentials.get(name) or "").strip()
        if not value:
            raise RuntimeError(f"缺少场景记忆工具配置：{name}")
        return value
