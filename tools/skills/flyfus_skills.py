from __future__ import annotations

import json
import re
from collections.abc import Generator

import requests
from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage


class FlyfusSkillsTool(Tool):
    _REFERENCE_PART_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
    _SKILL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
    _REQUEST_TIMEOUT = (10, 60)

    def _invoke(self, tool_parameters: dict) -> Generator[ToolInvokeMessage, None, None]:
        method = str(tool_parameters.get("method") or "").strip()
        agent_name = str(tool_parameters.get("agent_name") or "").strip()

        if method not in {"list_skills", "load_skill"}:
            yield self.create_text_message("Error: method must be list_skills or load_skill.")
            return
        if not self._REFERENCE_PART_PATTERN.fullmatch(agent_name):
            yield self.create_text_message("Error: agent_name must contain only letters, numbers, hyphens, or underscores.")
            return

        try:
            if method == "list_skills":
                yield self.create_text_message(self._list_skills(agent_name))
                return

            skill_names_value = tool_parameters.get("skill_names")
            if skill_names_value in (None, ""):
                skill_names_value = tool_parameters.get("skill_name")
            skill_names = self._parse_skill_names(skill_names_value)
            if not skill_names:
                yield self.create_text_message("Error: skill_names is required for load_skill.")
                return
            if any(not self._SKILL_NAME_PATTERN.fullmatch(skill_name) for skill_name in skill_names):
                yield self.create_text_message(
                    "Error: skill_names may contain only letters, numbers, periods, hyphens, or underscores."
                )
                return
            skills = [
                {"skill_name": skill_name, "skill_prompt": self._load_skill(agent_name, skill_name)}
                for skill_name in skill_names
            ]
            yield self.create_text_message(json.dumps(skills, ensure_ascii=False))
        except RuntimeError as error:
            yield self.create_text_message(f"Error: {error}")

    def _list_skills(self, agent_name: str) -> str:
        response = self._post("/dify_prompt/skills/list", {"agent_name": agent_name, "env": self._credential("env")})
        return self._response_text(response, "content")

    def _load_skill(self, agent_name: str, skill_name: str) -> str:
        reference = f"{{{{geo_prompt:{agent_name}.{skill_name}@{self._credential('env')}}}}}"
        response = self._post("/dify_prompt/render", {"type": "skills", "text": reference})
        return self._response_text(response, "rendered_text")

    @staticmethod
    def _parse_skill_names(value: object) -> list[str]:
        if isinstance(value, list):
            raw_names = value
        elif isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            try:
                parsed = json.loads(text)
            except ValueError:
                parsed = None

            if isinstance(parsed, list):
                raw_names = parsed
            elif isinstance(parsed, str):
                raw_names = [parsed]
            else:
                raw_names = re.split(r"[,\r\n]+", text)
        else:
            return []

        skill_names: list[str] = []
        seen: set[str] = set()
        for raw_name in raw_names:
            skill_name = str(raw_name).strip()
            if skill_name and skill_name not in seen:
                seen.add(skill_name)
                skill_names.append(skill_name)
        return skill_names

    def _post(self, path: str, payload: dict) -> requests.Response:
        url = f"{self._credential('geo_url').rstrip('/')}{path}"
        try:
            response = requests.post(
                url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._credential('geo_key')}",
                },
                json=payload,
                timeout=self._REQUEST_TIMEOUT,
            )
        except requests.RequestException as error:
            raise RuntimeError(f"Skills request failed: {error}") from error

        if response.status_code != 200:
            raise RuntimeError(f"Skills request failed with status {response.status_code}: {response.text}")
        return response

    @staticmethod
    def _response_text(response: requests.Response, field: str) -> str:
        try:
            payload = response.json()
        except ValueError as error:
            raise RuntimeError("Skills response is not valid JSON.") from error

        text = payload.get("data", {}).get(field) if isinstance(payload, dict) else None
        if not isinstance(text, str):
            raise RuntimeError(f"Skills response is missing data.{field}.")
        return text

    def _credential(self, name: str) -> str:
        value = str(self.runtime.credentials.get(name) or "").strip()
        if not value:
            raise RuntimeError(f"Missing required Skills credential: {name}.")
        return value
