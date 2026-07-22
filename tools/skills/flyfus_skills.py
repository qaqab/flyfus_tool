from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Generator

import requests
from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

from tools._sls_logging import write_tool_log


class FlyfusSkillsTool(Tool):
    _REFERENCE_PART_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
    _SKILL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
    _REQUEST_TIMEOUT = (10, 60)
    _REQUEST_ATTEMPTS = 3
    _RETRY_DELAY_SECONDS = 10

    def _invoke(self, tool_parameters: dict) -> Generator[ToolInvokeMessage, None, None]:
        log_id = str(uuid.uuid4())
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
                yield self.create_text_message(self._list_skills(agent_name, log_id))
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
                {
                    "skill_name": skill_name,
                    "skill_prompt": self._load_skill(agent_name, skill_name, log_id),
                }
                for skill_name in skill_names
            ]
            yield self.create_text_message(json.dumps(skills, ensure_ascii=False))
        except RuntimeError as error:
            yield self.create_text_message(f"Error: {error}")

    def _list_skills(self, agent_name: str, log_id: str) -> str:
        response = self._post("/dify_admin/skills/list", {"agent_name": agent_name}, log_id)
        return self._response_text(response, "content")

    def _load_skill(self, agent_name: str, skill_name: str, log_id: str) -> str:
        reference = f"{{{{dify_admin:{agent_name}.{skill_name}}}}}"
        response = self._post("/dify_admin/render", {"type": "skills", "text": reference}, log_id)
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

    def _post(self, path: str, payload: dict, log_id: str) -> requests.Response:
        url = f"{self._credential('geo_url').rstrip('/')}{path}"
        for attempt in range(1, self._REQUEST_ATTEMPTS + 1):
            self._write_log(
                log_id,
                "request_attempt_started",
                path=path,
                attempt=attempt,
                max_attempts=self._REQUEST_ATTEMPTS,
            )
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
                break
            except requests.RequestException as error:
                if attempt == self._REQUEST_ATTEMPTS:
                    self._write_log(
                        log_id,
                        "request_failed",
                        path=path,
                        attempt=attempt,
                        max_attempts=self._REQUEST_ATTEMPTS,
                        error_type=type(error).__name__,
                    )
                    raise RuntimeError(f"Skills request failed after {attempt} attempts: {error}") from error
                self._write_log(
                    log_id,
                    "request_retry",
                    path=path,
                    attempt=attempt,
                    max_attempts=self._REQUEST_ATTEMPTS,
                    retry_delay_seconds=self._RETRY_DELAY_SECONDS,
                    error_type=type(error).__name__,
                )
                time.sleep(self._RETRY_DELAY_SECONDS)

        if response.status_code != 200:
            self._write_log(
                log_id,
                "request_failed",
                path=path,
                attempt=attempt,
                max_attempts=self._REQUEST_ATTEMPTS,
                status_code=response.status_code,
            )
            raise RuntimeError(f"Skills request failed with status {response.status_code}: {response.text}")
        self._write_log(
            log_id,
            "request_succeeded",
            path=path,
            attempt=attempt,
            status_code=response.status_code,
        )
        return response

    def _write_log(self, log_id: str, event: str, **fields: object) -> None:
        write_tool_log(self.runtime.credentials, log_id, f"skills_{event}", **fields)

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
