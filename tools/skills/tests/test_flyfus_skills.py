from __future__ import annotations

import json
from unittest.mock import Mock

import requests

from tools.skills.flyfus_skills import FlyfusSkillsTool


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = "failure" if status_code != 200 else ""

    def json(self) -> dict:
        return self._payload


def _tool() -> FlyfusSkillsTool:
    return FlyfusSkillsTool.from_credentials(
        {
            "geo_url": "https://skills.example",
            "geo_key": "test-token",
        }
    )


def test_list_skills_returns_content_text(monkeypatch) -> None:
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse({"code": 200, "data": {"content": "listing-diagnosis"}})

    monkeypatch.setattr("tools.skills.flyfus_skills.requests.post", post)

    messages = list(_tool().invoke({"method": "list_skills", "agent_name": "listing-agent"}))

    assert len(messages) == 1
    assert messages[0].message.text == "listing-diagnosis"
    assert calls == [
        (
            "https://skills.example/dify_admin/skills/list",
            {
                "headers": {"Content-Type": "application/json", "Authorization": "Bearer test-token"},
                "json": {"agent_name": "listing-agent"},
                "timeout": (10, 60),
            },
        )
    ]


def test_load_skill_returns_multiple_rendered_prompts(monkeypatch) -> None:
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        reference = kwargs["json"]["text"]
        return FakeResponse({"code": 200, "data": {"rendered_text": f"Prompt for {reference}"}})

    monkeypatch.setattr("tools.skills.flyfus_skills.requests.post", post)

    messages = list(
        _tool().invoke(
            {
                "method": "load_skill",
                "agent_name": "listing-agent",
                "skill_names": ["listing-diagnosis", "listing-optimization", "listing-diagnosis"],
            }
        )
    )

    assert len(messages) == 1
    assert json.loads(messages[0].message.text) == [
        {
            "skill_name": "listing-diagnosis",
            "skill_prompt": "Prompt for {{dify_admin:listing-agent.listing-diagnosis}}",
        },
        {
            "skill_name": "listing-optimization",
            "skill_prompt": "Prompt for {{dify_admin:listing-agent.listing-optimization}}",
        },
    ]
    assert len(calls) == 2
    assert all(call[0] == "https://skills.example/dify_admin/render" for call in calls)


def test_load_skill_requires_skill_name() -> None:
    messages = list(_tool().invoke({"method": "load_skill", "agent_name": "listing-agent"}))

    assert len(messages) == 1
    assert messages[0].message.text.startswith("Error: skill_names")


def test_load_skill_accepts_legacy_single_skill_name(monkeypatch) -> None:
    monkeypatch.setattr(
        "tools.skills.flyfus_skills.requests.post",
        lambda url, **kwargs: FakeResponse({"code": 200, "data": {"rendered_text": "Prompt"}}),
    )

    messages = list(
        _tool().invoke(
            {"method": "load_skill", "agent_name": "listing-agent", "skill_name": "listing-diagnosis"}
        )
    )

    assert json.loads(messages[0].message.text) == [
        {"skill_name": "listing-diagnosis", "skill_prompt": "Prompt"}
    ]


def test_load_skill_accepts_json_array_text(monkeypatch) -> None:
    monkeypatch.setattr(
        "tools.skills.flyfus_skills.requests.post",
        lambda url, **kwargs: FakeResponse({"code": 200, "data": {"rendered_text": "Prompt"}}),
    )

    messages = list(
        _tool().invoke(
            {
                "method": "load_skill",
                "agent_name": "listing-agent",
                "skill_names": '["listing-diagnosis", "listing-optimization"]',
            }
        )
    )

    assert [item["skill_name"] for item in json.loads(messages[0].message.text)] == [
        "listing-diagnosis",
        "listing-optimization",
    ]


def test_load_skill_retries_and_logs_network_errors(monkeypatch) -> None:
    post = Mock(
        side_effect=[
            requests.ConnectTimeout("first timeout"),
            requests.ConnectTimeout("second timeout"),
            FakeResponse({"code": 200, "data": {"rendered_text": "Prompt"}}),
        ]
    )
    sleep = Mock()
    write_log = Mock()
    monkeypatch.setattr("tools.skills.flyfus_skills.requests.post", post)
    monkeypatch.setattr("tools.skills.flyfus_skills.time.sleep", sleep)
    monkeypatch.setattr("tools.skills.flyfus_skills.write_tool_log", write_log)

    messages = list(
        _tool().invoke(
            {"method": "load_skill", "agent_name": "listing-agent", "skill_name": "listing-diagnosis"}
        )
    )

    assert json.loads(messages[0].message.text) == [{"skill_name": "listing-diagnosis", "skill_prompt": "Prompt"}]
    assert post.call_count == 3
    assert sleep.call_args_list == [((10,), {}), ((10,), {})]
    assert [call.args[2] for call in write_log.call_args_list] == [
        "skills_request_attempt_started",
        "skills_request_retry",
        "skills_request_attempt_started",
        "skills_request_retry",
        "skills_request_attempt_started",
        "skills_request_succeeded",
    ]
