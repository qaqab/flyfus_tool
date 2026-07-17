from __future__ import annotations

import json

from tools.skills.flypower_skills import FlypowerSkillsTool


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = "failure" if status_code != 200 else ""

    def json(self) -> dict:
        return self._payload


def _tool() -> FlypowerSkillsTool:
    return FlypowerSkillsTool.from_credentials(
        {
            "geo_url": "https://skills.example",
            "geo_key": "test-token",
            "env": "prod",
        }
    )


def test_list_skills_returns_content_text(monkeypatch) -> None:
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse({"code": 200, "data": {"content": "listing-diagnosis"}})

    monkeypatch.setattr("tools.skills.flypower_skills.requests.post", post)

    messages = list(_tool().invoke({"method": "list_skills", "agent_name": "listing-agent"}))

    assert len(messages) == 1
    assert messages[0].message.text == "listing-diagnosis"
    assert calls == [
        (
            "https://skills.example/dify_prompt/skills/list",
            {
                "headers": {"Content-Type": "application/json", "Authorization": "Bearer test-token"},
                "json": {"agent_name": "listing-agent", "env": "prod"},
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

    monkeypatch.setattr("tools.skills.flypower_skills.requests.post", post)

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
            "skill_prompt": "Prompt for {{geo_prompt:listing-agent.listing-diagnosis@prod}}",
        },
        {
            "skill_name": "listing-optimization",
            "skill_prompt": "Prompt for {{geo_prompt:listing-agent.listing-optimization@prod}}",
        },
    ]
    assert len(calls) == 2
    assert all(call[0] == "https://skills.example/dify_prompt/render" for call in calls)


def test_load_skill_requires_skill_name() -> None:
    messages = list(_tool().invoke({"method": "load_skill", "agent_name": "listing-agent"}))

    assert len(messages) == 1
    assert messages[0].message.text.startswith("Error: skill_names")


def test_load_skill_accepts_legacy_single_skill_name(monkeypatch) -> None:
    monkeypatch.setattr(
        "tools.skills.flypower_skills.requests.post",
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
        "tools.skills.flypower_skills.requests.post",
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
