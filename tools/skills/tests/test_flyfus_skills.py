from __future__ import annotations

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
            "env": "prod",
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
            "https://skills.example/dify_prompt/skills/list",
            {
                "headers": {"Content-Type": "application/json", "Authorization": "Bearer test-token"},
                "json": {"agent_name": "listing-agent", "env": "prod"},
                "timeout": (10, 60),
            },
        )
    ]


def test_load_skill_returns_rendered_prompt_text(monkeypatch) -> None:
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse({"code": 200, "data": {"rendered_text": "Complete skill prompt"}})

    monkeypatch.setattr("tools.skills.flyfus_skills.requests.post", post)

    messages = list(
        _tool().invoke(
            {"method": "load_skill", "agent_name": "listing-agent", "skill_name": "listing-diagnosis"}
        )
    )

    assert len(messages) == 1
    assert messages[0].message.text == "Complete skill prompt"
    assert calls[0][0] == "https://skills.example/dify_prompt/render"
    assert calls[0][1]["json"] == {
        "type": "skills",
        "text": "{{geo_prompt:listing-agent.listing-diagnosis@prod}}",
    }


def test_load_skill_requires_skill_name() -> None:
    messages = list(_tool().invoke({"method": "load_skill", "agent_name": "listing-agent"}))

    assert len(messages) == 1
    assert messages[0].message.text.startswith("Error: skill_name")
