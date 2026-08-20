from __future__ import annotations

from tools.memory.flyfus_memory import FlyfusMemoryTool


USER_ID = "87:web_chat:7841eab3-ca44-4522-9956-840a4d749924"


class FakeResponse:
    def __init__(self, payload: object, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> object:
        return self._payload


def _tool() -> FlyfusMemoryTool:
    return FlyfusMemoryTool.from_credentials(
        {"geo_url": "https://geo.example/api/geo/v2", "geo_key": "test-key"}
    )


def test_list_scenarios_parses_dify_user_id_and_returns_data(monkeypatch) -> None:
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse(
            {
                "code": 200,
                "message": "success",
                "data": {"total": 1, "items": [{"path": "listing/title.md"}]},
            }
        )

    monkeypatch.setattr("tools.memory.flyfus_memory.requests.post", post)

    result = list(
        _tool().invoke(
            {"method": "list_scenarios", "user_id": USER_ID, "path_prefix": "listing"}
        )
    )[0].message.json_object

    assert result == {"total": 1, "items": [{"path": "listing/title.md"}]}
    assert calls == [
        (
            "https://geo.example/api/geo/v2/chat/memory/scenarios",
            {
                "headers": {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer test-key",
                },
                "json": {"user_id": 87, "path_prefix": "listing"},
                "timeout": (10, 60),
            },
        )
    ]


def test_read_scenario_uses_list_path_and_preserves_null_content(monkeypatch) -> None:
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse(
            {
                "code": 200,
                "message": "success",
                "data": {"path": "listing/title.md", "content": None, "version": 1},
            }
        )

    monkeypatch.setattr(
        "tools.memory.flyfus_memory.requests.post",
        post,
    )

    result = list(
        _tool().invoke(
            {"method": "read_scenario", "user_id": USER_ID, "path": "listing/title.md"}
        )
    )[0].message.json_object

    assert result == {"path": "listing/title.md", "content": None, "version": 1}
    assert calls[0][0] == "https://geo.example/api/geo/v2/chat/memory/scenarios/read"
    assert calls[0][1]["json"] == {"user_id": 87, "path": "listing/title.md"}


def test_invalid_user_id_does_not_call_api(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        "tools.memory.flyfus_memory.requests.post",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    for user_id in ("87", "87:api:7841eab3-ca44-4522-9956-840a4d749924", "bad"):
        message = list(_tool().invoke({"method": "list_scenarios", "user_id": user_id}))[0]
        assert message.message.text == "Error: user_id 不正确"
    assert calls == []


def test_read_rejects_unsafe_path_without_calling_api(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        "tools.memory.flyfus_memory.requests.post",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    for path in ("", "/listing/title.md", "listing\\title.md", "listing/../title.md", "listing/a..md"):
        message = list(
            _tool().invoke({"method": "read_scenario", "user_id": USER_ID, "path": path})
        )[0]
        assert message.message.text == "Error: path 不正确"
    assert calls == []


def test_business_error_is_returned_as_error_message(monkeypatch) -> None:
    monkeypatch.setattr(
        "tools.memory.flyfus_memory.requests.post",
        lambda url, **kwargs: FakeResponse({"code": 412, "message": "user_id invalid", "data": None}),
    )

    message = list(_tool().invoke({"method": "list_scenarios", "user_id": USER_ID}))[0]

    assert message.message.text == "Error: 场景记忆接口业务码 412：user_id invalid"
