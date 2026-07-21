from __future__ import annotations

from types import SimpleNamespace

from dify_plugin.entities.tool import ToolInvokeMessage

from tools.image.flyfus_image_generate import FlyfusImageGenerateTool


def test_image_generation_returns_urls_as_a_json_array(monkeypatch) -> None:
    class FakeImages:
        def generate(self, **kwargs):
            assert kwargs["model"] == "gpt-image-2"
            return SimpleNamespace(
                data=[
                    SimpleNamespace(url="https://upstream.example/image-1.png"),
                    SimpleNamespace(url="https://upstream.example/image-2.png"),
                ]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            self.images = FakeImages()

    monkeypatch.setattr(
        "tools.image.flyfus_image_generate.fetch_openai_model_ids",
        lambda endpoint_url, api_key: {"gpt-image-2"},
    )
    monkeypatch.setattr("tools.image.flyfus_image_generate.OpenAI", FakeOpenAI)
    monkeypatch.setattr(
        FlyfusImageGenerateTool,
        "_upload_output_to_oss",
        staticmethod(lambda upload, **kwargs: f"https://cdn.example/{upload[1].rsplit('/', 1)[-1]}"),
    )

    tool = FlyfusImageGenerateTool.from_credentials(
        {"api_key": "test-api-key", "endpoint_url": "https://images.example"}
    )
    messages = list(tool.invoke({"prompt": "Two test images", "model": "gpt-image-2"}))

    assert len(messages) == 2
    message: ToolInvokeMessage = messages[0]
    assert message.message.json_object["urls"] == [
        "https://cdn.example/image-1.png",
        "https://cdn.example/image-2.png",
    ]
    assert message.message.json_object["log"]["log_id"]
    assert message.message.json_object["log"]["request_fingerprint"]
    assert messages[1].message.text == '["https://cdn.example/image-1.png", "https://cdn.example/image-2.png"]'


def test_gpt_image_2_4k_generation_passes_configured_parameters(monkeypatch) -> None:
    received_args: dict = {}

    class FakeImages:
        def generate(self, **kwargs):
            received_args.update(kwargs)
            return SimpleNamespace(data=[SimpleNamespace(url="https://upstream.example/image-4k.webp")])

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            self.images = FakeImages()

    monkeypatch.setattr(
        "tools.image.flyfus_image_generate.fetch_openai_model_ids",
        lambda endpoint_url, api_key: {"gpt-image-2-4k"},
    )
    monkeypatch.setattr("tools.image.flyfus_image_generate.OpenAI", FakeOpenAI)
    monkeypatch.setattr(
        FlyfusImageGenerateTool,
        "_upload_output_to_oss",
        staticmethod(lambda upload, **kwargs: "https://cdn.example/image-4k.webp"),
    )

    tool = FlyfusImageGenerateTool.from_credentials(
        {"api_key": "test-api-key", "endpoint_url": "https://images.example"}
    )
    messages = list(
        tool.invoke(
            {
                "prompt": "4K cloudscape",
                "model": "gpt-image-2-4k",
                "size": "3840x2160",
                "output_format": "webp",
                "moderation": "low",
            }
        )
    )

    assert received_args == {
        "model": "gpt-image-2-4k",
        "prompt": "4K cloudscape",
        "size": "3840x2160",
        "n": 1,
        "output_format": "webp",
        "moderation": "low",
    }
    assert messages[0].message.json_object["urls"] == ["https://cdn.example/image-4k.webp"]


def test_image_generation_returns_an_empty_url_array_and_error_on_failure() -> None:
    tool = FlyfusImageGenerateTool.from_credentials({})
    messages = list(tool.invoke({"prompt": "A test image", "model": "gpt-image-2"}))

    assert len(messages) == 2
    assert messages[0].message.json_object["urls"] == []
    assert messages[0].message.json_object["error"] == "API key is required for image generation."
    assert messages[0].message.json_object["log"]["log_id"]
    assert messages[1].message.text == "[]"


def test_image_generation_logs_the_start_and_failure_of_every_invocation(monkeypatch) -> None:
    events: list[tuple[str, str, dict]] = []

    monkeypatch.setattr(
        "tools.image.flyfus_image_generate.write_tool_log",
        lambda credentials, log_id, event, **fields: events.append((log_id, event, fields)),
    )

    tool = FlyfusImageGenerateTool.from_credentials({})
    list(tool.invoke({"prompt": "A test image", "model": "gpt-image-2"}))

    assert [event for _, event, _ in events] == ["image_started", "image_validated", "image_failed"]
    assert events[0][0] == events[-1][0]
    assert events[-1][2]["stage"] == "credentials"


def test_image_generation_retries_invalid_json_responses_three_times(monkeypatch) -> None:
    calls = 0
    events: list[tuple[str, dict]] = []

    class FakeImages:
        def generate(self, **kwargs):
            nonlocal calls
            calls += 1
            if calls <= 3:
                raise ValueError("Invalid JSON: expected value at line 1 column 1; input_value='<!DOCTYPE html>'")
            return SimpleNamespace(data=[SimpleNamespace(url="https://upstream.example/recovered.png")])

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            self.images = FakeImages()

    monkeypatch.setattr(
        "tools.image.flyfus_image_generate.fetch_openai_model_ids",
        lambda endpoint_url, api_key: {"gpt-image-2"},
    )
    monkeypatch.setattr("tools.image.flyfus_image_generate.OpenAI", FakeOpenAI)
    monkeypatch.setattr("tools.image.flyfus_image_generate.time.sleep", lambda seconds: None)
    monkeypatch.setattr(
        "tools.image.flyfus_image_generate.write_tool_log",
        lambda credentials, log_id, event, **fields: events.append((event, fields)),
    )
    monkeypatch.setattr(
        FlyfusImageGenerateTool,
        "_upload_output_to_oss",
        staticmethod(lambda upload, **kwargs: "https://cdn.example/recovered.png"),
    )

    tool = FlyfusImageGenerateTool.from_credentials(
        {"api_key": "test-api-key", "endpoint_url": "https://images.example"}
    )
    messages = list(tool.invoke({"prompt": "Retry test", "model": "gpt-image-2"}))

    assert calls == 4
    assert [event for event, _ in events].count("image_request_retry") == 3
    attempt_events = [(event, fields) for event, fields in events if event.startswith("image_request_attempt_")]
    assert [event for event, _ in attempt_events] == [
        "image_request_attempt_started",
        "image_request_attempt_failed",
        "image_request_attempt_started",
        "image_request_attempt_failed",
        "image_request_attempt_started",
        "image_request_attempt_failed",
        "image_request_attempt_started",
        "image_request_attempt_succeeded",
    ]
    assert [fields["attempt"] for _, fields in attempt_events] == [1, 1, 2, 2, 3, 3, 4, 4]
    assert all("elapsed_ms" in fields for event, fields in attempt_events if event.endswith(("failed", "succeeded")))
    assert len({fields["request_fingerprint"] for _, fields in attempt_events}) == 1
    assert messages[0].message.json_object["urls"] == ["https://cdn.example/recovered.png"]
    assert messages[0].message.json_object["log"]["request_fingerprint"]
    assert messages[1].message.text == '["https://cdn.example/recovered.png"]'


def test_image_generation_logs_empty_upstream_responses(monkeypatch) -> None:
    events: list[tuple[str, dict]] = []

    class FakeImages:
        def generate(self, **kwargs):
            return SimpleNamespace(data=[], _request_id="upstream-empty-response")

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            self.images = FakeImages()

    monkeypatch.setattr(
        "tools.image.flyfus_image_generate.fetch_openai_model_ids",
        lambda endpoint_url, api_key: {"gpt-image-2"},
    )
    monkeypatch.setattr("tools.image.flyfus_image_generate.OpenAI", FakeOpenAI)
    monkeypatch.setattr(
        "tools.image.flyfus_image_generate.write_tool_log",
        lambda credentials, log_id, event, **fields: events.append((event, fields)),
    )

    tool = FlyfusImageGenerateTool.from_credentials(
        {"api_key": "test-api-key", "endpoint_url": "https://images.example"}
    )
    messages = list(tool.invoke({"prompt": "Empty response test", "model": "gpt-image-2"}))

    response_empty = next(fields for event, fields in events if event == "image_response_empty")
    assert response_empty["upstream_request_id"] == "upstream-empty-response"
    assert response_empty["request_fingerprint"]
    assert messages[0].message.json_object["urls"] == []
    assert messages[0].message.json_object["error"] == "The image model did not return any images."
    assert messages[0].message.json_object["log"]["log_id"]
    assert messages[0].message.json_object["log"]["request_fingerprint"] == response_empty["request_fingerprint"]
