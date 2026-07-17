from __future__ import annotations

from types import SimpleNamespace

from dify_plugin.entities.tool import ToolInvokeMessage

from tools.image.flypower_image_generate import FlypowerImageGenerateTool


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
        "tools.image.flypower_image_generate.fetch_openai_model_ids",
        lambda endpoint_url, api_key: {"gpt-image-2"},
    )
    monkeypatch.setattr("tools.image.flypower_image_generate.OpenAI", FakeOpenAI)
    monkeypatch.setattr(
        FlypowerImageGenerateTool,
        "_upload_output_to_oss",
        staticmethod(lambda upload, **kwargs: f"https://cdn.example/{upload[1].rsplit('/', 1)[-1]}"),
    )

    tool = FlypowerImageGenerateTool.from_credentials(
        {"api_key": "test-api-key", "endpoint_url": "https://images.example"}
    )
    messages = list(tool.invoke({"prompt": "Two test images", "model": "gpt-image-2"}))

    assert len(messages) == 2
    message: ToolInvokeMessage = messages[0]
    assert message.message.json_object == {
        "urls": [
            "https://cdn.example/image-1.png",
            "https://cdn.example/image-2.png",
        ]
    }
    assert messages[1].message.text == '["https://cdn.example/image-1.png", "https://cdn.example/image-2.png"]'


def test_image_generation_returns_an_empty_url_array_and_error_on_failure() -> None:
    tool = FlypowerImageGenerateTool.from_credentials({})
    messages = list(tool.invoke({"prompt": "A test image", "model": "gpt-image-2"}))

    assert len(messages) == 2
    assert messages[0].message.json_object == {
        "urls": [],
        "error": "API key is required for image generation.",
    }
    assert messages[1].message.text == "[]"


def test_image_generation_logs_the_start_and_failure_of_every_invocation(monkeypatch) -> None:
    events: list[tuple[str, str, dict]] = []

    monkeypatch.setattr(
        "tools.image.flypower_image_generate.write_tool_log",
        lambda credentials, log_id, event, **fields: events.append((log_id, event, fields)),
    )

    tool = FlypowerImageGenerateTool.from_credentials({})
    list(tool.invoke({"prompt": "A test image", "model": "gpt-image-2"}))

    assert [event for _, event, _ in events] == ["image_started", "image_validated", "image_failed"]
    assert events[0][0] == events[-1][0]
    assert events[-1][2]["stage"] == "credentials"


def test_image_generation_retries_invalid_json_responses_three_times(monkeypatch) -> None:
    calls = 0
    events: list[str] = []

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
        "tools.image.flypower_image_generate.fetch_openai_model_ids",
        lambda endpoint_url, api_key: {"gpt-image-2"},
    )
    monkeypatch.setattr("tools.image.flypower_image_generate.OpenAI", FakeOpenAI)
    monkeypatch.setattr("tools.image.flypower_image_generate.time.sleep", lambda seconds: None)
    monkeypatch.setattr(
        "tools.image.flypower_image_generate.write_tool_log",
        lambda credentials, log_id, event, **fields: events.append(event),
    )
    monkeypatch.setattr(
        FlypowerImageGenerateTool,
        "_upload_output_to_oss",
        staticmethod(lambda upload, **kwargs: "https://cdn.example/recovered.png"),
    )

    tool = FlypowerImageGenerateTool.from_credentials(
        {"api_key": "test-api-key", "endpoint_url": "https://images.example"}
    )
    messages = list(tool.invoke({"prompt": "Retry test", "model": "gpt-image-2"}))

    assert calls == 4
    assert events.count("image_request_retry") == 3
    assert messages[0].message.json_object == {"urls": ["https://cdn.example/recovered.png"]}
    assert messages[1].message.text == '["https://cdn.example/recovered.png"]'
