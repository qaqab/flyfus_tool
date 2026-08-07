from __future__ import annotations

import hashlib
import io
import json
from types import SimpleNamespace

from dify_plugin.entities.tool import ToolInvokeMessage

from tools.image.flyfus_image_generate import FlyfusImageGenerateTool


def test_parse_urls_splits_multiline_items_inside_json_array() -> None:
    value = json.dumps(
        [
            "https://o1.flyfus.com/I/dz4pVY.png",
            (
                "https://o1.flyfus.com/F/wCS6Xn.jpg\n"
                "https://o1.flyfus.com/F/wm41xH.jpg\r\n"
                "https://o1.flyfus.com/F/0Fi3Rb.jpg"
            ),
        ]
    )

    assert FlyfusImageGenerateTool._parse_urls(value) == [
        "https://o1.flyfus.com/I/dz4pVY.png",
        "https://o1.flyfus.com/F/wCS6Xn.jpg",
        "https://o1.flyfus.com/F/wm41xH.jpg",
        "https://o1.flyfus.com/F/0Fi3Rb.jpg",
    ]


def test_image_generation_returns_urls_as_a_json_array(monkeypatch) -> None:
    received_args: dict = {}

    class FakeImages:
        def generate(self, **kwargs):
            received_args.update(kwargs)
            return SimpleNamespace(
                data=[
                    SimpleNamespace(b64_json="aW1hZ2UtMQ=="),
                    SimpleNamespace(b64_json="aW1hZ2UtMg=="),
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
        staticmethod(lambda upload, **kwargs: f"https://cdn.example/{upload[2]}"),
    )

    tool = FlyfusImageGenerateTool.from_credentials(
        {"api_key": "test-api-key", "endpoint_url": "https://images.example"}
    )
    messages = list(
        tool.invoke(
            {
                "prompt": "Two test images",
                "model": "gpt-image-2",
                "openai_4k_size": "3840x2160",
                "gemini_image_size": "2K",
                "gemini_aspect_ratio": "16:9",
            }
        )
    )

    assert received_args == {
        "model": "gpt-image-2",
        "prompt": "Two test images",
        "output_format": "png",
        "response_format": "b64_json",
    }
    assert len(messages) == 2
    message: ToolInvokeMessage = messages[0]
    assert message.message.json_object["urls"] == [
        "https://cdn.example/generated_image_1.png",
        "https://cdn.example/generated_image_2.png",
    ]
    assert message.message.json_object["log"]["log_id"]
    assert message.message.json_object["log"]["request_fingerprint"]
    assert (
        messages[1].message.text
        == '["https://cdn.example/generated_image_1.png", "https://cdn.example/generated_image_2.png"]'
    )


def test_gpt_image_2_4k_generation_passes_configured_parameters(monkeypatch) -> None:
    received_args: dict = {}

    class FakeImages:
        def generate(self, **kwargs):
            received_args.update(kwargs)
            return SimpleNamespace(data=[SimpleNamespace(b64_json="aW1hZ2U=")])

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
                "openai_4k_size": "2160x3840",
                "gemini_image_size": "2K",
                "gemini_aspect_ratio": "16:9",
            }
        )
    )

    assert received_args == {
        "model": "gpt-image-2-4k",
        "prompt": "4K cloudscape",
        "size": "2160x3840",
        "output_format": "png",
        "response_format": "b64_json",
    }
    assert messages[0].message.json_object["urls"] == [
        "https://cdn.example/image-4k.webp"
    ]


def test_gpt_image_2_4k_edit_uses_multipart_image_and_fixed_output(monkeypatch) -> None:
    received_args: dict = {}

    class FakeImages:
        def edit(self, **kwargs):
            received_args.update(
                {key: value for key, value in kwargs.items() if key != "image"}
            )
            received_args["image_name"] = kwargs["image"].name
            return SimpleNamespace(data=[SimpleNamespace(b64_json="aW1hZ2U=")])

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            self.images = FakeImages()

    def fake_download(*args, **kwargs) -> io.BytesIO:
        image = io.BytesIO(b"reference")
        image.name = "reference.png"
        return image

    monkeypatch.setattr(
        "tools.image.flyfus_image_generate.fetch_openai_model_ids",
        lambda endpoint_url, api_key: {"gpt-image-2-4k"},
    )
    monkeypatch.setattr("tools.image.flyfus_image_generate.OpenAI", FakeOpenAI)
    monkeypatch.setattr(
        FlyfusImageGenerateTool, "_download_input_image", staticmethod(fake_download)
    )
    monkeypatch.setattr(
        FlyfusImageGenerateTool,
        "_upload_output_to_oss",
        staticmethod(lambda upload, **kwargs: "https://cdn.example/edited.png"),
    )

    tool = FlyfusImageGenerateTool.from_credentials(
        {"api_key": "test-api-key", "endpoint_url": "https://images.example"}
    )
    messages = list(
        tool.invoke(
            {
                "prompt": "Edit the image",
                "model": "gpt-image-2-4k",
                "reference_image_urls": "https://images.example/reference.png",
                "openai_4k_size": "2048x1152",
            }
        )
    )

    assert received_args == {
        "model": "gpt-image-2-4k",
        "prompt": "Edit the image",
        "output_format": "png",
        "response_format": "b64_json",
        "size": "2048x1152",
        "image_name": "reference.png",
    }
    assert messages[0].message.json_object["urls"] == ["https://cdn.example/edited.png"]


def test_gemini_uses_native_payload_and_direct_https_reference_urls(
    monkeypatch,
) -> None:
    observed: dict = {}

    def fake_request(endpoint: str, api_key: str, payload: dict) -> SimpleNamespace:
        observed.update({"endpoint": endpoint, "api_key": api_key, "payload": payload})
        return SimpleNamespace(data=[SimpleNamespace(b64_json="aW1hZ2U=")])

    monkeypatch.setattr(
        "tools.image.flyfus_image_generate.fetch_openai_model_ids",
        lambda endpoint_url, api_key: {"gemini-3.1-flash-image-preview"},
    )
    monkeypatch.setattr(
        FlyfusImageGenerateTool, "_request_gemini", staticmethod(fake_request)
    )
    monkeypatch.setattr(
        FlyfusImageGenerateTool,
        "_upload_output_to_oss",
        staticmethod(lambda upload, **kwargs: "https://cdn.example/gemini.png"),
    )

    tool = FlyfusImageGenerateTool.from_credentials(
        {"api_key": "test-api-key", "endpoint_url": "https://images.example"}
    )
    messages = list(
        tool.invoke(
            {
                "prompt": "Restyle both images",
                "model": "gemini-3.1-flash-image-preview",
                "reference_image_urls": "https://cdn.example/a.png,https://cdn.example/b.jpg",
                "openai_4k_size": "3840x2160",
                "gemini_image_size": "2K",
                "gemini_aspect_ratio": "16:9",
            }
        )
    )

    assert observed == {
        "endpoint": "https://images.example/v1beta/models/gemini-3.1-flash-image-preview:generateContent",
        "api_key": "test-api-key",
        "payload": {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": "Restyle both images"},
                        {
                            "fileData": {
                                "mimeType": "image/png",
                                "fileUri": "https://cdn.example/a.png",
                            }
                        },
                        {
                            "fileData": {
                                "mimeType": "image/jpeg",
                                "fileUri": "https://cdn.example/b.jpg",
                            }
                        },
                    ],
                }
            ],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "imageConfig": {"imageSize": "2K", "aspectRatio": "16:9"},
                "candidateCount": 1,
            },
        },
    }
    assert messages[0].message.json_object["urls"] == ["https://cdn.example/gemini.png"]


def test_gemini_converts_webp_reference_to_inline_data(monkeypatch) -> None:
    image = io.BytesIO(b"webp-image")
    image.name = "reference.webp"
    monkeypatch.setattr(
        FlyfusImageGenerateTool,
        "_download_input_image",
        staticmethod(lambda *args, **kwargs: image),
    )

    payload, error = FlyfusImageGenerateTool._build_gemini_payload(
        "Edit this image",
        ["https://cdn.example/reference.webp"],
        {"gemini_image_size": "1K", "gemini_aspect_ratio": "1:1"},
    )

    assert error is None
    assert payload["contents"][0]["parts"][1] == {
        "inlineData": {
            "mimeType": "image/webp",
            "data": "d2VicC1pbWFnZQ==",
        }
    }
    assert image.closed


def test_gemini_native_response_is_normalized_to_image_data(monkeypatch) -> None:
    observed: dict = {}

    class FakeResponse:
        headers = {"x-request-id": "gemini-request-id"}

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "done"},
                                {
                                    "inlineData": {
                                        "mimeType": "image/webp",
                                        "data": "d2VicA==",
                                    }
                                },
                            ]
                        }
                    }
                ]
            }

    def fake_post(endpoint: str, **kwargs) -> FakeResponse:
        observed.update({"endpoint": endpoint, **kwargs})
        return FakeResponse()

    monkeypatch.setattr("tools.image.flyfus_image_generate.requests.post", fake_post)
    response = FlyfusImageGenerateTool._request_gemini(
        "https://images.example/v1beta/models/gemini:generateContent",
        "secret-key",
        {"contents": []},
    )

    assert response._request_id == "gemini-request-id"
    assert response.data[0].b64_json == "data:image/webp;base64,d2VicA=="
    assert observed == {
        "endpoint": "https://images.example/v1beta/models/gemini:generateContent",
        "headers": {
            "Accept": "application/json",
            "Authorization": "Bearer secret-key",
            "Content-Type": "application/json",
        },
        "json": {"contents": []},
        "timeout": (10.0, 300.0),
        "allow_redirects": False,
    }


def test_gemini_response_logs_redact_base64_image_data() -> None:
    image_data = "sensitive-base64-image-data"
    response = SimpleNamespace(
        data=[SimpleNamespace(b64_json=image_data)],
        _request_id="gemini-request-id",
    )

    fields = FlyfusImageGenerateTool._response_log_fields(response)
    response_body = json.loads(fields["upstream_response_body"])

    assert image_data not in fields["upstream_response_body"]
    assert response_body["data"][0]["b64_json"] == {
        "omitted": True,
        "characters": len(image_data),
        "sha256": hashlib.sha256(image_data.encode("utf-8")).hexdigest(),
    }


def test_image_generation_returns_an_empty_url_array_and_error_on_failure() -> None:
    tool = FlyfusImageGenerateTool.from_credentials({})
    messages = list(tool.invoke({"prompt": "A test image", "model": "gpt-image-2"}))

    assert len(messages) == 2
    assert messages[0].message.json_object["urls"] == []
    assert (
        messages[0].message.json_object["error"]
        == "API key is required for image generation."
    )
    assert messages[0].message.json_object["log"]["log_id"]
    assert messages[1].message.text == "[]"


def test_image_generation_logs_the_start_and_failure_of_every_invocation(
    monkeypatch,
) -> None:
    events: list[tuple[str, str, dict]] = []

    monkeypatch.setattr(
        "tools.image.flyfus_image_generate.write_tool_log",
        lambda credentials, log_id, event, **fields: events.append(
            (log_id, event, fields)
        ),
    )

    tool = FlyfusImageGenerateTool.from_credentials({})
    list(tool.invoke({"prompt": "A test image", "model": "gpt-image-2"}))

    assert [event for _, event, _ in events] == [
        "image_started",
        "image_validated",
        "image_failed",
    ]
    assert events[0][0] == events[-1][0]
    assert events[-1][2]["stage"] == "credentials"


def test_image_generation_retries_invalid_json_responses_three_times(
    monkeypatch,
) -> None:
    calls = 0
    events: list[tuple[str, dict]] = []

    class FakeImages:
        def generate(self, **kwargs):
            nonlocal calls
            calls += 1
            if calls <= 3:
                raise ValueError(
                    "Invalid JSON: expected value at line 1 column 1; input_value='<!DOCTYPE html>'"
                )
            return SimpleNamespace(data=[SimpleNamespace(b64_json="aW1hZ2U=")])

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            self.images = FakeImages()

    monkeypatch.setattr(
        "tools.image.flyfus_image_generate.fetch_openai_model_ids",
        lambda endpoint_url, api_key: {"gpt-image-2"},
    )
    monkeypatch.setattr("tools.image.flyfus_image_generate.OpenAI", FakeOpenAI)
    monkeypatch.setattr(
        "tools.image.flyfus_image_generate.time.sleep", lambda seconds: None
    )
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
    attempt_events = [
        (event, fields)
        for event, fields in events
        if event.startswith("image_request_attempt_")
    ]
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
    assert [fields["attempt"] for _, fields in attempt_events] == [
        1,
        1,
        2,
        2,
        3,
        3,
        4,
        4,
    ]
    assert all(
        "elapsed_ms" in fields
        for event, fields in attempt_events
        if event.endswith(("failed", "succeeded"))
    )
    assert len({fields["request_fingerprint"] for _, fields in attempt_events}) == 1
    assert messages[0].message.json_object["urls"] == [
        "https://cdn.example/recovered.png"
    ]
    assert messages[0].message.json_object["log"]["request_fingerprint"]
    assert messages[1].message.text == '["https://cdn.example/recovered.png"]'


def test_image_request_retries_an_upstream_error_payload_validation_failure(
    monkeypatch,
) -> None:
    calls = 0
    retries: list[int] = []

    def request() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError(
                "input_value={'error': {'message': 'Upstream request failed', 'type': 'upstream_error'}}"
            )
        return "recovered"

    monkeypatch.setattr(
        "tools.image.flyfus_image_generate.time.sleep", lambda seconds: None
    )

    response = FlyfusImageGenerateTool._run_image_request_with_retry(
        request,
        on_retry=lambda attempt, error: retries.append(attempt),
    )

    assert response == "recovered"
    assert calls == 2
    assert retries == [1]


def test_image_request_retries_when_the_response_data_is_missing(monkeypatch) -> None:
    calls = 0
    retries: list[int] = []

    def request() -> SimpleNamespace:
        nonlocal calls
        calls += 1
        data = None if calls == 1 else []
        return FlyfusImageGenerateTool._require_image_data(SimpleNamespace(data=data))

    monkeypatch.setattr(
        "tools.image.flyfus_image_generate.time.sleep", lambda seconds: None
    )

    response = FlyfusImageGenerateTool._run_image_request_with_retry(
        request,
        on_retry=lambda attempt, error: retries.append(attempt),
    )

    assert response.data == []
    assert calls == 2
    assert retries == [1]


def test_image_error_logs_the_upstream_response_header_request_id() -> None:
    class GatewayTimeout(Exception):
        def __init__(self) -> None:
            self.status_code = 524
            self.response = SimpleNamespace(
                headers={
                    "x-request-id": "gateway-request-123",
                    "server": "edge-gateway",
                    "set-cookie": "session=secret",
                }
            )

    fields = FlyfusImageGenerateTool._error_log_fields(GatewayTimeout())

    assert fields["status_code"] == 524
    assert fields["exception_message"] == ""
    assert fields["upstream_header_request_id"] == "gateway-request-123"
    assert json.loads(fields["upstream_response_headers"]) == {
        "server": "edge-gateway",
        "set-cookie": "[REDACTED]",
        "x-request-id": "gateway-request-123",
    }


def test_image_generation_logs_empty_upstream_responses(monkeypatch) -> None:
    events: list[tuple[str, dict]] = []

    class FakeImages:
        def generate(self, **kwargs):
            return SimpleNamespace(
                data=[],
                _request_id="upstream-empty-response",
                _response=SimpleNamespace(
                    status_code=200,
                    headers={"x-request-id": "upstream-header-request-id"},
                ),
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
        "tools.image.flyfus_image_generate.write_tool_log",
        lambda credentials, log_id, event, **fields: events.append((event, fields)),
    )

    tool = FlyfusImageGenerateTool.from_credentials(
        {"api_key": "test-api-key", "endpoint_url": "https://images.example"}
    )
    messages = list(
        tool.invoke({"prompt": "Empty response test", "model": "gpt-image-2"})
    )

    response_empty = next(
        fields for event, fields in events if event == "image_response_empty"
    )
    assert response_empty["upstream_request_id"] == "upstream-empty-response"
    assert response_empty["response_type"] == "SimpleNamespace"
    assert response_empty["response_data_type"] == "list"
    assert response_empty["response_data_count"] == 0
    assert json.loads(response_empty["upstream_response_body"]) == {
        "_request_id": "upstream-empty-response",
        "data": [],
    }
    assert response_empty["upstream_response_body_source"] == "sdk_parsed"
    assert response_empty["upstream_status_code"] == 200
    assert response_empty["upstream_header_request_id"] == "upstream-header-request-id"
    assert json.loads(response_empty["upstream_response_headers"]) == {
        "x-request-id": "upstream-header-request-id"
    }
    assert response_empty["request_fingerprint"]
    assert messages[0].message.json_object["urls"] == []
    assert (
        messages[0].message.json_object["error"]
        == "The image model did not return any images."
    )
    assert messages[0].message.json_object["log"]["log_id"]
    assert (
        messages[0].message.json_object["log"]["request_fingerprint"]
        == response_empty["request_fingerprint"]
    )
