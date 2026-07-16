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
        staticmethod(lambda upload: f"https://cdn.example/{upload[1].rsplit('/', 1)[-1]}"),
    )

    tool = FlypowerImageGenerateTool.from_credentials(
        {"api_key": "test-api-key", "endpoint_url": "https://images.example"}
    )
    messages = list(tool.invoke({"prompt": "Two test images", "model": "gpt-image-2"}))

    assert len(messages) == 1
    message: ToolInvokeMessage = messages[0]
    assert message.message.json_object == {
        "urls": [
            "https://cdn.example/image-1.png",
            "https://cdn.example/image-2.png",
        ]
    }
