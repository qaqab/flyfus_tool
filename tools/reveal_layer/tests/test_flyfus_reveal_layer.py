from __future__ import annotations

import importlib.util
from io import BytesIO
import json
from pathlib import Path

from PIL import Image
import pytest

from tools.reveal_layer.flyfus_reveal_layer import (
    FlyfusRevealLayerTool,
    build_submit_payload,
    _log_json,
    parse_layers_json,
)


def test_module_loads_without_sys_modules_registration() -> None:
    module_path = Path(__file__).parents[1] / "flyfus_reveal_layer.py"
    spec = importlib.util.spec_from_file_location("dify_dynamic_reveal_layer", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    assert module.LayerSpec("主体", (0, 0, 100, 100)).name == "主体"


def test_parse_layers_json_and_build_crop_payload() -> None:
    layers = parse_layers_json(
        json.dumps([{"name": "主产品", "bbox": [10, 20, 300, 400]}], ensure_ascii=False)
    )

    payload = build_submit_payload(
        "data:image/png;base64,abc",
        layers,
        version="v2.3",
        pipeline_type="crop",
        steps=10,
        seed=42,
    )

    assert layers[0].name == "主产品"
    assert payload["image_boxes"] == [[10, 20, 300, 400]]
    assert payload["steps"] == 10
    assert payload["seed"] == 42


def test_all_payload_omits_image_boxes() -> None:
    payload = build_submit_payload(
        "data:image/png;base64,abc",
        [],
        version="v2.4-test",
        pipeline_type="all",
        steps=20,
        seed=7,
    )

    assert payload["pipeline_type"] == "all"
    assert payload["version"] == "v2.4-test"
    assert "image_boxes" not in payload


def test_tiny_box_is_rejected_before_reveal_request() -> None:
    layers = parse_layers_json('[{"name":"小图标","bbox":[0,0,20,20]}]')

    with pytest.raises(ValueError, match="bbox 太小"):
        FlyfusRevealLayerTool._validate_layers(layers, (5000, 3500))


def test_returned_layer_count_may_differ_by_one_but_not_more(monkeypatch) -> None:
    source = Image.new("RGB", (128, 128), "white")
    image = Image.new("RGBA", (128, 128), (255, 255, 255, 255))
    monkeypatch.setattr(
        FlyfusRevealLayerTool,
        "_download_output_image",
        staticmethod(lambda url: image.copy()),
    )
    tool = FlyfusRevealLayerTool.from_credentials({})
    two_layers = parse_layers_json(
        '[{"name":"A","bbox":[0,0,64,64]},{"name":"B","bbox":[64,64,128,128]}]'
    )
    delivery_images = tool._build_delivery_images(
        source,
        two_layers,
        {"output": {"layers_base": ["bg", "one"]}},
    )
    assert len(delivery_images) - 1 == 1

    three_layers = parse_layers_json(
        '[{"name":"A","bbox":[0,0,64,64]},{"name":"B","bbox":[32,32,96,96]},'
        '{"name":"C","bbox":[64,64,128,128]}]'
    )
    with pytest.raises(RuntimeError, match="数量差异超过 1"):
        tool._build_delivery_images(
            source,
            three_layers,
            {"output": {"layers_base": ["bg", "one"]}},
        )


def test_delivery_images_contain_background_and_original_rgb_layers(monkeypatch) -> None:
    source = Image.new("RGB", (4, 4))
    source.putdata([(index, index + 1, index + 2) for index in range(16)])
    background = Image.new("RGBA", (4, 4), (240, 240, 240, 255))
    first = Image.new("RGBA", (4, 4), (99, 88, 77, 0))
    first.putalpha(Image.new("L", (4, 4), 255))
    second = Image.new("RGBA", (4, 4), (66, 55, 44, 0))
    second.putalpha(Image.new("L", (4, 4), 128))
    images = {"bg": background, "first": first, "second": second}
    monkeypatch.setattr(
        FlyfusRevealLayerTool,
        "_download_output_image",
        staticmethod(lambda url: images[url].copy()),
    )
    tool = FlyfusRevealLayerTool.from_credentials({})
    layers = parse_layers_json(
        '[{"name":"A","bbox":[0,0,2,2]},{"name":"B","bbox":[2,2,4,4]}]'
    )

    delivery_images = tool._build_delivery_images(
        source,
        layers,
        {
            "output": {
                "layers_base": ["bg", "first", "second"],
                "boxes_mapping_index": [1, 0],
            }
        },
    )

    assert [name for name, _ in delivery_images] == [
        "background.png",
        "layer_01_B.png",
        "layer_02_A.png",
    ]
    with Image.open(BytesIO(delivery_images[1][1])) as result:
        assert result.convert("RGB").tobytes() == source.tobytes()
        assert set(result.getchannel("A").tobytes()) == {255}


def test_sync_tool_returns_uploaded_image_urls(monkeypatch) -> None:
    captured: dict = {}
    logs: list[tuple[str, dict]] = []
    source = Image.new("RGB", (128, 128), "white")
    png_header = b"\x89PNG\r\n\x1a\n"

    monkeypatch.setattr(
        FlyfusRevealLayerTool,
        "_download_source",
        staticmethod(lambda url: (b"image", "image/png", source)),
    )

    def fake_submit(self, payload, version, log_id):
        captured.update({"payload": payload, "version": version, "log_id": log_id})
        return {"generation_time": 12.5, "output": {"layers_base": ["bg", "fg"]}}

    monkeypatch.setattr(FlyfusRevealLayerTool, "_submit_and_wait", fake_submit)
    monkeypatch.setattr(
        FlyfusRevealLayerTool,
        "_build_delivery_images",
        lambda self, source_rgb, layers, completed: [
            ("background.png", png_header + b"background"),
            ("layer_01_main.png", png_header + b"foreground"),
        ],
    )
    monkeypatch.setattr(
        FlyfusRevealLayerTool,
        "_upload_images",
        lambda self, images, log_id: [
            "https://cdn.example/background.png",
            "https://cdn.example/layer_01.png",
        ],
    )
    monkeypatch.setattr(
        FlyfusRevealLayerTool,
        "_write_log",
        lambda self, log_id, event, **fields: logs.append((event, fields)),
    )
    tool = FlyfusRevealLayerTool.from_credentials({"reveal_layer_api_key": "test"})

    messages = list(
        tool.invoke(
            {
                "image_url": "https://images.example/source.png",
                "layers_json": '[{"name":"主产品","bbox":[0,0,100,100]}]',
                "version": "v2.3",
                "pipeline_type": "crop",
                "steps": 10,
                "seed": 42,
                "debug": False,
            }
        )
    )

    result = messages[0].message.json_object
    assert result["log_id"] == captured["log_id"]
    assert result["boxed_image_url"] == ""
    assert result["image_urls"] == [
        "https://cdn.example/background.png",
        "https://cdn.example/layer_01.png",
    ]
    assert result["source_size"] == [128, 128]
    text_result = json.loads(messages[1].message.text)
    assert text_result["log_id"] == result["log_id"]
    assert text_result["image_urls"] == result["image_urls"]
    assert captured["payload"]["image_boxes"] == [[0, 0, 100, 100]]
    assert [event for event, _ in logs] == [
        "input_checked",
        "submit_request",
        "delivery_images_created",
        "succeeded",
    ]
    assert logs[0][1]["image_name"] == "source.png"
    assert "image_base64" not in logs[1][1]["request_json"]
    assert "background.png" in logs[2][1]["files_json"]


def test_images_upload_concurrently_to_general_file_endpoint(monkeypatch) -> None:
    observed: list[dict] = []
    logs: list[tuple[str, dict]] = []

    class FakeResponse:
        status_code = 200

        def __init__(self, public_url: str) -> None:
            self.public_url = public_url

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"success": True, "data": {"public_url": self.public_url}}

    def fake_post(url, **kwargs):
        filename = kwargs["files"]["file"][0]
        observed.append({"url": url, **kwargs})
        return FakeResponse(f"https://cdn.example/{filename}")

    monkeypatch.setattr("tools.reveal_layer.flyfus_reveal_layer.requests.post", fake_post)
    monkeypatch.setattr(
        FlyfusRevealLayerTool,
        "_write_log",
        lambda self, log_id, event, **fields: logs.append((event, fields)),
    )
    tool = FlyfusRevealLayerTool.from_credentials(
        {"oss_api_base_url": "https://oss.example", "oss_api_token": "token"}
    )

    result = tool._upload_images(
        [("background.png", b"background"), ("layer_01.png", b"foreground")],
        "log-id",
    )

    assert len(result) == 2
    assert all(url.endswith(".png") for url in result)
    assert all(url.rsplit("/", 1)[-1].isascii() for url in result)
    assert len(observed) == 2
    assert all(
        request["url"] == "https://oss.example/v1/oss-assets/file-file/upload"
        for request in observed
    )
    assert all(request["files"]["file"][2] == "image/png" for request in observed)
    assert [event for event, _ in logs] == [
        "oss_batch_upload_request",
        "oss_batch_upload_response",
    ]
    assert logs[0][1]["file_count"] == 2
    assert logs[1][1]["file_count"] == 2


def test_debug_preview_draws_numbered_box() -> None:
    source = Image.new("RGB", (200, 200), "white")
    layers = parse_layers_json('[{"name":"产品","bbox":[20,30,120,150]}]')

    preview = FlyfusRevealLayerTool._build_boxed_preview(source, layers)

    assert preview.getpixel((20, 100)) != (255, 255, 255)
    assert preview.getpixel((120, 100)) != (255, 255, 255)


def test_submit_and_final_query_responses_are_logged(monkeypatch) -> None:
    responses = [
        {"response_status": 0, "task_id": "task-123"},
        {
            "response_status": 0,
            "status": "done",
            "generation_time": 9.5,
            "output": {
                "layers_base": [
                    "https://cdn.example/background.png?signature=secret",
                    "https://cdn.example/foreground.png?signature=secret",
                ],
                "boxes_mapping_index": [0],
            },
        },
    ]
    logs: list[tuple[str, dict]] = []
    tool = FlyfusRevealLayerTool.from_credentials({"reveal_layer_api_key": "test-key"})
    monkeypatch.setattr(tool, "_post", lambda *args, **kwargs: responses.pop(0))
    monkeypatch.setattr(
        tool,
        "_write_log",
        lambda log_id, event, **fields: logs.append((event, fields)),
    )

    result = tool._submit_and_wait({}, "v2.3", "log-id")

    assert result["status"] == "done"
    assert [event for event, _ in logs] == ["submit_response", "query_response"]
    assert logs[0][1]["response_json"] == '{"response_status":0,"task_id":"task-123"}'
    assert "signature=secret" not in logs[1][1]["response_json"]
    assert "background.png" in logs[1][1]["response_json"]
    assert logs[1][1]["poll_count"] == 1


def test_log_json_replaces_image_data_and_urls_with_names() -> None:
    logged = _log_json(
        {
            "input": "data:image/png;base64,secret-image-data",
            "layers_base": [
                "https://cdn.example/path/background.png?signature=secret",
                "https://cdn.example/path/foreground.webp?signature=secret",
            ],
            "task_id": "task-123",
        }
    )

    assert "secret-image-data" not in logged
    assert "signature=secret" not in logged
    assert "background.png" in logged
    assert "foreground.webp" in logged
    assert "task-123" in logged
