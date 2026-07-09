from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

IMAGE_MODELS_DIR = Path(__file__).resolve().parents[1] / "models" / "image"


def normalize_openai_base_url(base_url: str | None) -> str | None:
    if not base_url:
        return None

    cleaned = str(base_url).strip().rstrip("/")
    if not cleaned:
        return None

    if cleaned.endswith("/v1"):
        return cleaned
    return f"{cleaned}/v1"


def decode_image(base64_image: str) -> tuple[str, bytes]:
    if not base64_image.startswith("data:image"):
        return "image/png", base64.b64decode(base64_image)

    try:
        mime_type = base64_image.split(";")[0].split(":")[1]
        image_data_base64 = base64_image.split(",", 1)[1]
        return mime_type, base64.b64decode(image_data_base64)
    except (IndexError, ValueError):
        return "image/png", base64.b64decode(base64_image.split(",")[-1])


def build_usage_metadata(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if not usage:
        return {}

    token_usage: dict[str, Any] = {}
    for key in ("total_tokens", "input_tokens", "output_tokens"):
        if hasattr(usage, key):
            token_usage[key] = getattr(usage, key)

    details = getattr(usage, "input_tokens_details", None)
    if details:
        token_usage["input_tokens_details"] = {
            "text_tokens": getattr(details, "text_tokens", None),
            "image_tokens": getattr(details, "image_tokens", None),
        }

    return {"token_usage": token_usage} if token_usage else {}


def build_usage_output(response: Any, model: str, operation: str, image_count: int) -> dict[str, Any] | None:
    usage_metadata = build_usage_metadata(response)
    token_usage = usage_metadata.get("token_usage")
    if not token_usage:
        return None

    return {
        "data": [
            {
                "model": model,
                "operation": operation,
                "image_count": image_count,
                "usage": token_usage,
            }
        ]
    }


@lru_cache(maxsize=1)
def load_image_model_schemas() -> dict[str, dict[str, Any]]:
    schemas: dict[str, dict[str, Any]] = {}
    for schema_file in sorted(IMAGE_MODELS_DIR.glob("*.yaml")):
        with schema_file.open("r", encoding="utf-8") as file:
            schema = yaml.safe_load(file) or {}
        model_id = str(schema.get("model") or "").strip()
        if not model_id:
            continue
        schemas[model_id] = schema
    return schemas


def image_model_ids() -> frozenset[str]:
    return frozenset(load_image_model_schemas())


def image_model_supports_operation(model: str, operation: str) -> bool:
    schema = load_image_model_schemas().get(model) or {}
    return operation in set(schema.get("supported_operations") or [])


def extract_model_ids(models_response: Any) -> set[str]:
    data = getattr(models_response, "data", models_response)
    if isinstance(data, dict):
        data = data.get("data", [])

    model_ids: set[str] = set()
    for item in data or []:
        model_id = getattr(item, "id", None)
        if model_id is None and isinstance(item, dict):
            model_id = item.get("id")
        if model_id:
            model_ids.add(str(model_id))
    return model_ids
