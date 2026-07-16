from __future__ import annotations

import base64
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Literal
from urllib.parse import urlsplit, urlunsplit

import requests
import yaml

IMAGE_MODELS_DIR = Path(__file__).resolve().parents[1] / "models" / "image"
MODEL_LIST_TIMEOUT: Final = (10.0, 30.0)


@dataclass(frozen=True, slots=True)
class ModelListRequestError(RuntimeError):
    category: Literal["http", "invalid_response", "network", "timeout"]
    endpoint: str
    status_code: int | None = None

    def __str__(self) -> str:
        if self.status_code is not None:
            return f"{self.category} error from {self.endpoint} (HTTP {self.status_code})"
        return f"{self.category} error from {self.endpoint}"


def normalize_openai_base_url(base_url: str | None) -> str | None:
    if not base_url:
        return None

    cleaned = str(base_url).strip()
    if not cleaned:
        return None

    parsed = urlsplit(cleaned)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("API 地址必须是 HTTPS URL")
    if "@" in parsed.netloc:
        raise ValueError("API 地址不能包含用户名或密码")
    if parsed.query or parsed.fragment:
        raise ValueError("API 地址不能包含查询参数或片段")

    path = parsed.path.rstrip("/")
    versioned_path = path if path.endswith("/v1") else f"{path}/v1"
    return urlunsplit(("https", parsed.netloc, versioned_path, "", ""))


def fetch_openai_model_ids(endpoint_url: str, api_key: str) -> set[str]:
    normalized_base_url = normalize_openai_base_url(endpoint_url)
    if normalized_base_url is None:
        raise ValueError("请填写 API 地址")

    models_endpoint = f"{normalized_base_url}/models"
    try:
        response = requests.get(
            models_endpoint,
            headers={"Accept": "application/json", "Authorization": f"Bearer {api_key}"},
            timeout=MODEL_LIST_TIMEOUT,
            allow_redirects=False,
        )
    except requests.Timeout:
        raise ModelListRequestError("timeout", models_endpoint) from None
    except requests.RequestException:
        raise ModelListRequestError("network", models_endpoint) from None

    if not 200 <= response.status_code < 300:
        raise ModelListRequestError("http", models_endpoint, response.status_code)

    try:
        payload = response.json()
    except requests.JSONDecodeError:
        raise ModelListRequestError("invalid_response", models_endpoint) from None

    if not isinstance(payload, dict):
        raise ModelListRequestError("invalid_response", models_endpoint)
    return extract_model_ids(payload)


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
