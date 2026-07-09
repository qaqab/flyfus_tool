from __future__ import annotations

import json
import mimetypes
from collections.abc import Generator
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

import requests
from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from openai import OpenAI

from tools._image_utils import (
    build_usage_metadata,
    build_usage_output,
    decode_image,
    extract_model_ids,
    image_model_ids,
    image_model_supports_operation,
    normalize_openai_base_url,
)


MAX_REFERENCE_IMAGES = 16
MAX_OUTPUT_DOWNLOAD_BYTES = 50 * 1024 * 1024
OUTPUT_DOWNLOAD_TIMEOUT = 300


class FlypowerImageGenerateTool(Tool):
    def _invoke(self, tool_parameters: dict) -> Generator[ToolInvokeMessage, None, None]:
        prompt = tool_parameters.get("prompt")
        if not prompt or not isinstance(prompt, str):
            yield self.create_text_message("Error: Prompt is required.")
            return

        model = tool_parameters.get("model", "gpt-image-2")
        supported_models = image_model_ids()
        if model not in supported_models:
            yield self.create_text_message(f"Invalid model. Choose from: {', '.join(sorted(supported_models))}.")
            return

        reference_urls = self._parse_urls(tool_parameters.get("reference_image_urls"))
        mask_url = self._first_url(tool_parameters.get("mask_url"))
        operation = "edit" if reference_urls else "generate"
        if not image_model_supports_operation(model, operation):
            yield self.create_text_message(f"Model {model} does not support {operation} in the image model YAML.")
            return

        client = OpenAI(
            api_key=self.runtime.credentials["api_key"],
            base_url=normalize_openai_base_url(self.runtime.credentials.get("endpoint_url")),
        )

        try:
            available_models = extract_model_ids(client.models.list())
        except Exception as error:
            yield self.create_text_message(f"Failed to list models: {error}")
            return
        if model not in available_models:
            matched_models = sorted(supported_models & available_models)
            if matched_models:
                yield self.create_text_message(
                    f"Model {model} is not available from /models. Available image models: {', '.join(matched_models)}."
                )
            else:
                yield self.create_text_message(
                    f"No supported image model was returned by /models. Expected one of: {', '.join(sorted(supported_models))}."
                )
            return

        try:
            args: dict[str, Any] = {
                "model": model,
                "prompt": prompt,
            }
            error = self._apply_common_parameters(args, tool_parameters, model=model)
            if error:
                yield self.create_text_message(error)
                return

            if reference_urls:
                if len(reference_urls) > MAX_REFERENCE_IMAGES:
                    yield self.create_text_message(f"Error: At most {MAX_REFERENCE_IMAGES} reference image URLs are supported.")
                    return

                args["images"] = [{"image_url": url} for url in reference_urls]

                if mask_url:
                    args["mask"] = {"image_url": mask_url}

                response = self._edit_images_with_url_inputs(client, args)
            else:
                if mask_url:
                    yield self.create_text_message("Error: mask_url requires at least one reference_image_urls value.")
                    return
                response = client.images.generate(**args)
        except Exception as error:
            yield self.create_text_message(f"Failed to {operation} image: {error}")
            return

        usage_metadata = build_usage_metadata(response)
        image_count = 0
        output_format = tool_parameters.get("output_format", "auto")
        for image in getattr(response, "data", []):
            b64_json = getattr(image, "b64_json", None)
            image_url = getattr(image, "url", None)
            if b64_json:
                mime_type, blob_image = decode_image(b64_json)
            elif image_url:
                mime_type, blob_image = self._download_output_image(str(image_url))
            else:
                continue
            if output_format in {"png", "jpeg", "webp"}:
                mime_type = f"image/{output_format}"
            image_count += 1
            yield self.create_blob_message(blob=blob_image, meta={"mime_type": mime_type, **usage_metadata})

        usage_output = build_usage_output(response, model=model, operation=operation, image_count=image_count)
        if usage_output:
            yield self.create_json_message(usage_output)

    @staticmethod
    def _parse_urls(value: object) -> list[str]:
        if value in (None, ""):
            return []

        if isinstance(value, list):
            raw_items = value
        else:
            text = str(value).strip()
            if not text:
                return []
            try:
                parsed = json.loads(text)
            except ValueError:
                parsed = None

            if isinstance(parsed, list):
                raw_items = parsed
            elif isinstance(parsed, str):
                raw_items = [parsed]
            else:
                raw_items = text.replace("\n", ",").split(",")

        urls: list[str] = []
        for item in raw_items:
            url = str(item).strip()
            if url:
                FlypowerImageGenerateTool._validate_http_url(url)
                urls.append(url)
        return urls

    @staticmethod
    def _first_url(value: object) -> str | None:
        urls = FlypowerImageGenerateTool._parse_urls(value)
        return urls[0] if urls else None

    @staticmethod
    def _validate_http_url(url: str) -> None:
        parsed = urlparse(url)
        if url.startswith("data:image/"):
            return
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"Invalid image URL: {url}")

    @staticmethod
    def _edit_images_with_url_inputs(client: OpenAI, args: dict[str, Any]) -> Any:
        base_url = str(client.base_url).rstrip("/")
        response = requests.post(
            f"{base_url}/images/edits",
            headers={
                "Authorization": f"Bearer {client.api_key}",
                "Content-Type": "application/json",
            },
            json=args,
            timeout=300,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"{response.status_code} {response.text[:1000]}")
        return FlypowerImageGenerateTool._to_namespace(response.json())

    @staticmethod
    def _download_output_image(url: str) -> tuple[str, bytes]:
        FlypowerImageGenerateTool._validate_http_url(url)
        if url.startswith("data:image/"):
            return decode_image(url)

        response = requests.get(url, timeout=OUTPUT_DOWNLOAD_TIMEOUT, stream=True)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        chunks: list[bytes] = []
        downloaded = 0
        for chunk in response.iter_content(chunk_size=1024 * 256):
            if not chunk:
                continue
            downloaded += len(chunk)
            if downloaded > MAX_OUTPUT_DOWNLOAD_BYTES:
                raise ValueError(f"Output image is larger than {MAX_OUTPUT_DOWNLOAD_BYTES // 1024 // 1024}MB: {url}")
            chunks.append(chunk)

        if not chunks:
            raise ValueError(f"Output image URL returned an empty body: {url}")

        mime_type = content_type or mimetypes.guess_type(urlparse(url).path)[0] or "image/png"
        return mime_type, b"".join(chunks)

    @staticmethod
    def _to_namespace(value: Any) -> Any:
        if isinstance(value, dict):
            return SimpleNamespace(**{key: FlypowerImageGenerateTool._to_namespace(item) for key, item in value.items()})
        if isinstance(value, list):
            return [FlypowerImageGenerateTool._to_namespace(item) for item in value]
        return value

    @staticmethod
    def _apply_common_parameters(
        args: dict[str, Any],
        tool_parameters: dict,
        *,
        model: str,
    ) -> str | None:
        size = tool_parameters.get("size", "auto")
        if size and size != "auto":
            args["size"] = str(size)

        quality = tool_parameters.get("quality", "auto")
        if quality not in {"auto", "low", "medium", "high"}:
            return "Invalid quality. Choose auto, low, medium, or high."
        if quality != "auto":
            args["quality"] = quality

        output_format = tool_parameters.get("output_format", "auto")
        if output_format not in {"auto", "png", "jpeg", "webp"}:
            return "Invalid output_format. Choose auto, png, jpeg, or webp."
        if output_format != "auto":
            args["output_format"] = output_format

        output_compression = tool_parameters.get("output_compression")
        if output_compression not in (None, ""):
            try:
                output_compression_value = int(output_compression)
            except (TypeError, ValueError):
                return "Invalid output_compression. Choose an integer between 0 and 100."
            if not 0 <= output_compression_value <= 100:
                return "Invalid output_compression. Choose an integer between 0 and 100."
            if output_format in {"jpeg", "webp"}:
                args["output_compression"] = output_compression_value

        background = tool_parameters.get("background", "auto")
        if background not in {"auto", "opaque", "transparent"}:
            return "Invalid background. Choose auto, opaque, or transparent."
        if background == "transparent" and model == "gpt-image-2":
            return "Invalid background. gpt-image-2 does not support transparent background."
        if background != "auto":
            args["background"] = background

        moderation = tool_parameters.get("moderation", "auto")
        if moderation not in {"auto", "low"}:
            return "Invalid moderation. Choose auto or low."
        if moderation != "auto":
            args["moderation"] = moderation

        n = tool_parameters.get("n", 1)
        try:
            n_value = int(n)
        except (TypeError, ValueError):
            return "Invalid n value. Must be a number between 1 and 10."
        if not 1 <= n_value <= 10:
            return "Invalid n value. Must be between 1 and 10."
        args["n"] = n_value

        return None
