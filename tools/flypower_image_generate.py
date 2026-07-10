from __future__ import annotations

import io
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
MAX_INPUT_DOWNLOAD_BYTES = 50 * 1024 * 1024
MAX_OUTPUT_DOWNLOAD_BYTES = 50 * 1024 * 1024
INPUT_DOWNLOAD_TIMEOUT = 300
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
        reference_files = self._extract_files(tool_parameters.get("reference_image_files"))
        mask_url = self._first_url(tool_parameters.get("mask_url"))
        mask_file = self._first_file(tool_parameters.get("mask_file"))
        operation = "edit" if reference_urls or reference_files else "generate"
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

            if reference_urls or reference_files:
                reference_count = len(reference_urls) + len(reference_files)
                if reference_count > MAX_REFERENCE_IMAGES:
                    yield self.create_text_message(f"Error: At most {MAX_REFERENCE_IMAGES} reference images are supported.")
                    return
                if mask_url and mask_file:
                    yield self.create_text_message("Error: Use either mask_url or mask_file, not both.")
                    return
                response = self._edit_images_with_files(client, args, reference_urls, reference_files, mask_url, mask_file)
            else:
                if mask_url or mask_file:
                    yield self.create_text_message("Error: mask requires at least one reference image URL or file.")
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
    def _extract_files(value: object) -> list[Any]:
        if value in (None, ""):
            return []
        if isinstance(value, list):
            return [item for item in value if item]
        return [value]

    @staticmethod
    def _first_file(value: object) -> Any | None:
        files = FlypowerImageGenerateTool._extract_files(value)
        return files[0] if files else None

    @staticmethod
    def _validate_http_url(url: str) -> None:
        parsed = urlparse(url)
        if url.startswith("data:image/"):
            return
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"Invalid image URL: {url}")

    @staticmethod
    def _edit_images_with_files(
        client: OpenAI,
        args: dict[str, Any],
        reference_urls: list[str],
        reference_files: list[Any],
        mask_url: str | None,
        mask_file_param: Any | None,
    ) -> Any:
        image_files: list[io.BytesIO] = []
        mask_file: io.BytesIO | None = None
        try:
            for index, url in enumerate(reference_urls, start=1):
                image_files.append(FlypowerImageGenerateTool._download_input_image(url, default_name=f"reference_image_{index}"))
            for index, file_obj in enumerate(reference_files, start=len(image_files) + 1):
                image_files.append(FlypowerImageGenerateTool._uploaded_file_to_image(file_obj, default_name=f"reference_image_{index}"))

            multipart_args = dict(args)
            multipart_args["image"] = image_files[0] if len(image_files) == 1 else image_files

            if mask_url:
                mask_file = FlypowerImageGenerateTool._download_input_image(mask_url, default_name="mask_image")
                multipart_args["mask"] = mask_file
            elif mask_file_param:
                mask_file = FlypowerImageGenerateTool._uploaded_file_to_image(mask_file_param, default_name="mask_image")
                multipart_args["mask"] = mask_file

            return client.images.edit(**multipart_args)
        finally:
            for image_file in image_files:
                image_file.close()
            if mask_file:
                mask_file.close()

    @staticmethod
    def _download_input_image(url: str, *, default_name: str) -> io.BytesIO:
        FlypowerImageGenerateTool._validate_http_url(url)
        if url.startswith("data:image/"):
            mime_type, image_data = decode_image(url)
            image_file = io.BytesIO(image_data)
            image_file.name = f"{default_name}{FlypowerImageGenerateTool._extension_for_mime_type(mime_type)}"
            return image_file

        response = requests.get(url, timeout=INPUT_DOWNLOAD_TIMEOUT, stream=True)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type and not content_type.startswith("image/"):
            raise ValueError(f"URL is not an image: {url}")

        chunks: list[bytes] = []
        downloaded = 0
        for chunk in response.iter_content(chunk_size=1024 * 256):
            if not chunk:
                continue
            downloaded += len(chunk)
            if downloaded > MAX_INPUT_DOWNLOAD_BYTES:
                raise ValueError(f"Input image is larger than {MAX_INPUT_DOWNLOAD_BYTES // 1024 // 1024}MB: {url}")
            chunks.append(chunk)

        if not chunks:
            raise ValueError(f"Input image URL returned an empty body: {url}")

        image_file = io.BytesIO(b"".join(chunks))
        image_file.name = f"{default_name}{FlypowerImageGenerateTool._guess_extension(url, content_type)}"
        return image_file

    @staticmethod
    def _uploaded_file_to_image(file_obj: Any, *, default_name: str) -> io.BytesIO:
        mime_type = FlypowerImageGenerateTool._file_value(file_obj, "mime_type")
        if mime_type and not str(mime_type).startswith("image/"):
            raise ValueError(f"Uploaded file is not an image: {FlypowerImageGenerateTool._file_value(file_obj, 'filename') or default_name}")

        blob = FlypowerImageGenerateTool._file_value(file_obj, "blob")
        if blob is None and hasattr(file_obj, "read"):
            blob = file_obj.read()
        if blob is None and isinstance(file_obj, dict) and file_obj.get("url"):
            return FlypowerImageGenerateTool._download_input_image(str(file_obj["url"]), default_name=default_name)
        if not isinstance(blob, bytes):
            raise ValueError(f"Uploaded file has no readable image content: {default_name}")
        if len(blob) > MAX_INPUT_DOWNLOAD_BYTES:
            raise ValueError(f"Uploaded image is larger than {MAX_INPUT_DOWNLOAD_BYTES // 1024 // 1024}MB: {default_name}")

        filename = FlypowerImageGenerateTool._file_value(file_obj, "filename")
        image_file = io.BytesIO(blob)
        image_file.name = FlypowerImageGenerateTool._upload_filename(str(filename or ""), str(mime_type or ""), default_name)
        return image_file

    @staticmethod
    def _file_value(file_obj: Any, key: str) -> Any:
        if isinstance(file_obj, dict):
            return file_obj.get(key)
        return getattr(file_obj, key, None)

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
    def _guess_extension(url: str, content_type: str) -> str:
        if content_type:
            extension = FlypowerImageGenerateTool._extension_for_mime_type(content_type)
            if extension:
                return extension

        guessed_type, _ = mimetypes.guess_type(urlparse(url).path)
        return FlypowerImageGenerateTool._extension_for_mime_type(guessed_type or "") or ".png"

    @staticmethod
    def _upload_filename(filename: str, mime_type: str, default_name: str) -> str:
        cleaned = filename.strip().split("/")[-1]
        if "." in cleaned:
            return cleaned

        guessed_type, _ = mimetypes.guess_type(cleaned)
        extension = FlypowerImageGenerateTool._extension_for_mime_type(mime_type or guessed_type or "")
        return f"{cleaned or default_name}{extension}"

    @staticmethod
    def _extension_for_mime_type(mime_type: str) -> str:
        if mime_type == "image/jpeg":
            return ".jpg"
        return mimetypes.guess_extension(mime_type) or ".png"

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
