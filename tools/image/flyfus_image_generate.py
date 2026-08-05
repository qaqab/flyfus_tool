from __future__ import annotations

import base64
import hashlib
import io
import json
import mimetypes
import time
import uuid
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

import requests
from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from openai import OpenAI

from tools.image._image_utils import (
    ModelListRequestError,
    build_usage_metadata,
    decode_image,
    fetch_openai_model_ids,
    image_model_ids,
    image_model_supports_operation,
    normalize_openai_base_url,
)
from tools._sls_logging import write_tool_log


MAX_REFERENCE_IMAGES = 16
MAX_GEMINI_REFERENCE_IMAGES = 2
MAX_INPUT_DOWNLOAD_BYTES = 50 * 1024 * 1024
INPUT_DOWNLOAD_TIMEOUT = 300
GEMINI_REQUEST_TIMEOUT = (10.0, 300.0)
OSS_UPLOAD_TIMEOUT = (10.0, 120.0)
MAX_OSS_UPLOAD_WORKERS = 4
MAX_IMAGE_REQUEST_RETRIES = 3
MAX_OSS_UPLOAD_RETRIES = 2
SENSITIVE_RESPONSE_HEADERS = frozenset(
    {"authorization", "cookie", "proxy-authorization", "set-cookie", "x-api-key"}
)
OPENAI_MODELS = frozenset({"gpt-image-2", "gpt-image-2-4k"})
OPENAI_4K_SIZES = frozenset(
    {
        "1024x1024",
        "1536x1024",
        "1024x1536",
        "2048x2048",
        "2048x1152",
        "3840x2160",
        "2160x3840",
    }
)
GEMINI_IMAGE_SIZES = frozenset({"512", "1K", "2K"})
GEMINI_ASPECT_RATIOS = frozenset(
    {
        "1:1",
        "1:4",
        "1:8",
        "3:2",
        "2:3",
        "3:4",
        "4:1",
        "4:3",
        "4:5",
        "5:4",
        "8:1",
        "16:9",
        "21:9",
    }
)


class FlyfusImageGenerateTool(Tool):
    def _invoke(
        self, tool_parameters: dict
    ) -> Generator[ToolInvokeMessage, None, None]:
        log_id = str(uuid.uuid4())
        started_at = time.monotonic()
        prompt = tool_parameters.get("prompt")
        requested_model = str(tool_parameters.get("model", "gpt-image-2"))
        request_context: dict[str, object] = {}
        self._write_invocation_log(
            log_id,
            "started",
            model=requested_model,
            prompt_characters=len(prompt) if isinstance(prompt, str) else 0,
            has_reference_input=bool(tool_parameters.get("reference_image_urls")),
            user_id=self.runtime.user_id or "-",
            session_id=self.runtime.session_id or "-",
        )

        def fail(error: str, stage: str) -> Generator[ToolInvokeMessage, None, None]:
            self._write_invocation_log(
                log_id,
                "failed",
                model=requested_model,
                stage=stage,
                elapsed_ms=self._elapsed_ms(started_at),
                **request_context,
            )
            yield from self._error_messages(
                error, log_id, request_context.get("request_fingerprint")
            )

        if not prompt or not isinstance(prompt, str):
            yield from fail("Error: Prompt is required.", "validation")
            return

        model = requested_model
        supported_models = image_model_ids()
        if model not in supported_models:
            yield from fail(
                f"Invalid model. Choose from: {', '.join(sorted(supported_models))}.",
                "validation",
            )
            return

        try:
            reference_urls = self._parse_urls(
                tool_parameters.get("reference_image_urls")
            )
        except ValueError as error:
            yield from fail(str(error), "validation")
            return
        operation = "edit" if reference_urls else "generate"
        request_context.update(
            {
                "request_fingerprint": self._request_fingerprint(
                    model, prompt, reference_urls, tool_parameters
                ),
                "reference_count": len(reference_urls),
                "reference_hosts": ",".join(
                    sorted({urlparse(url).netloc for url in reference_urls})
                )
                or "-",
            }
        )
        self._write_invocation_log(
            log_id, "validated", model=model, operation=operation, **request_context
        )
        if not image_model_supports_operation(model, operation):
            yield from fail(
                f"Model {model} does not support {operation} in the image model YAML.",
                "validation",
            )
            return

        api_key = str(self.runtime.credentials.get("api_key") or "")
        if not api_key:
            yield from fail("API key is required for image generation.", "credentials")
            return
        try:
            normalized_base_url = normalize_openai_base_url(
                self.runtime.credentials.get("endpoint_url")
            )
            if normalized_base_url is None:
                yield from fail("API endpoint is missing.", "credentials")
                return
            available_models = fetch_openai_model_ids(normalized_base_url, api_key)
        except ValueError as error:
            yield from fail(f"Invalid API endpoint: {error}", "model_validation")
            return
        except ModelListRequestError as error:
            yield from fail(
                f"Failed to validate API access: {error}", "model_validation"
            )
            return
        if model not in available_models:
            matched_models = sorted(supported_models & available_models)
            if matched_models:
                yield from fail(
                    f"Model {model} is not available from /models. Available image models: {', '.join(matched_models)}.",
                    "model_validation",
                )
            else:
                yield from fail(
                    f"No supported image model was returned by /models. Expected one of: {', '.join(sorted(supported_models))}.",
                    "model_validation",
                )
            return

        try:

            def on_download_event(event, **fields) -> None:
                self._write_invocation_log(
                    log_id,
                    f"reference_download_{event}",
                    model=model,
                    operation=operation,
                    **request_context,
                    **fields,
                )

            if model in OPENAI_MODELS:
                args, parameter_error = self._build_openai_args(
                    model, prompt, tool_parameters
                )
                if parameter_error:
                    yield from fail(parameter_error, "parameter_validation")
                    return
                if len(reference_urls) > MAX_REFERENCE_IMAGES:
                    yield from fail(
                        f"Error: At most {MAX_REFERENCE_IMAGES} reference images are supported.",
                        "parameter_validation",
                    )
                    return

                client = OpenAI(
                    api_key=api_key, base_url=normalized_base_url, max_retries=0
                )
                if reference_urls:
                    upstream_endpoint = f"{normalized_base_url}/images/edits"
                    upstream_request_body = {
                        **args,
                        "reference_image_urls": reference_urls,
                    }

                    def request():
                        return self._edit_images_with_files(
                            client,
                            args,
                            reference_urls,
                            on_download_event=on_download_event,
                        )
                else:
                    upstream_endpoint = f"{normalized_base_url}/images/generations"
                    upstream_request_body = args

                    def request():
                        return client.images.generate(**args)
            else:
                if len(reference_urls) > MAX_GEMINI_REFERENCE_IMAGES:
                    yield from fail(
                        f"Error: Gemini supports at most {MAX_GEMINI_REFERENCE_IMAGES} reference images.",
                        "parameter_validation",
                    )
                    return
                payload, parameter_error = self._build_gemini_payload(
                    prompt,
                    reference_urls,
                    tool_parameters,
                    on_download_event=on_download_event,
                )
                if parameter_error:
                    yield from fail(parameter_error, "parameter_validation")
                    return
                upstream_endpoint = self._gemini_endpoint(normalized_base_url, model)
                upstream_request_body = payload

                def request():
                    return self._request_gemini(upstream_endpoint, api_key, payload)

            self._write_invocation_log(
                log_id,
                "request_started",
                model=model,
                operation=operation,
                parameters=self._request_parameter_summary(upstream_request_body),
                upstream_endpoint=upstream_endpoint,
                upstream_request_body=self._bounded_log_json(upstream_request_body),
                **request_context,
            )
            response = self._run_image_request_with_retry(
                lambda: self._require_image_data(request()),
                on_attempt_started=lambda attempt: self._write_invocation_log(
                    log_id,
                    "request_attempt_started",
                    model=model,
                    operation=operation,
                    attempt=attempt,
                    **request_context,
                ),
                on_attempt_finished=lambda attempt, elapsed_ms, error=None: (
                    self._write_invocation_log(
                        log_id,
                        "request_attempt_failed"
                        if error
                        else "request_attempt_succeeded",
                        model=model,
                        operation=operation,
                        attempt=attempt,
                        elapsed_ms=elapsed_ms,
                        **self._error_log_fields(error),
                        **request_context,
                    )
                ),
                on_retry=lambda attempt, error: self._write_invocation_log(
                    log_id,
                    "request_retry",
                    model=model,
                    operation=operation,
                    attempt=attempt,
                    max_retries=MAX_IMAGE_REQUEST_RETRIES,
                    **self._error_log_fields(error),
                    **request_context,
                ),
            )
        except Exception as error:
            self._write_invocation_log(
                log_id,
                "request_failed",
                model=model,
                operation=operation,
                elapsed_ms=self._elapsed_ms(started_at),
                **self._error_log_fields(error),
                **request_context,
            )
            yield from fail(f"Failed to {operation} image: {error}", "image_request")
            return
        self._write_invocation_log(
            log_id,
            "request_succeeded",
            model=model,
            operation=operation,
            elapsed_ms=self._elapsed_ms(started_at),
            image_count=len(getattr(response, "data", []) or []),
            upstream_request_id=getattr(response, "_request_id", None) or "-",
            **self._response_log_fields(response),
            **request_context,
        )

        uploads: list[tuple[bytes, str, str]] = []
        try:
            for index, image in enumerate(getattr(response, "data", []) or [], start=1):
                b64_json = getattr(image, "b64_json", None)
                if b64_json:
                    mime_type, blob_image = decode_image(b64_json)
                    uploads.append(
                        (blob_image, mime_type, self._output_filename(index, mime_type))
                    )
        except Exception as error:
            yield from fail(
                f"Failed to process generated images: {error}", "response_processing"
            )
            return

        if not uploads:
            self._write_invocation_log(
                log_id,
                "response_empty",
                model=model,
                operation=operation,
                elapsed_ms=self._elapsed_ms(started_at),
                upstream_request_id=getattr(response, "_request_id", None) or "-",
                **self._response_log_fields(response),
                **request_context,
            )
            yield from fail(
                "The image model did not return any images.", "response_processing"
            )
            return

        try:
            self._write_invocation_log(
                log_id,
                "oss_upload_started",
                model=model,
                operation=operation,
                image_count=len(uploads),
                **request_context,
            )
            with ThreadPoolExecutor(
                max_workers=min(MAX_OSS_UPLOAD_WORKERS, len(uploads))
            ) as executor:
                upload_to_oss = partial(self._upload_output_to_oss, log_id=log_id)
                oss_urls = list(executor.map(upload_to_oss, uploads))
        except Exception as error:
            yield from fail(
                f"Failed to upload generated images to OSS (log_id={log_id}): {error}",
                "oss_upload",
            )
            return

        usage_metadata = build_usage_metadata(response)
        self._write_invocation_log(
            log_id,
            "succeeded",
            model=model,
            operation=operation,
            elapsed_ms=self._elapsed_ms(started_at),
            image_count=len(oss_urls),
            **request_context,
        )
        yield self.create_json_message(
            {
                "urls": oss_urls,
                "log": self._log_reference(
                    log_id, request_context.get("request_fingerprint")
                ),
                **usage_metadata,
            }
        )
        yield self.create_text_message(json.dumps(oss_urls, ensure_ascii=False))

    @staticmethod
    def _build_openai_args(
        model: str, prompt: str, tool_parameters: dict
    ) -> tuple[dict[str, Any], str | None]:
        args: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "output_format": "png",
            "response_format": "b64_json",
        }
        if model == "gpt-image-2-4k":
            size = str(tool_parameters.get("openai_4k_size") or "3840x2160")
            if size not in OPENAI_4K_SIZES:
                return (
                    {},
                    f"Invalid GPT Image 2 4K size. Choose from: {', '.join(sorted(OPENAI_4K_SIZES))}.",
                )
            args["size"] = size
        return args, None

    @staticmethod
    def _build_gemini_payload(
        prompt: str,
        reference_urls: list[str],
        tool_parameters: dict,
        *,
        on_download_event=None,
    ) -> tuple[dict[str, Any], str | None]:
        image_size = str(tool_parameters.get("gemini_image_size") or "1K")
        aspect_ratio = str(tool_parameters.get("gemini_aspect_ratio") or "1:1")
        if image_size not in GEMINI_IMAGE_SIZES:
            return (
                {},
                f"Invalid Gemini image size. Choose from: {', '.join(sorted(GEMINI_IMAGE_SIZES))}.",
            )
        if aspect_ratio not in GEMINI_ASPECT_RATIOS:
            return (
                {},
                f"Invalid Gemini aspect ratio. Choose from: {', '.join(sorted(GEMINI_ASPECT_RATIOS))}.",
            )

        parts: list[dict[str, Any]] = [{"text": prompt}]
        for index, url in enumerate(reference_urls, start=1):
            direct_part = FlyfusImageGenerateTool._gemini_url_part(url)
            if direct_part:
                parts.append(direct_part)
                continue

            image_file = FlyfusImageGenerateTool._download_input_image(
                url,
                default_name=f"reference_image_{index}",
                on_event=on_download_event,
                input_kind="reference",
                index=index,
            )
            try:
                mime_type = mimetypes.guess_type(image_file.name)[0] or "image/png"
                if mime_type not in {"image/png", "image/jpeg", "image/webp"}:
                    return (
                        {},
                        f"Gemini reference image {index} must be PNG, JPEG, or WebP.",
                    )
                parts.append(
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": base64.b64encode(image_file.getvalue()).decode(
                                "ascii"
                            ),
                        }
                    }
                )
            finally:
                image_file.close()

        return {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "imageConfig": {"imageSize": image_size, "aspectRatio": aspect_ratio},
                "candidateCount": 1,
            },
        }, None

    @staticmethod
    def _gemini_url_part(url: str) -> dict[str, Any] | None:
        parsed = urlparse(url)
        if parsed.scheme != "https":
            return None
        mime_type = mimetypes.guess_type(parsed.path)[0]
        if mime_type not in {"image/png", "image/jpeg"}:
            return None
        return {"fileData": {"mimeType": mime_type, "fileUri": url}}

    @staticmethod
    def _gemini_endpoint(normalized_openai_base_url: str, model: str) -> str:
        base_url = normalized_openai_base_url.removesuffix("/v1")
        return f"{base_url}/v1beta/models/{model}:generateContent"

    @staticmethod
    def _request_gemini(
        endpoint: str, api_key: str, payload: dict[str, Any]
    ) -> SimpleNamespace:
        response = requests.post(
            endpoint,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=GEMINI_REQUEST_TIMEOUT,
            allow_redirects=False,
        )
        response.raise_for_status()
        try:
            response_payload = response.json()
        except requests.JSONDecodeError:
            raise ValueError("Gemini returned invalid JSON.") from None

        images: list[SimpleNamespace] = []
        for candidate in (
            response_payload.get("candidates", [])
            if isinstance(response_payload, dict)
            else []
        ):
            content = candidate.get("content") if isinstance(candidate, dict) else None
            for part in content.get("parts", []) if isinstance(content, dict) else []:
                inline_data = part.get("inlineData") if isinstance(part, dict) else None
                if not isinstance(inline_data, dict) or not inline_data.get("data"):
                    continue
                mime_type = str(inline_data.get("mimeType") or "image/png")
                images.append(
                    SimpleNamespace(
                        b64_json=f"data:{mime_type};base64,{inline_data['data']}"
                    )
                )

        request_id = response.headers.get("x-request-id") or response.headers.get(
            "request-id"
        )
        return SimpleNamespace(data=images, _request_id=request_id, _response=response)

    def _error_messages(
        self, error: str, log_id: str, request_fingerprint: object | None
    ) -> Generator[ToolInvokeMessage, None, None]:
        yield self.create_json_message(
            {
                "urls": [],
                "error": error,
                "log": self._log_reference(log_id, request_fingerprint),
            }
        )
        yield self.create_text_message("[]")

    @staticmethod
    def _log_reference(
        log_id: str, request_fingerprint: object | None
    ) -> dict[str, str]:
        reference = {"log_id": log_id}
        if request_fingerprint:
            reference["request_fingerprint"] = str(request_fingerprint)
        return reference

    @staticmethod
    def _run_image_request_with_retry(
        request, on_retry=None, on_attempt_started=None, on_attempt_finished=None
    ):
        for attempt in range(MAX_IMAGE_REQUEST_RETRIES + 1):
            attempt_number = attempt + 1
            started_at = time.monotonic()
            if on_attempt_started:
                on_attempt_started(attempt_number)
            try:
                response = request()
            except Exception as error:
                if on_attempt_finished:
                    on_attempt_finished(
                        attempt_number,
                        FlyfusImageGenerateTool._elapsed_ms(started_at),
                        error,
                    )
                if (
                    attempt >= MAX_IMAGE_REQUEST_RETRIES
                    or not FlyfusImageGenerateTool._is_retryable_image_error(error)
                ):
                    raise
                if on_retry:
                    on_retry(attempt_number, error)
                time.sleep(0.5 * (attempt + 1))
            else:
                if on_attempt_finished:
                    on_attempt_finished(
                        attempt_number, FlyfusImageGenerateTool._elapsed_ms(started_at)
                    )
                return response

        raise RuntimeError("Image request retry loop exited unexpectedly.")

    @staticmethod
    def _is_retryable_image_error(error: Exception) -> bool:
        message = str(error).lower()
        return (
            "json_invalid" in message
            or (
                "invalid json" in message
                and ("<!doctype html" in message or "expected value" in message)
            )
            or "gemini returned invalid json" in message
            or any(
                marker in message
                for marker in (
                    "image response did not include data",
                    "upstream request failed",
                    "timeout",
                    "connection",
                    "rate limit",
                    "429",
                    "502",
                    "503",
                    "504",
                )
            )
        )

    @staticmethod
    def _require_image_data(response: object) -> object:
        if getattr(response, "data", None) is None:
            error = RuntimeError("Image response did not include data.")
            error.response = getattr(response, "_response", None)
            error.request_id = getattr(response, "_request_id", None)
            raise error
        return response

    @staticmethod
    def _request_fingerprint(
        model: str, prompt: str, reference_urls: list[str], tool_parameters: dict
    ) -> str:
        # Groups externally retried calls without placing prompt text or URLs in SLS.
        payload = {
            "model": model,
            "prompt": prompt,
            "reference_urls": reference_urls,
            "parameters": {
                key: value
                for key, value in tool_parameters.items()
                if key not in {"prompt", "reference_image_urls"}
            },
        }
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _request_parameter_summary(args: dict[str, Any]) -> str:
        return FlyfusImageGenerateTool._bounded_log_json(
            {key: value for key, value in args.items() if key != "prompt"}
        )

    @staticmethod
    def _bounded_log_json(value: object) -> str:
        """Serialize reproducibility data while keeping a single SLS field bounded."""
        encoded = json.dumps(
            FlyfusImageGenerateTool._redact_large_image_data(value),
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        if len(encoded) <= 32_000:
            return encoded
        return json.dumps(
            {
                "truncated": True,
                "original_characters": len(encoded),
                "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
                "preview": encoded[:31_000],
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    @staticmethod
    def _redact_large_image_data(value: object) -> object:
        if isinstance(value, SimpleNamespace):
            return FlyfusImageGenerateTool._redact_large_image_data(vars(value))
        if isinstance(value, dict):
            return {
                key: FlyfusImageGenerateTool._image_data_log_value(item)
                if key == "b64_json" or (key == "data" and "mimeType" in value)
                else FlyfusImageGenerateTool._redact_large_image_data(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [
                FlyfusImageGenerateTool._redact_large_image_data(item) for item in value
            ]
        return value

    @staticmethod
    def _image_data_log_value(value: object) -> object:
        if not isinstance(value, str):
            return value
        return {
            "omitted": True,
            "characters": len(value),
            "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
        }

    @staticmethod
    def _error_log_fields(error: Exception | None) -> dict[str, object]:
        if error is None:
            return {}
        response = getattr(error, "response", None)
        status_code = getattr(error, "status_code", None) or getattr(
            response, "status_code", None
        )
        fields: dict[str, object] = {
            "exception_type": type(error).__name__,
            "exception_message": str(error)[:2_000],
        }
        if status_code is not None:
            fields["status_code"] = status_code
        request_id = getattr(error, "request_id", None) or getattr(
            response, "_request_id", None
        )
        if request_id:
            fields["upstream_request_id"] = request_id
        fields.update(
            FlyfusImageGenerateTool._response_header_log_fields(
                getattr(response, "headers", None)
            )
        )
        return fields

    @staticmethod
    def _response_header_log_fields(headers: object) -> dict[str, object]:
        try:
            normalized_headers = {
                str(key).lower(): str(value) for key, value in headers.items()
            }
        except AttributeError:
            return {}
        if not normalized_headers:
            return {}

        fields: dict[str, object] = {
            "upstream_response_headers": FlyfusImageGenerateTool._bounded_log_json(
                {
                    key: "[REDACTED]" if key in SENSITIVE_RESPONSE_HEADERS else value
                    for key, value in normalized_headers.items()
                }
            )
        }
        request_id = normalized_headers.get("x-request-id") or normalized_headers.get(
            "request-id"
        )
        if request_id:
            fields["upstream_header_request_id"] = request_id
        return fields

    @staticmethod
    def _response_log_fields(response: object) -> dict[str, object]:
        """Return structural response diagnostics without logging image content."""
        data = getattr(response, "data", None)
        fields: dict[str, object] = {
            "response_type": type(response).__name__,
            "response_data_type": type(data).__name__,
        }
        try:
            fields["response_data_count"] = len(data)
        except TypeError:
            pass

        model_dump = getattr(response, "model_dump", None)
        if callable(model_dump):
            try:
                dumped = model_dump()
            except (TypeError, ValueError):
                dumped = None
            if isinstance(dumped, dict):
                fields["response_fields"] = ",".join(sorted(dumped)) or "-"
                fields["upstream_response_body"] = (
                    FlyfusImageGenerateTool._bounded_log_json(dumped)
                )
                fields["upstream_response_body_source"] = "sdk_parsed"
        elif hasattr(response, "__dict__"):
            dumped = {
                key: value
                for key, value in vars(response).items()
                if key != "_response"
            }
            fields["response_fields"] = ",".join(sorted(dumped)) or "-"
            fields["upstream_response_body"] = (
                FlyfusImageGenerateTool._bounded_log_json(dumped)
            )
            fields["upstream_response_body_source"] = "sdk_parsed"

        if isinstance(data, (list, tuple)) and data:
            first_item = data[0]
            item_dump = getattr(first_item, "model_dump", None)
            if callable(item_dump):
                try:
                    dumped_item = item_dump()
                except (TypeError, ValueError):
                    dumped_item = None
                if isinstance(dumped_item, dict):
                    fields["response_first_image_fields"] = (
                        ",".join(sorted(dumped_item)) or "-"
                    )
            elif hasattr(first_item, "__dict__"):
                fields["response_first_image_fields"] = (
                    ",".join(sorted(vars(first_item))) or "-"
                )

        http_response = getattr(response, "_response", None)
        status_code = getattr(http_response, "status_code", None)
        if status_code is not None:
            fields["upstream_status_code"] = status_code
        fields.update(
            FlyfusImageGenerateTool._response_header_log_fields(
                getattr(http_response, "headers", None)
            )
        )
        return fields

    @staticmethod
    def _output_filename(index: int, mime_type: str) -> str:
        return f"generated_image_{index}{FlyfusImageGenerateTool._extension_for_mime_type(mime_type)}"

    def _upload_output_to_oss(
        self, upload: tuple[bytes, str, str], *, log_id: str
    ) -> str:
        payload, mime_type, filename = upload
        payload_size = len(payload)
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        oss_api_base_url = (
            str(self.runtime.credentials.get("oss_api_base_url") or "")
            .strip()
            .rstrip("/")
        )
        oss_api_token = str(self.runtime.credentials.get("oss_api_token") or "")
        if not oss_api_base_url or not oss_api_token:
            raise RuntimeError("OSS API base URL and token are required.")

        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {oss_api_token}",
        }
        endpoint = f"{oss_api_base_url}/v1/oss-assets/image-file/upload"
        response = None
        started_at = time.monotonic()
        for attempt in range(MAX_OSS_UPLOAD_RETRIES + 1):
            try:
                response = requests.post(
                    endpoint,
                    headers=headers,
                    files={
                        "file": (filename, payload, mime_type),
                        "filename": (None, filename),
                    },
                    timeout=OSS_UPLOAD_TIMEOUT,
                    allow_redirects=False,
                )
            except requests.RequestException as error:
                if attempt < MAX_OSS_UPLOAD_RETRIES:
                    self._write_oss_log(
                        log_id,
                        "retry",
                        filename=filename,
                        attempt=attempt + 1,
                        max_retries=MAX_OSS_UPLOAD_RETRIES,
                        exception_type=type(error).__name__,
                    )
                    time.sleep(0.5 * (attempt + 1))
                    continue
                self._write_oss_log(
                    log_id,
                    "request_failed",
                    filename=filename,
                    mime_type=mime_type or "-",
                    payload_size=payload_size,
                    payload_sha256=payload_sha256,
                    elapsed_ms=self._elapsed_ms(started_at),
                    exception_type=type(error).__name__,
                )
                raise RuntimeError(
                    f"OSS upload request failed (log_id={log_id}): {type(error).__name__}"
                ) from error

            if response.status_code < 500 and response.status_code != 429:
                break
            if attempt < MAX_OSS_UPLOAD_RETRIES:
                self._write_oss_log(
                    log_id,
                    "retry",
                    filename=filename,
                    attempt=attempt + 1,
                    max_retries=MAX_OSS_UPLOAD_RETRIES,
                    status_code=response.status_code,
                )
                time.sleep(0.5 * (attempt + 1))
                continue
            break

        if response is None:
            raise RuntimeError(
                f"OSS upload did not return a response (log_id={log_id})"
            )

        elapsed_ms = self._elapsed_ms(started_at)
        request_id = (
            response.headers.get("x-fc-request-id")
            or response.headers.get("x-request-id")
            or ""
        )

        if not 200 <= response.status_code < 300:
            response_text = str(getattr(response, "text", "")).strip()
            self._write_oss_log(
                log_id,
                "failed",
                filename=filename,
                mime_type=mime_type or "-",
                payload_size=payload_size,
                payload_sha256=payload_sha256,
                status_code=response.status_code,
                elapsed_ms=elapsed_ms,
                request_id=request_id or "-",
            )
            detail = f": {response_text[:500]}" if response_text else ""
            request_id_detail = f" (request_id={request_id})" if request_id else ""
            raise RuntimeError(
                f"OSS upload returned HTTP {response.status_code}{request_id_detail} (log_id={log_id}){detail}"
            )

        try:
            response_body = response.json()
            public_url = response_body["data"]["public_url"]
        except (KeyError, TypeError, ValueError, requests.JSONDecodeError):
            raise RuntimeError("OSS upload returned an invalid response") from None
        if not isinstance(public_url, str) or not public_url:
            raise RuntimeError("OSS upload returned an invalid public URL")
        self._write_oss_log(
            log_id,
            "succeeded",
            endpoint=endpoint,
            filename=filename,
            mime_type=mime_type or "-",
            payload_size=payload_size,
            payload_sha256=payload_sha256,
            response_body=self._bounded_log_json(response_body),
            public_url=public_url,
            status_code=response.status_code,
            elapsed_ms=elapsed_ms,
            request_id=request_id or "-",
        )
        return public_url

    def _write_oss_log(self, log_id: str, event: str, **fields: object) -> None:
        write_tool_log(
            self.runtime.credentials, log_id, f"oss_upload_{event}", **fields
        )

    def _write_invocation_log(self, log_id: str, event: str, **fields: object) -> None:
        write_tool_log(self.runtime.credentials, log_id, f"image_{event}", **fields)

    @staticmethod
    def _elapsed_ms(started_at: float) -> int:
        return round((time.monotonic() - started_at) * 1000)

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
                FlyfusImageGenerateTool._validate_http_url(url)
                urls.append(url)
        return urls

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
        on_download_event=None,
    ) -> Any:
        image_files: list[io.BytesIO] = []
        try:
            for index, url in enumerate(reference_urls, start=1):
                image_files.append(
                    FlyfusImageGenerateTool._download_input_image(
                        url,
                        default_name=f"reference_image_{index}",
                        on_event=on_download_event,
                        input_kind="reference",
                        index=index,
                    )
                )

            multipart_args = dict(args)
            multipart_args["image"] = (
                image_files[0] if len(image_files) == 1 else image_files
            )
            return client.images.edit(**multipart_args)
        finally:
            for image_file in image_files:
                image_file.close()

    @staticmethod
    def _download_input_image(
        url: str, *, default_name: str, on_event=None, input_kind: str, index: int
    ) -> io.BytesIO:
        FlyfusImageGenerateTool._validate_http_url(url)
        started_at = time.monotonic()
        if on_event:
            on_event("started", input_kind=input_kind, index=index, source_url=url)
        if url.startswith("data:image/"):
            mime_type, image_data = decode_image(url)
            image_file = io.BytesIO(image_data)
            image_file.name = f"{default_name}{FlyfusImageGenerateTool._extension_for_mime_type(mime_type)}"
            if on_event:
                on_event(
                    "succeeded",
                    input_kind=input_kind,
                    index=index,
                    source_url=url,
                    elapsed_ms=FlyfusImageGenerateTool._elapsed_ms(started_at),
                    payload_size=len(image_data),
                    content_type=mime_type,
                    payload_sha256=hashlib.sha256(image_data).hexdigest(),
                )
            return image_file

        try:
            response = requests.get(url, timeout=INPUT_DOWNLOAD_TIMEOUT, stream=True)
            response.raise_for_status()

            content_type = (
                response.headers.get("content-type", "")
                .split(";", 1)[0]
                .strip()
                .lower()
            )
            if content_type and not content_type.startswith("image/"):
                raise ValueError(f"URL is not an image: {url}")

            chunks: list[bytes] = []
            downloaded = 0
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if not chunk:
                    continue
                downloaded += len(chunk)
                if downloaded > MAX_INPUT_DOWNLOAD_BYTES:
                    raise ValueError(
                        f"Input image is larger than {MAX_INPUT_DOWNLOAD_BYTES // 1024 // 1024}MB: {url}"
                    )
                chunks.append(chunk)

            if not chunks:
                raise ValueError(f"Input image URL returned an empty body: {url}")

            image_file = io.BytesIO(b"".join(chunks))
            image_file.name = f"{default_name}{FlyfusImageGenerateTool._guess_extension(url, content_type)}"
        except Exception as error:
            if on_event:
                on_event(
                    "failed",
                    input_kind=input_kind,
                    index=index,
                    source_url=url,
                    elapsed_ms=FlyfusImageGenerateTool._elapsed_ms(started_at),
                    exception_type=type(error).__name__,
                )
            raise
        if on_event:
            on_event(
                "succeeded",
                input_kind=input_kind,
                index=index,
                source_url=url,
                elapsed_ms=FlyfusImageGenerateTool._elapsed_ms(started_at),
                payload_size=downloaded,
                content_type=content_type or "-",
                payload_sha256=hashlib.sha256(image_file.getvalue()).hexdigest(),
            )
        return image_file

    @staticmethod
    def _guess_extension(url: str, content_type: str) -> str:
        if content_type:
            extension = FlyfusImageGenerateTool._extension_for_mime_type(content_type)
            if extension:
                return extension

        guessed_type, _ = mimetypes.guess_type(urlparse(url).path)
        return (
            FlyfusImageGenerateTool._extension_for_mime_type(guessed_type or "")
            or ".png"
        )

    @staticmethod
    def _extension_for_mime_type(mime_type: str) -> str:
        if mime_type == "image/jpeg":
            return ".jpg"
        return mimetypes.guess_extension(mime_type) or ".png"
