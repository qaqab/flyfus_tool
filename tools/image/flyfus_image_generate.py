from __future__ import annotations

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
MAX_INPUT_DOWNLOAD_BYTES = 50 * 1024 * 1024
INPUT_DOWNLOAD_TIMEOUT = 300
OSS_UPLOAD_TIMEOUT = (10.0, 120.0)
MAX_OSS_UPLOAD_WORKERS = 4
MAX_IMAGE_REQUEST_RETRIES = 3
MAX_OSS_UPLOAD_RETRIES = 2


class FlyfusImageGenerateTool(Tool):
    def _invoke(self, tool_parameters: dict) -> Generator[ToolInvokeMessage, None, None]:
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
            yield from self._error_messages(error, log_id, request_context.get("request_fingerprint"))

        if not prompt or not isinstance(prompt, str):
            yield from fail("Error: Prompt is required.", "validation")
            return

        model = requested_model
        supported_models = image_model_ids()
        if model not in supported_models:
            yield from fail(f"Invalid model. Choose from: {', '.join(sorted(supported_models))}.", "validation")
            return

        try:
            reference_urls = self._parse_urls(tool_parameters.get("reference_image_urls"))
            mask_url = self._first_url(tool_parameters.get("mask_url"))
        except ValueError as error:
            yield from fail(str(error), "validation")
            return
        operation = "edit" if reference_urls else "generate"
        request_context.update({
            "request_fingerprint": self._request_fingerprint(model, prompt, reference_urls, mask_url, tool_parameters),
            "reference_count": len(reference_urls),
            "reference_hosts": ",".join(sorted({urlparse(url).netloc for url in reference_urls})) or "-",
            "has_mask": bool(mask_url),
        })
        self._write_invocation_log(log_id, "validated", model=model, operation=operation, **request_context)
        if not image_model_supports_operation(model, operation):
            yield from fail(f"Model {model} does not support {operation} in the image model YAML.", "validation")
            return

        api_key = str(self.runtime.credentials.get("api_key") or "")
        if not api_key:
            yield from fail("API key is required for image generation.", "credentials")
            return
        try:
            normalized_base_url = normalize_openai_base_url(self.runtime.credentials.get("endpoint_url"))
            if normalized_base_url is None:
                yield from fail("API endpoint is missing.", "credentials")
                return
            available_models = fetch_openai_model_ids(normalized_base_url, api_key)
        except ValueError as error:
            yield from fail(f"Invalid API endpoint: {error}", "model_validation")
            return
        except ModelListRequestError as error:
            yield from fail(f"Failed to validate API access: {error}", "model_validation")
            return
        if model not in available_models:
            matched_models = sorted(supported_models & available_models)
            if matched_models:
                yield from fail(f"Model {model} is not available from /models. Available image models: {', '.join(matched_models)}.", "model_validation")
            else:
                yield from fail(f"No supported image model was returned by /models. Expected one of: {', '.join(sorted(supported_models))}.", "model_validation")
            return

        # Keep retry behavior in this tool so every retry can be logged.
        client = OpenAI(api_key=api_key, base_url=normalized_base_url, max_retries=0)

        try:
            args: dict[str, Any] = {
                "model": model,
                "prompt": prompt,
            }
            error = self._apply_common_parameters(args, tool_parameters, model=model)
            if error:
                yield from fail(error, "parameter_validation")
                return

            if reference_urls:
                reference_count = len(reference_urls)
                if reference_count > MAX_REFERENCE_IMAGES:
                    yield from fail(f"Error: At most {MAX_REFERENCE_IMAGES} reference images are supported.", "parameter_validation")
                    return
                self._write_invocation_log(
                    log_id,
                    "request_started",
                    model=model,
                    operation=operation,
                    parameters=self._request_parameter_summary(args),
                    **request_context,
                )
                response = self._run_image_request_with_retry(
                    lambda: self._edit_images_with_files(
                        client,
                        args,
                        reference_urls,
                        mask_url,
                        on_download_event=lambda event, **fields: self._write_invocation_log(
                            log_id, f"reference_download_{event}", model=model, operation=operation, **request_context, **fields
                        ),
                    ),
                    on_attempt_started=lambda attempt: self._write_invocation_log(
                        log_id, "request_attempt_started", model=model, operation=operation, attempt=attempt, **request_context
                    ),
                    on_attempt_finished=lambda attempt, elapsed_ms, error=None: self._write_invocation_log(
                        log_id,
                        "request_attempt_failed" if error else "request_attempt_succeeded",
                        model=model,
                        operation=operation,
                        attempt=attempt,
                        elapsed_ms=elapsed_ms,
                        **self._error_log_fields(error),
                        **request_context,
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
            else:
                if mask_url:
                    yield from fail("Error: mask requires at least one reference image URL.", "parameter_validation")
                    return
                self._write_invocation_log(
                    log_id,
                    "request_started",
                    model=model,
                    operation=operation,
                    parameters=self._request_parameter_summary(args),
                    **request_context,
                )
                response = self._run_image_request_with_retry(
                    lambda: client.images.generate(**args),
                    on_attempt_started=lambda attempt: self._write_invocation_log(
                        log_id, "request_attempt_started", model=model, operation=operation, attempt=attempt, **request_context
                    ),
                    on_attempt_finished=lambda attempt, elapsed_ms, error=None: self._write_invocation_log(
                        log_id,
                        "request_attempt_failed" if error else "request_attempt_succeeded",
                        model=model,
                        operation=operation,
                        attempt=attempt,
                        elapsed_ms=elapsed_ms,
                        **self._error_log_fields(error),
                        **request_context,
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
            image_count=len(getattr(response, "data", [])),
            upstream_request_id=getattr(response, "_request_id", None) or "-",
            **request_context,
        )

        uploads: list[tuple[str, bytes | str, str, str]] = []
        try:
            for index, image in enumerate(getattr(response, "data", []), start=1):
                b64_json = getattr(image, "b64_json", None)
                image_url = getattr(image, "url", None)
                if b64_json:
                    mime_type, blob_image = decode_image(b64_json)
                    uploads.append(("file", blob_image, mime_type, self._output_filename(index, mime_type)))
                elif image_url:
                    uploads.append(("url", str(image_url), "", ""))
        except Exception as error:
            yield from fail(f"Failed to process generated images: {error}", "response_processing")
            return

        if not uploads:
            self._write_invocation_log(
                log_id,
                "response_empty",
                model=model,
                operation=operation,
                elapsed_ms=self._elapsed_ms(started_at),
                upstream_request_id=getattr(response, "_request_id", None) or "-",
                **request_context,
            )
            yield from fail("The image model did not return any images.", "response_processing")
            return

        try:
            self._write_invocation_log(
                log_id, "oss_upload_started", model=model, operation=operation, image_count=len(uploads), **request_context
            )
            with ThreadPoolExecutor(max_workers=min(MAX_OSS_UPLOAD_WORKERS, len(uploads))) as executor:
                upload_to_oss = partial(self._upload_output_to_oss, log_id=log_id)
                oss_urls = list(executor.map(upload_to_oss, uploads))
        except Exception as error:
            yield from fail(f"Failed to upload generated images to OSS (log_id={log_id}): {error}", "oss_upload")
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
            {"urls": oss_urls, "log": self._log_reference(log_id, request_context.get("request_fingerprint")), **usage_metadata}
        )
        yield self.create_text_message(json.dumps(oss_urls, ensure_ascii=False))

    def _error_messages(
        self, error: str, log_id: str, request_fingerprint: object | None
    ) -> Generator[ToolInvokeMessage, None, None]:
        yield self.create_json_message(
            {"urls": [], "error": error, "log": self._log_reference(log_id, request_fingerprint)}
        )
        yield self.create_text_message("[]")

    @staticmethod
    def _log_reference(log_id: str, request_fingerprint: object | None) -> dict[str, str]:
        reference = {"log_id": log_id}
        if request_fingerprint:
            reference["request_fingerprint"] = str(request_fingerprint)
        return reference

    @staticmethod
    def _run_image_request_with_retry(request, on_retry=None, on_attempt_started=None, on_attempt_finished=None):
        for attempt in range(MAX_IMAGE_REQUEST_RETRIES + 1):
            attempt_number = attempt + 1
            started_at = time.monotonic()
            if on_attempt_started:
                on_attempt_started(attempt_number)
            try:
                response = request()
            except Exception as error:
                if on_attempt_finished:
                    on_attempt_finished(attempt_number, FlyfusImageGenerateTool._elapsed_ms(started_at), error)
                if attempt >= MAX_IMAGE_REQUEST_RETRIES or not FlyfusImageGenerateTool._is_retryable_image_error(error):
                    raise
                if on_retry:
                    on_retry(attempt_number, error)
                time.sleep(0.5 * (attempt + 1))
            else:
                if on_attempt_finished:
                    on_attempt_finished(attempt_number, FlyfusImageGenerateTool._elapsed_ms(started_at))
                return response

        raise RuntimeError("Image request retry loop exited unexpectedly.")

    @staticmethod
    def _is_retryable_image_error(error: Exception) -> bool:
        message = str(error).lower()
        return "json_invalid" in message or (
            "invalid json" in message and ("<!doctype html" in message or "expected value" in message)
        ) or any(marker in message for marker in ("timeout", "connection", "rate limit", "429", "502", "503", "504"))

    @staticmethod
    def _request_fingerprint(
        model: str, prompt: str, reference_urls: list[str], mask_url: str | None, tool_parameters: dict
    ) -> str:
        # Groups externally retried calls without placing prompt text or URLs in SLS.
        payload = {
            "model": model,
            "prompt": prompt,
            "reference_urls": reference_urls,
            "mask_url": mask_url,
            "parameters": {key: value for key, value in tool_parameters.items() if key not in {"prompt", "reference_image_urls", "mask_url"}},
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _request_parameter_summary(args: dict[str, Any]) -> str:
        return json.dumps({key: value for key, value in args.items() if key != "prompt"}, ensure_ascii=True, sort_keys=True)

    @staticmethod
    def _error_log_fields(error: Exception | None) -> dict[str, object]:
        if error is None:
            return {}
        response = getattr(error, "response", None)
        status_code = getattr(error, "status_code", None) or getattr(response, "status_code", None)
        fields: dict[str, object] = {"exception_type": type(error).__name__}
        if status_code is not None:
            fields["status_code"] = status_code
        request_id = getattr(error, "request_id", None) or getattr(response, "_request_id", None)
        if request_id:
            fields["upstream_request_id"] = request_id
        return fields

    @staticmethod
    def _output_filename(index: int, mime_type: str) -> str:
        return f"generated_image_{index}{FlyfusImageGenerateTool._extension_for_mime_type(mime_type)}"

    def _upload_output_to_oss(self, upload: tuple[str, bytes | str, str, str], *, log_id: str) -> str:
        upload_type, payload, mime_type, filename = upload
        payload_size = len(payload) if isinstance(payload, bytes) else None
        payload_sha256 = hashlib.sha256(payload).hexdigest() if isinstance(payload, bytes) else None
        oss_api_base_url = str(self.runtime.credentials.get("oss_api_base_url") or "").strip().rstrip("/")
        oss_api_token = str(self.runtime.credentials.get("oss_api_token") or "")
        if not oss_api_base_url or not oss_api_token:
            raise RuntimeError("OSS API base URL and token are required.")

        headers = {"Accept": "application/json", "Authorization": f"Bearer {oss_api_token}"}
        endpoint = f"{oss_api_base_url}/v1/oss-assets/image-{'url' if upload_type == 'url' else 'file'}/upload"
        response = None
        started_at = time.monotonic()
        for attempt in range(MAX_OSS_UPLOAD_RETRIES + 1):
            try:
                if upload_type == "url":
                    response = requests.post(
                        endpoint,
                        headers=headers,
                        json={"image_url": payload},
                        timeout=OSS_UPLOAD_TIMEOUT,
                        allow_redirects=False,
                    )
                else:
                    response = requests.post(
                        endpoint,
                        headers=headers,
                        files={"file": (filename, payload, mime_type), "filename": (None, filename)},
                        timeout=OSS_UPLOAD_TIMEOUT,
                        allow_redirects=False,
                    )
            except requests.RequestException as error:
                if attempt < MAX_OSS_UPLOAD_RETRIES:
                    self._write_oss_log(
                        log_id,
                        "retry",
                        upload_type=upload_type,
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
                    upload_type=upload_type,
                    filename=filename,
                    mime_type=mime_type or "-",
                    payload_size=payload_size,
                    payload_sha256=payload_sha256,
                    elapsed_ms=self._elapsed_ms(started_at),
                    exception_type=type(error).__name__,
                )
                raise RuntimeError(f"OSS upload request failed (log_id={log_id}): {type(error).__name__}") from error

            if response.status_code < 500 and response.status_code != 429:
                break
            if attempt < MAX_OSS_UPLOAD_RETRIES:
                self._write_oss_log(
                    log_id,
                    "retry",
                    upload_type=upload_type,
                    filename=filename,
                    attempt=attempt + 1,
                    max_retries=MAX_OSS_UPLOAD_RETRIES,
                    status_code=response.status_code,
                )
                time.sleep(0.5 * (attempt + 1))
                continue
            break

        if response is None:
            raise RuntimeError(f"OSS upload did not return a response (log_id={log_id})")

        elapsed_ms = self._elapsed_ms(started_at)
        request_id = response.headers.get("x-fc-request-id") or response.headers.get("x-request-id") or ""

        if not 200 <= response.status_code < 300:
            response_text = str(getattr(response, "text", "")).strip()
            self._write_oss_log(
                log_id,
                "failed",
                upload_type=upload_type,
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
            raise RuntimeError(f"OSS upload returned HTTP {response.status_code}{request_id_detail} (log_id={log_id}){detail}")

        self._write_oss_log(
            log_id,
            "succeeded",
            upload_type=upload_type,
            endpoint=endpoint,
            filename=filename,
            mime_type=mime_type or "-",
            payload_size=payload_size,
            payload_sha256=payload_sha256,
            status_code=response.status_code,
            elapsed_ms=elapsed_ms,
            request_id=request_id or "-",
        )
        try:
            response_body = response.json()
            public_url = response_body["data"]["public_url"]
        except (KeyError, TypeError, ValueError, requests.JSONDecodeError):
            raise RuntimeError("OSS upload returned an invalid response") from None
        if not isinstance(public_url, str) or not public_url:
            raise RuntimeError("OSS upload returned an invalid public URL")
        return public_url

    def _write_oss_log(self, log_id: str, event: str, **fields: object) -> None:
        write_tool_log(self.runtime.credentials, log_id, f"oss_upload_{event}", **fields)

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
    def _first_url(value: object) -> str | None:
        urls = FlyfusImageGenerateTool._parse_urls(value)
        return urls[0] if urls else None

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
        mask_url: str | None,
        on_download_event=None,
    ) -> Any:
        image_files: list[io.BytesIO] = []
        mask_file: io.BytesIO | None = None
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
            multipart_args["image"] = image_files[0] if len(image_files) == 1 else image_files

            if mask_url:
                mask_file = FlyfusImageGenerateTool._download_input_image(
                    mask_url, default_name="mask_image", on_event=on_download_event, input_kind="mask", index=1
                )
                multipart_args["mask"] = mask_file

            return client.images.edit(**multipart_args)
        finally:
            for image_file in image_files:
                image_file.close()
            if mask_file:
                mask_file.close()

    @staticmethod
    def _download_input_image(url: str, *, default_name: str, on_event=None, input_kind: str, index: int) -> io.BytesIO:
        FlyfusImageGenerateTool._validate_http_url(url)
        started_at = time.monotonic()
        if on_event:
            on_event("started", input_kind=input_kind, index=index)
        if url.startswith("data:image/"):
            mime_type, image_data = decode_image(url)
            image_file = io.BytesIO(image_data)
            image_file.name = f"{default_name}{FlyfusImageGenerateTool._extension_for_mime_type(mime_type)}"
            if on_event:
                on_event("succeeded", input_kind=input_kind, index=index, elapsed_ms=FlyfusImageGenerateTool._elapsed_ms(started_at), payload_size=len(image_data), content_type=mime_type)
            return image_file

        try:
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
            image_file.name = f"{default_name}{FlyfusImageGenerateTool._guess_extension(url, content_type)}"
        except Exception as error:
            if on_event:
                on_event("failed", input_kind=input_kind, index=index, elapsed_ms=FlyfusImageGenerateTool._elapsed_ms(started_at), exception_type=type(error).__name__)
            raise
        if on_event:
            on_event("succeeded", input_kind=input_kind, index=index, elapsed_ms=FlyfusImageGenerateTool._elapsed_ms(started_at), payload_size=downloaded, content_type=content_type or "-")
        return image_file

    @staticmethod
    @staticmethod
    def _guess_extension(url: str, content_type: str) -> str:
        if content_type:
            extension = FlyfusImageGenerateTool._extension_for_mime_type(content_type)
            if extension:
                return extension

        guessed_type, _ = mimetypes.guess_type(urlparse(url).path)
        return FlyfusImageGenerateTool._extension_for_mime_type(guessed_type or "") or ".png"

    @staticmethod
    def _extension_for_mime_type(mime_type: str) -> str:
        if mime_type == "image/jpeg":
            return ".jpg"
        return mimetypes.guess_extension(mime_type) or ".png"

    @staticmethod
    def _to_namespace(value: Any) -> Any:
        if isinstance(value, dict):
            return SimpleNamespace(**{key: FlyfusImageGenerateTool._to_namespace(item) for key, item in value.items()})
        if isinstance(value, list):
            return [FlyfusImageGenerateTool._to_namespace(item) for item in value]
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
