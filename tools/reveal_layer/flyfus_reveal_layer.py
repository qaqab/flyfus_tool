import base64
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
import json
import re
import time
from pathlib import Path
from urllib.parse import unquote, urlparse
import uuid

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from PIL import Image, ImageDraw
import requests

from tools._sls_logging import write_tool_log


REVEAL_LAYER_BASE_URL = "https://api.research.360.cn/v1"
MODEL = "reveal_layer"
MAX_IMAGE_BYTES = 50 * 1024 * 1024
DOWNLOAD_TIMEOUT = (10, 120)
API_TIMEOUT = (10, 60)
MAX_WAIT_SECONDS = 180
POLL_INTERVAL_SECONDS = 3


@dataclass(frozen=True)
class LayerSpec:
    name: str
    bbox: tuple[int, int, int, int]


def parse_layers_json(value: object) -> list[LayerSpec]:
    if isinstance(value, str):
        try:
            raw_layers = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("layers_json 必须是有效的 JSON 数组") from error
    else:
        raw_layers = value
    if not isinstance(raw_layers, list):
        raise ValueError("layers_json 必须是 JSON 数组")
    if not 1 <= len(raw_layers) <= 6:
        raise ValueError("crop 模式需要 1 到 6 个图层框")

    layers: list[LayerSpec] = []
    for index, item in enumerate(raw_layers, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"第 {index} 个图层必须是 JSON 对象")
        name = str(item.get("name") or "").strip()
        bbox = item.get("bbox")
        if not name:
            raise ValueError(f"第 {index} 个图层缺少 name")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"第 {index} 个图层 bbox 必须包含四个数字")
        try:
            coordinates = tuple(_integer(value, "bbox 坐标") for value in bbox)
        except ValueError as error:
            raise ValueError(f"第 {index} 个图层 {error}") from error
        x1, y1, x2, y2 = coordinates
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError(f"第 {index} 个图层 bbox 无效")
        layers.append(LayerSpec(name=name, bbox=coordinates))
    return layers


def build_submit_payload(
    image_data_url: str,
    layers: list[LayerSpec],
    *,
    version: str,
    pipeline_type: str,
    steps: int,
    seed: int,
) -> dict:
    payload = {
        "model": MODEL,
        "version": version,
        "time_out": 3600,
        "input": [{"type": "input_image", "image_url": image_data_url}],
        "pipeline_type": pipeline_type,
        "steps": steps,
        "seed": seed,
    }
    if pipeline_type == "crop":
        payload["image_boxes"] = [list(layer.bbox) for layer in layers]
    return payload


def original_rgb_layer(source_rgb: Image.Image, alpha: Image.Image) -> Image.Image:
    if alpha.size != source_rgb.size:
        alpha = alpha.resize(source_rgb.size, Image.Resampling.LANCZOS)
    result = source_rgb.convert("RGBA")
    result.putalpha(alpha.convert("L"))
    return result


class FlyfusRevealLayerTool(Tool):
    def _invoke(self, tool_parameters: dict) -> Generator[ToolInvokeMessage, None, None]:
        log_id = str(uuid.uuid4())
        started_at = time.monotonic()
        stage = "validate_input"
        try:
            image_url = self._validate_image_url(tool_parameters.get("image_url"))
            version = str(tool_parameters.get("version") or "v2.3").strip()
            pipeline_type = str(tool_parameters.get("pipeline_type") or "crop").strip().lower()
            steps = _integer(tool_parameters.get("steps", 10), "steps")
            seed = _integer(tool_parameters.get("seed", 42), "seed")
            debug = _boolean(tool_parameters.get("debug", False), "debug")
            self._validate_options(version, pipeline_type, steps, seed)
            layers = (
                parse_layers_json(tool_parameters.get("layers_json"))
                if pipeline_type == "crop"
                else []
            )
            stage = "download_source"
            source_bytes, mime_type, source_rgb = self._download_source(image_url)
            stage = "validate_layers"
            self._validate_layers(layers, source_rgb.size)
            layer_plan = [{"name": layer.name, "bbox": list(layer.bbox)} for layer in layers]
            self._write_log(
                log_id,
                "input_checked",
                image_name=_url_filename(image_url),
                image_host=urlparse(image_url).hostname or "-",
                image_mime_type=mime_type,
                image_bytes=len(source_bytes),
                image_size=_log_json(list(source_rgb.size)),
                pipeline_type=pipeline_type,
                version=version,
                steps=steps,
                seed=seed,
                debug=debug,
                requested_layers=len(layers),
                layers_json=_log_json(layer_plan),
                checks_json=_log_json(
                    {
                        "source_image_valid": True,
                        "layer_boxes_in_bounds": True,
                        "layer_boxes_large_enough": True,
                    }
                ),
            )
            payload = build_submit_payload(
                self._data_url(source_bytes, mime_type),
                layers,
                version=version,
                pipeline_type=pipeline_type,
                steps=steps,
                seed=seed,
            )
            self._write_log(
                log_id,
                "submit_request",
                endpoint=f"{REVEAL_LAYER_BASE_URL}/submit_task",
                request_json=_log_json(
                    {
                        **{key: value for key, value in payload.items() if key != "input"},
                        "input": [
                            {
                                "type": "input_image",
                                "image_name": _url_filename(image_url),
                                "mime_type": mime_type,
                                "bytes": len(source_bytes),
                            }
                        ],
                    }
                ),
            )
            stage = "reveal_layer_api"
            reveal_started_at = time.monotonic()
            completed = self._submit_and_wait(payload, version, log_id)
            reveal_api_time = round(time.monotonic() - reveal_started_at, 3)
            stage = "build_delivery_images"
            build_started_at = time.monotonic()
            delivery_images = self._build_delivery_images(source_rgb, layers, completed)
            build_images_time = round(time.monotonic() - build_started_at, 3)
            layer_count = len(delivery_images) - 1
            output = completed.get("output") if isinstance(completed.get("output"), dict) else {}
            returned_images = output.get("layers_base") if isinstance(output, dict) else []
            self._write_log(
                log_id,
                "delivery_images_created",
                requested_layer_count=len(layers),
                returned_layer_count=layer_count,
                returned_images_json=_log_json(returned_images),
                files_json=_log_json(
                    [
                        {"filename": filename, "bytes": len(content)}
                        for filename, content in delivery_images
                    ]
                ),
                checks_json=_log_json(
                    {
                        "layer_count_difference_allowed": abs(layer_count - len(layers)) <= 1,
                        "background_included": bool(delivery_images)
                        and delivery_images[0][0] == "background.png",
                        "foregrounds_restored_from_original_rgb": True,
                        "all_images_are_png": all(
                            content.startswith(b"\x89PNG\r\n\x1a\n")
                            for _, content in delivery_images
                        ),
                    }
                ),
            )
            stage = "upload_layer_images"
            upload_started_at = time.monotonic()
            image_urls = self._upload_images(delivery_images, log_id)
            upload_images_time = round(time.monotonic() - upload_started_at, 3)
            boxed_image_url = ""
            debug_image_time = 0.0
            if debug:
                stage = "upload_debug_image"
                debug_started_at = time.monotonic()
                debug_image = self._build_boxed_preview(source_rgb, layers)
                boxed_image_url = self._upload_images(
                    [("debug_boxes.png", _png_bytes(debug_image))],
                    log_id,
                )[0]
                debug_image_time = round(time.monotonic() - debug_started_at, 3)
            result = {
                "boxed_image_url": boxed_image_url,
                "image_urls": image_urls,
                "layer_count": layer_count,
                "source_size": list(source_rgb.size),
                "version": version,
                "pipeline_type": pipeline_type,
                "steps": steps,
                "seed": seed,
                "debug": debug,
                "generation_time": completed.get("generation_time"),
                "reveal_api_time": reveal_api_time,
                "build_images_time": build_images_time,
                "upload_images_time": upload_images_time,
                "debug_image_time": debug_image_time,
                "total_time": round(time.monotonic() - started_at, 3),
            }
            self._write_log(log_id, "succeeded", **result)
            result["log_id"] = log_id
            yield self.create_json_message(result)
            yield self.create_text_message(
                json.dumps(
                    {
                        "log_id": log_id,
                        "boxed_image_url": boxed_image_url,
                        "image_urls": image_urls,
                    },
                    ensure_ascii=False,
                )
            )
        except Exception as error:
            self._write_log(
                log_id,
                "failed",
                stage=stage,
                elapsed_ms=round((time.monotonic() - started_at) * 1000),
                exception_type=type(error).__name__,
                error=str(error),
            )
            message = f"RevealLayer 图层拆解失败：{error}"
            error_result = {
                "boxed_image_url": "",
                "image_urls": [],
                "error": message,
                "log_id": log_id,
            }
            yield self.create_json_message(error_result)
            yield self.create_text_message(json.dumps(error_result, ensure_ascii=False))

    @staticmethod
    def _validate_image_url(value: object) -> str:
        url = str(value or "").strip()
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("image_url 必须是单张 HTTP 或 HTTPS 图片 URL")
        return url

    @staticmethod
    def _validate_options(version: str, pipeline_type: str, steps: int, seed: int) -> None:
        if not version or len(version) > 32:
            raise ValueError("version 无效")
        if pipeline_type not in {"crop", "all"}:
            raise ValueError("pipeline_type 只能是 crop 或 all")
        if not 10 <= steps <= 30:
            raise ValueError("steps 必须在 10 到 30 之间")
        if not 0 <= seed <= 0xFFFFFFFF:
            raise ValueError("seed 必须在 0 到 4294967295 之间")

    @staticmethod
    def _validate_layers(layers: list[LayerSpec], size: tuple[int, int]) -> None:
        width, height = size
        server_scale = 1024 / max(width, height)
        for layer in layers:
            x1, y1, x2, y2 = layer.bbox
            if x2 > width or y2 > height:
                raise ValueError(f"图层 {layer.name} 的 bbox 超出原图范围")
            if min(x2 - x1, y2 - y1) * server_scale < 8:
                raise ValueError(
                    f"图层 {layer.name} 的 bbox 太小；请不要单独框细碎小对象，"
                    "将它并入相邻主体或面板"
                )

    @staticmethod
    def _download_source(url: str) -> tuple[bytes, str, Image.Image]:
        response = requests.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True)
        response.raise_for_status()
        content = _read_bounded_response(response)
        with Image.open(BytesIO(content)) as image:
            image.load()
            if image.format not in {"PNG", "JPEG", "WEBP"}:
                raise ValueError("输入图片只支持 PNG、JPEG 或 WebP")
            width, height = image.size
            if min(width, height) < 128:
                raise ValueError("输入图片最短边不能小于 128 像素")
            if max(width, height) / min(width, height) > 5:
                raise ValueError("输入图片宽高比不能超过 5")
            source_rgb = image.convert("RGB")
            mime_type = {
                "PNG": "image/png",
                "JPEG": "image/jpeg",
                "WEBP": "image/webp",
            }[image.format]
        return content, mime_type, source_rgb

    @staticmethod
    def _data_url(content: bytes, mime_type: str) -> str:
        return f"data:{mime_type};base64,{base64.b64encode(content).decode('ascii')}"

    def _submit_and_wait(self, payload: dict, version: str, log_id: str = "") -> dict:
        api_key = str(self.runtime.credentials.get("reveal_layer_api_key") or "").strip()
        if not api_key:
            raise ValueError("未配置 reveal_layer_api_key")
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        submitted = self._post("submit_task", payload, headers)
        if log_id:
            self._write_log(
                log_id,
                "submit_response",
                endpoint=f"{REVEAL_LAYER_BASE_URL}/submit_task",
                response_json=_log_json(submitted),
            )
        task_id = str(submitted.get("task_id") or "").strip()
        if not task_id:
            raise RuntimeError("RevealLayer 提交响应缺少 task_id")

        deadline = time.monotonic() + MAX_WAIT_SECONDS
        poll_count = 0
        while True:
            poll_count += 1
            result = self._post(
                "query_task",
                {"model": MODEL, "version": version, "task_id": task_id},
                headers,
            )
            status = str(result.get("status") or "").lower()
            if status in {"done", "failed", "not_found"} and log_id:
                self._write_log(
                    log_id,
                    "query_response",
                    endpoint=f"{REVEAL_LAYER_BASE_URL}/query_task",
                    task_id=task_id,
                    poll_count=poll_count,
                    status=status,
                    response_json=_log_json(result),
                )
            if status == "done":
                if not isinstance(result.get("output"), dict):
                    raise RuntimeError("RevealLayer 完成响应缺少 output")
                return result
            if status in {"failed", "not_found"}:
                raise RuntimeError(f"RevealLayer 任务状态为 {status}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"RevealLayer 任务超过 {MAX_WAIT_SECONDS} 秒")
            time.sleep(POLL_INTERVAL_SECONDS)

    @staticmethod
    def _post(endpoint: str, payload: dict, headers: dict[str, str]) -> dict:
        response = requests.post(
            f"{REVEAL_LAYER_BASE_URL}/{endpoint}",
            headers=headers,
            json=payload,
            timeout=API_TIMEOUT,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as error:
            body = response.text[:4000]
            raise RuntimeError(f"{endpoint} HTTP {response.status_code}: {body}") from error
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("RevealLayer 返回了无效 JSON")
        if data.get("response_status") == -1:
            raise RuntimeError(str(data.get("message") or "RevealLayer 请求失败"))
        return data

    def _build_delivery_images(
        self,
        source_rgb: Image.Image,
        layers: list[LayerSpec],
        completed: dict,
    ) -> list[tuple[str, bytes]]:
        output = completed["output"]
        urls = output.get("layers_base")
        if not isinstance(urls, list) or len(urls) < 2:
            raise RuntimeError("RevealLayer 未返回背景和前景基础图层")
        returned_layer_count = len(urls) - 1
        if layers and abs(returned_layer_count - len(layers)) > 1:
            raise RuntimeError(
                f"RevealLayer 请求 {len(layers)} 个前景，实际返回 {returned_layer_count} 个，"
                "数量差异超过 1"
            )
        mapping = output.get("boxes_mapping_index")
        mapping = mapping if isinstance(mapping, list) else []

        with ThreadPoolExecutor(max_workers=min(8, len(urls))) as executor:
            downloaded = list(executor.map(self._download_output_image, map(str, urls)))

        background = downloaded[0].convert("RGBA")
        if background.size != source_rgb.size:
            background = background.resize(source_rgb.size, Image.Resampling.LANCZOS)
        prepared_images: list[tuple[str, Image.Image]] = [("background.png", background)]

        for index, downloaded_layer in enumerate(downloaded[1:], start=1):
            base_layer = downloaded_layer.convert("RGBA")
            layer_name = self._layer_name(index - 1, layers, mapping)
            filename = f"layer_{index:02d}_{_safe_name(layer_name)}.png"
            image = original_rgb_layer(source_rgb, base_layer.getchannel("A"))
            prepared_images.append((filename, image))

        with ThreadPoolExecutor(max_workers=min(8, len(prepared_images))) as executor:
            encoded = list(executor.map(_png_bytes, [image for _, image in prepared_images]))
        return [(prepared_images[index][0], content) for index, content in enumerate(encoded)]

    @staticmethod
    def _build_boxed_preview(source_rgb: Image.Image, layers: list[LayerSpec]) -> Image.Image:
        preview = source_rgb.convert("RGB").copy()
        draw = ImageDraw.Draw(preview)
        line_width = max(2, round(max(preview.size) / 400))
        colors = ["#ff3b30", "#007aff", "#34c759", "#ff9500", "#af52de", "#00a7a7"]
        for index, layer in enumerate(layers, start=1):
            x1, y1, x2, y2 = layer.bbox
            color = colors[(index - 1) % len(colors)]
            draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
            label = f"{index} [{x1},{y1},{x2},{y2}]"
            text_box = draw.textbbox((x1, y1), label, stroke_width=1)
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]
            text_y = y1 if y1 + text_height + 6 < y2 else max(0, y1 - text_height - 6)
            draw.rectangle(
                (x1, text_y, x1 + text_width + 6, text_y + text_height + 6),
                fill=color,
            )
            draw.text(
                (x1 + 3, text_y + 3),
                label,
                fill="white",
                stroke_width=1,
                stroke_fill="black",
            )
        return preview

    @staticmethod
    def _layer_name(index: int, layers: list[LayerSpec], mapping: list) -> str:
        if not layers:
            return f"图层{index + 1}"
        try:
            source_index = int(mapping[index]) if index < len(mapping) else index
        except (TypeError, ValueError):
            source_index = index
        return layers[source_index].name if 0 <= source_index < len(layers) else f"图层{index + 1}"

    @staticmethod
    def _download_output_image(url: str) -> Image.Image:
        response = requests.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True)
        response.raise_for_status()
        with Image.open(BytesIO(_read_bounded_response(response))) as image:
            image.load()
            return image.copy()

    def _upload_images(self, images: list[tuple[str, bytes]], log_id: str) -> list[str]:
        base_url = str(self.runtime.credentials.get("oss_api_base_url") or "").strip().rstrip("/")
        token = str(self.runtime.credentials.get("oss_api_token") or "").strip()
        if not base_url or not token:
            raise ValueError("OSS API 地址或 Token 未配置")
        endpoint = f"{base_url}/v1/oss-assets/file-file/upload"
        self._write_log(
            log_id,
            "oss_batch_upload_request",
            endpoint=endpoint,
            file_count=len(images),
            files_json=_log_json(
                [{"filename": name, "bytes": len(content)} for name, content in images]
            ),
        )

        def upload(image: tuple[str, bytes]) -> tuple[str, dict]:
            filename, content = image
            upload_name = f"reveal_layer_{uuid.uuid4().hex}.png"
            response = requests.post(
                endpoint,
                headers={"Accept": "application/json", "Authorization": f"Bearer {token}"},
                files={"file": (upload_name, content, "image/png")},
                timeout=DOWNLOAD_TIMEOUT,
            )
            try:
                response.raise_for_status()
            except requests.HTTPError as error:
                body = response.text[:4000]
                raise RuntimeError(f"OSS 上传 {filename} HTTP {response.status_code}: {body}") from error
            payload = response.json()
            if not isinstance(payload, dict) or not payload.get("success"):
                raise RuntimeError(f"OSS 上传 {filename} 失败：{_log_json(payload)}")
            try:
                public_url = payload["data"]["public_url"]
            except (KeyError, TypeError):
                raise RuntimeError(f"OSS 上传 {filename} 未返回 public_url") from None
            if not isinstance(public_url, str) or not public_url:
                raise RuntimeError(f"OSS 上传 {filename} 返回了无效 public_url")
            return public_url, payload

        with ThreadPoolExecutor(max_workers=min(8, len(images))) as executor:
            uploaded = list(executor.map(upload, images))
        urls = [url for url, _ in uploaded]
        self._write_log(
            log_id,
            "oss_batch_upload_response",
            endpoint=endpoint,
            file_count=len(urls),
            responses_json=_log_json([payload for _, payload in uploaded]),
        )
        return urls

    def _write_log(self, log_id: str, event: str, **fields: object) -> None:
        write_tool_log(self.runtime.credentials, log_id, f"reveal_layer_{event}", **fields)


def _read_bounded_response(response: requests.Response) -> bytes:
    chunks: list[bytes] = []
    size = 0
    for chunk in response.iter_content(chunk_size=256 * 1024):
        if not chunk:
            continue
        size += len(chunk)
        if size > MAX_IMAGE_BYTES:
            raise ValueError(f"图片大小不能超过 {MAX_IMAGE_BYTES // 1024 // 1024} MB")
        chunks.append(chunk)
    if not chunks:
        raise ValueError("图片 URL 返回了空内容")
    return b"".join(chunks)


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是整数")
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} 必须是整数") from None
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} 必须是整数")
    if isinstance(value, str) and str(number) != value.strip():
        raise ValueError(f"{name} 必须是整数")
    return number


def _boolean(value: object, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError(f"{name} 必须是布尔值")


def _png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _url_filename(url: str) -> str:
    filename = Path(unquote(urlparse(url).path)).name
    return filename or "image"


def _log_json(value: object) -> str:
    return json.dumps(_sanitize_log_value(value), ensure_ascii=False, separators=(",", ":"))


def _sanitize_log_value(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _sanitize_log_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_log_value(item) for item in value]
    if isinstance(value, str):
        if value.startswith("data:image/"):
            return "<image_base64 omitted>"
        parsed = urlparse(value)
        if parsed.scheme in {"http", "https"}:
            suffix = Path(parsed.path).suffix.lower()
            if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
                return _url_filename(value)
    return value


def _safe_name(value: str) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip(" ._") or "图层"
