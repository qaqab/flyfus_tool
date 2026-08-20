from __future__ import annotations

from collections.abc import Generator
from datetime import datetime
from zoneinfo import ZoneInfo

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
import requests


class CreateTextFileTool(Tool):
    _MAX_CONTENT_BYTES = 5 * 1024 * 1024
    _MIME_TYPES = {
        "txt": "text/plain; charset=utf-8",
        "md": "text/markdown; charset=utf-8",
        "csv": "text/csv; charset=utf-8",
        "json": "application/json; charset=utf-8",
        "yaml": "application/yaml; charset=utf-8",
    }

    def _invoke(self, tool_parameters: dict) -> Generator[ToolInvokeMessage, None, None]:
        content = tool_parameters.get("content")
        file_format = tool_parameters.get("format")
        filename = tool_parameters.get("filename")

        if not isinstance(content, str):
            raise ValueError("content 必须是文本")
        payload = content.encode("utf-8")
        if len(payload) > self._MAX_CONTENT_BYTES:
            raise ValueError("文件内容不能超过 5 MiB")
        if file_format not in self._MIME_TYPES:
            raise ValueError("format 只能是 txt、md、csv、json 或 yaml")

        upload_name = self._build_filename(filename, file_format)
        public_url = self._upload(
            upload_name,
            payload,
            self._MIME_TYPES[file_format],
        )
        yield self.create_json_message(
            {
                "url": public_url,
                "filename": upload_name,
                "format": file_format,
                "size": len(payload),
            }
        )

    @classmethod
    def _build_filename(cls, value: object, file_format: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("filename 不能为空")

        filename = value.strip()
        if filename in {".", ".."} or any(char in filename for char in ("/", "\\", "\0")):
            raise ValueError("filename 不能包含路径或空字符")

        suffix = f".{file_format}"
        if filename.lower().endswith(suffix):
            filename = filename[: -len(suffix)].rstrip()
        if not filename:
            raise ValueError("filename 不能为空")

        date_suffix = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")
        return f"{filename}_{date_suffix}{suffix}"

    def _upload(self, filename: str, payload: bytes, mime_type: str) -> str:
        base_url = str(self.runtime.credentials.get("oss_api_base_url") or "").strip().rstrip("/")
        token = str(self.runtime.credentials.get("oss_api_token") or "").strip()
        if not base_url or not token:
            raise RuntimeError("OSS API 地址或 Token 未配置")

        try:
            response = requests.post(
                f"{base_url}/v1/oss-assets/file-file/upload",
                headers={"Accept": "application/json", "Authorization": f"Bearer {token}"},
                files={
                    "file": (filename, payload, mime_type),
                    "filename": (None, filename),
                },
                timeout=(10.0, 120.0),
                allow_redirects=False,
            )
        except requests.Timeout as error:
            raise RuntimeError("OSS 上传文本文件超时") from error
        except requests.RequestException as error:
            raise RuntimeError(f"OSS 上传文本文件请求失败：{type(error).__name__}") from error

        if not 200 <= response.status_code < 300:
            request_id = response.headers.get("x-fc-request-id") or response.headers.get("x-request-id")
            request_id_detail = f"，request_id={request_id}" if request_id else ""
            raise RuntimeError(f"OSS 上传文本文件 HTTP {response.status_code}{request_id_detail}")

        try:
            response_body = response.json()
        except (ValueError, requests.JSONDecodeError):
            raise RuntimeError("OSS 上传文本文件返回了无效 JSON") from None
        if not isinstance(response_body, dict) or not response_body.get("success"):
            raise RuntimeError("OSS 上传文本文件失败：接口返回 success=false")

        data = response_body.get("data")
        public_url = data.get("public_url") if isinstance(data, dict) else None
        if not isinstance(public_url, str) or not public_url:
            raise RuntimeError("OSS 上传文本文件未返回有效 public_url")
        return public_url
