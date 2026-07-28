from __future__ import annotations

import re
from collections.abc import Generator
from urllib.parse import urlparse

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage


class ReadFileTool(Tool):
    _TAG_NAME = "FLYFUS_CONTEXT"
    _INTERNAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "web", "nginx", "api"}
    _IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
    _FILE_SUFFIXES = (".pdf", ".md", ".xlsx", ".csv", ".txt", ".html")

    def _invoke(self, tool_parameters: dict) -> Generator[ToolInvokeMessage, None, None]:
        urls, invalid_urls = self._parse_urls(tool_parameters.get("url_list"))
        if invalid_urls:
            error = "Unsupported URL file type. Supported: PNG, JPG, JPEG, PDF, MD, XLSX, CSV, TXT, HTML."
            yield self.create_json_message({"urls": [], "error": error})
            yield self.create_text_message("[]")
            return
        if not urls:
            error = "Provide at least one public HTTP or HTTPS URL."
            yield self.create_json_message({"urls": [], "error": error})
            yield self.create_text_message("[]")
            return

        yield self.create_text_message(self._format_context(urls))

    @classmethod
    def _format_context(cls, urls: list[str]) -> str:
        image_count = 0
        file_count = 0
        lines: list[str] = []

        for url in urls:
            # Labels make tool output readable; the model plugin extracts only the URL.
            if urlparse(url).path.lower().endswith(cls._IMAGE_SUFFIXES):
                image_count += 1
                label = f"image{image_count}"
            else:
                file_count += 1
                label = f"file{file_count}"
            lines.append(f'{label}: "{url}"')

        return f"<{cls._TAG_NAME}>\n{"\n".join(lines)}\n</{cls._TAG_NAME}>"

    @classmethod
    def _parse_urls(cls, value: object) -> tuple[list[str], list[str]]:
        if not isinstance(value, str):
            return [], []

        urls: list[str] = []
        invalid_urls: list[str] = []
        seen: set[str] = set()
        for raw_url in re.split(r"[,\r\n]+", value):
            url = raw_url.strip().strip('"').strip("'")
            if not url or url in seen:
                continue
            if not cls._is_public_http_url(url) or not cls._is_supported_url(url):
                invalid_urls.append(url)
                continue
            seen.add(url)
            urls.append(url)
        return urls, invalid_urls

    @classmethod
    def _is_public_http_url(cls, url: str) -> bool:
        parsed = urlparse(url)
        return (
            parsed.scheme in {"http", "https"}
            and bool(parsed.hostname)
            and parsed.hostname not in cls._INTERNAL_HOSTS
        )

    @classmethod
    def _is_supported_url(cls, url: str) -> bool:
        path = urlparse(url).path.lower()
        return path.endswith(cls._IMAGE_SUFFIXES + cls._FILE_SUFFIXES)
