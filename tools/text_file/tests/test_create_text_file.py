from __future__ import annotations

from datetime import datetime

import pytest

from tools.text_file.create_text_file import CreateTextFileTool
import requests


class FakeResponse:
    status_code = 200
    headers: dict = {}

    @staticmethod
    def json() -> dict:
        return {"success": True, "data": {"public_url": "https://cdn.example/report.md"}}


def test_build_filename_appends_shanghai_date_and_avoids_duplicate_extension(monkeypatch) -> None:
    class FixedDateTime:
        @classmethod
        def now(cls, timezone):
            return datetime(2026, 8, 20, tzinfo=timezone)

    monkeypatch.setattr("tools.text_file.create_text_file.datetime", FixedDateTime)

    assert CreateTextFileTool._build_filename("销售报告.md", "md") == "销售报告_20260820.md"


def test_invoke_uploads_original_utf8_content_and_returns_metadata(monkeypatch) -> None:
    request: dict = {}

    def fake_post(url, **kwargs):
        request.update(url=url, **kwargs)
        return FakeResponse()

    monkeypatch.setattr("tools.text_file.create_text_file.requests.post", fake_post)
    monkeypatch.setattr(
        CreateTextFileTool,
        "_build_filename",
        classmethod(lambda cls, value, file_format: "报告_20260820.md"),
    )
    tool = CreateTextFileTool.from_credentials(
        {"oss_api_base_url": "https://oss.example/", "oss_api_token": "test-token"}
    )

    messages = list(
        tool.invoke(
            tool_parameters={"content": "# 标题\n正文", "format": "md", "filename": "报告"}
        )
    )

    assert request["url"] == "https://oss.example/v1/oss-assets/file-file/upload"
    assert request["headers"]["Authorization"] == "Bearer test-token"
    assert request["files"]["file"] == (
        "报告_20260820.md",
        "# 标题\n正文".encode(),
        "text/markdown; charset=utf-8",
    )
    result = messages[0].message.json_object
    assert result == {
        "url": "https://cdn.example/report.md",
        "filename": "报告_20260820.md",
        "format": "md",
        "size": 15,
    }


@pytest.mark.parametrize("filename", ["", "../report", "folder/report", "folder\\report"])
def test_build_filename_rejects_empty_or_path_names(filename: str) -> None:
    with pytest.raises(ValueError):
        CreateTextFileTool._build_filename(filename, "txt")


def test_invoke_rejects_content_over_5_mib() -> None:
    tool = CreateTextFileTool.from_credentials({})

    with pytest.raises(ValueError, match="5 MiB"):
        list(
            tool.invoke(
                tool_parameters={
                    "content": "a" * (5 * 1024 * 1024 + 1),
                    "format": "txt",
                    "filename": "large",
                }
            )
        )


def test_upload_reports_network_failure(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise requests.ConnectionError("unreachable")

    monkeypatch.setattr("tools.text_file.create_text_file.requests.post", fail)
    tool = CreateTextFileTool.from_credentials(
        {"oss_api_base_url": "https://oss.example", "oss_api_token": "token"}
    )

    with pytest.raises(RuntimeError, match="ConnectionError"):
        tool._upload("report.txt", b"text", "text/plain; charset=utf-8")


def test_upload_reports_http_status_and_request_id(monkeypatch) -> None:
    response = FakeResponse()
    response.status_code = 503
    response.headers = {"x-request-id": "req-123"}
    monkeypatch.setattr("tools.text_file.create_text_file.requests.post", lambda *args, **kwargs: response)
    tool = CreateTextFileTool.from_credentials(
        {"oss_api_base_url": "https://oss.example", "oss_api_token": "token"}
    )

    with pytest.raises(RuntimeError, match="HTTP 503，request_id=req-123"):
        tool._upload("report.txt", b"text", "text/plain; charset=utf-8")


def test_upload_rejects_unsuccessful_response(monkeypatch) -> None:
    response = FakeResponse()
    response.json = lambda: {"success": False, "data": {}}
    monkeypatch.setattr("tools.text_file.create_text_file.requests.post", lambda *args, **kwargs: response)
    tool = CreateTextFileTool.from_credentials(
        {"oss_api_base_url": "https://oss.example", "oss_api_token": "token"}
    )

    with pytest.raises(RuntimeError, match="success=false"):
        tool._upload("report.txt", b"text", "text/plain; charset=utf-8")
