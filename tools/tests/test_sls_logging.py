from __future__ import annotations

from tools._sls_logging import SLS_LOG_TIMEOUT_SECONDS, write_tool_log


def test_write_tool_log_uses_a_bounded_timeout(monkeypatch) -> None:
    clients = []

    class FakeLogItem:
        def set_time(self, value) -> None:
            pass

        def set_contents(self, value) -> None:
            pass

    class FakeClient:
        def __init__(self, *args) -> None:
            self.timeout = None
            clients.append(self)

        def put_logs(self, request) -> None:
            assert self.timeout == SLS_LOG_TIMEOUT_SECONDS

    monkeypatch.setattr("tools._sls_logging.LogItem", FakeLogItem)
    monkeypatch.setattr("tools._sls_logging.LogClient", FakeClient)

    uploaded = write_tool_log(
        {
            "sls_endpoint": "https://example.log.aliyuncs.com",
            "sls_project": "test-project",
            "sls_access_key_id": "test-key-id",
            "sls_access_key_secret": "test-key-secret",
        },
        "test-log-id",
        "image_started",
    )

    assert uploaded is True
    assert len(clients) == 1


def test_write_tool_log_reports_missing_credentials() -> None:
    assert write_tool_log({}, "test-log-id", "image_started") is False
