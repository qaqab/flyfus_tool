from __future__ import annotations

import logging
import time
from typing import Any

from aliyun.log import LogClient, LogItem, PutLogsRequest


SLS_LOGSTORE = "flyfus-dify-llm-log"
SLS_LOG_TIMEOUT_SECONDS = 10
logger = logging.getLogger(__name__)


def write_tool_log(credentials: dict[str, Any], log_id: str, event: str, **fields: object) -> bool:
    endpoint = str(credentials.get("sls_endpoint") or "").strip()
    project = str(credentials.get("sls_project") or "").strip()
    access_key_id = str(credentials.get("sls_access_key_id") or "").strip()
    access_key_secret = str(credentials.get("sls_access_key_secret") or "").strip()
    if not endpoint or not project or not access_key_id or not access_key_secret:
        logger.warning("Flyfus tool log skipped: SLS credentials are incomplete", extra={"event": event, "log_id": log_id})
        return False

    contents = [("log_id", log_id), ("event", event), ("source", "flyfus_tool")]
    contents.extend((key, str(value)) for key, value in fields.items() if value is not None)
    try:
        log_item = LogItem()
        log_item.set_time(int(time.time()))
        log_item.set_contents(contents)
        client = LogClient(endpoint, access_key_id, access_key_secret)
        client.timeout = SLS_LOG_TIMEOUT_SECONDS
        client.put_logs(
            PutLogsRequest(project, SLS_LOGSTORE, "flyfus-tool", "", [log_item])
        )
        return True
    except Exception as error:
        # Diagnostic delivery must never change the tool invocation result.
        logger.warning(
            "Flyfus tool log delivery failed",
            extra={"event": event, "log_id": log_id, "exception_type": type(error).__name__},
        )
        return False
