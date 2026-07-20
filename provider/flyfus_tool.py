import logging
from typing import Any
from urllib.parse import urlparse

from dify_plugin import ToolProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError

from tools.image._image_utils import ModelListRequestError, fetch_openai_model_ids, image_model_ids

logger = logging.getLogger(__name__)


class FlyfusToolProvider(ToolProvider):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        endpoint_url = str(credentials.get("endpoint_url") or "").strip()
        api_key = str(credentials.get("api_key") or "").strip()
        if not endpoint_url:
            raise ToolProviderCredentialValidationError("请填写 API 地址")
        if not api_key:
            raise ToolProviderCredentialValidationError("请填写 API Key")
        oss_api_base_url = str(credentials.get("oss_api_base_url") or "").strip()
        oss_api_token = str(credentials.get("oss_api_token") or "").strip()
        if not self._is_https_base_url(oss_api_base_url):
            raise ToolProviderCredentialValidationError("请填写有效的 HTTPS OSS API 基础地址")
        if not oss_api_token:
            raise ToolProviderCredentialValidationError("请填写 OSS API Token")
        sls_endpoint = str(credentials.get("sls_endpoint") or "").strip()
        sls_project = str(credentials.get("sls_project") or "").strip()
        sls_access_key_id = str(credentials.get("sls_access_key_id") or "").strip()
        sls_access_key_secret = str(credentials.get("sls_access_key_secret") or "").strip()
        if not self._is_https_base_url(sls_endpoint):
            raise ToolProviderCredentialValidationError("请填写有效的 HTTPS SLS 地址")
        if not sls_project:
            raise ToolProviderCredentialValidationError("请填写 SLS 项目")
        if not sls_access_key_id:
            raise ToolProviderCredentialValidationError("请填写 SLS AccessKey ID")
        if not sls_access_key_secret:
            raise ToolProviderCredentialValidationError("请填写 SLS AccessKey Secret")

        try:
            available_models = fetch_openai_model_ids(endpoint_url, api_key)
        except ValueError as error:
            raise ToolProviderCredentialValidationError(str(error)) from error
        except ModelListRequestError as error:
            logger.warning(
                "flyfus.model_list_validation_failed category=%s endpoint=%s status_code=%s",
                error.category,
                error.endpoint,
                error.status_code,
            )
            raise ToolProviderCredentialValidationError(f"/models 校验失败：{error}") from error

        supported_models = image_model_ids()
        matched_models = sorted(supported_models & available_models)
        if not matched_models:
            expected_models = ", ".join(sorted(supported_models))
            raise ToolProviderCredentialValidationError(
                f"/models 未返回工具支持的图像模型。需要至少一个：{expected_models}"
            )

    @staticmethod
    def _is_https_base_url(value: str) -> bool:
        parsed = urlparse(value)
        return parsed.scheme == "https" and bool(parsed.netloc) and not parsed.query and not parsed.fragment
