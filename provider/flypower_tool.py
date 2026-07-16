import logging
from typing import Any

from dify_plugin import ToolProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError

from tools.image._image_utils import ModelListRequestError, fetch_openai_model_ids, image_model_ids

logger = logging.getLogger(__name__)


class FlypowerToolProvider(ToolProvider):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        endpoint_url = str(credentials.get("endpoint_url") or "").strip()
        api_key = str(credentials.get("api_key") or "").strip()
        if not endpoint_url and not api_key:
            return
        if not endpoint_url:
            raise ToolProviderCredentialValidationError("请填写 API 地址")
        if not api_key:
            raise ToolProviderCredentialValidationError("请填写 API Key")

        try:
            available_models = fetch_openai_model_ids(endpoint_url, api_key)
        except ValueError as error:
            raise ToolProviderCredentialValidationError(str(error)) from error
        except ModelListRequestError as error:
            logger.warning(
                "flypower.model_list_validation_failed category=%s endpoint=%s status_code=%s",
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
