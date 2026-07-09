from typing import Any

from dify_plugin import ToolProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError
from openai import OpenAI

from tools._image_utils import extract_model_ids, image_model_ids, normalize_openai_base_url


class FlypowerImageProvider(ToolProvider):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        endpoint_url = str(credentials.get("endpoint_url") or "").strip()
        api_key = str(credentials.get("api_key") or "").strip()
        if not endpoint_url:
            raise ToolProviderCredentialValidationError("请填写 API 地址")
        if not api_key:
            raise ToolProviderCredentialValidationError("请填写 API Key")

        try:
            client = OpenAI(
                api_key=api_key,
                base_url=normalize_openai_base_url(endpoint_url),
            )
            available_models = extract_model_ids(client.models.list())
        except Exception as error:
            raise ToolProviderCredentialValidationError(f"凭据校验请求失败：{error}") from error

        supported_models = image_model_ids()
        matched_models = sorted(supported_models & available_models)
        if not matched_models:
            expected_models = ", ".join(sorted(supported_models))
            raise ToolProviderCredentialValidationError(
                f"/models 未返回工具支持的图像模型。需要至少一个：{expected_models}"
            )
