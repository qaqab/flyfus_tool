from __future__ import annotations

from dataclasses import dataclass

import pytest
import requests

from tools.image._image_utils import (
    ModelListRequestError,
    fetch_openai_model_ids,
    normalize_openai_base_url,
)


@dataclass(frozen=True, slots=True)
class FakeResponse:
    status_code: int
    payload: dict[str, list[dict[str, str]]]

    def json(self) -> dict[str, list[dict[str, str]]]:
        return self.payload


def test_fetch_openai_model_ids_uses_the_openai_models_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, str | bool | tuple[float, float]] = {}

    def fake_get(
        url: str,
        *,
        headers: dict[str, str],
        timeout: tuple[float, float],
        allow_redirects: bool,
    ) -> FakeResponse:
        observed["url"] = url
        observed["authorization"] = headers["Authorization"]
        observed["accept"] = headers["Accept"]
        observed["timeout"] = timeout
        observed["allow_redirects"] = allow_redirects
        return FakeResponse(
            status_code=200,
            payload={"data": [{"id": "gpt-image-2"}, {"id": "gpt-5.4"}]},
        )

    monkeypatch.setattr("tools.image._image_utils.requests.get", fake_get)

    model_ids = fetch_openai_model_ids("https://litellm.flyfus.com", "test-api-key")

    assert model_ids == {"gpt-image-2", "gpt-5.4"}
    assert observed == {
        "url": "https://litellm.flyfus.com/v1/models",
        "authorization": "Bearer test-api-key",
        "accept": "application/json",
        "timeout": (10.0, 30.0),
        "allow_redirects": False,
    }


@pytest.mark.parametrize(
    ("raw_url", "normalized_url"),
    [
        ("https://litellm.flyfus.com", "https://litellm.flyfus.com/v1"),
        ("https://litellm.flyfus.com/v1/", "https://litellm.flyfus.com/v1"),
        ("https://gateway.example.com/openai", "https://gateway.example.com/openai/v1"),
    ],
)
def test_normalize_openai_base_url_keeps_an_https_api_base(
    raw_url: str,
    normalized_url: str,
) -> None:
    assert normalize_openai_base_url(raw_url) == normalized_url


@pytest.mark.parametrize(
    "raw_url",
    [
        "http://litellm.flyfus.com",
        "https://@litellm.flyfus.com",
        "https://user:password@litellm.flyfus.com",
        "https://litellm.flyfus.com?access_token=should-not-be-here",
        "https://litellm.flyfus.com#fragment",
    ],
)
def test_normalize_openai_base_url_rejects_unsafe_endpoint_forms(raw_url: str) -> None:
    with pytest.raises(ValueError):
        normalize_openai_base_url(raw_url)


def test_fetch_openai_model_ids_redacts_api_key_from_http_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(
        url: str,
        *,
        headers: dict[str, str],
        timeout: tuple[float, float],
        allow_redirects: bool,
    ) -> FakeResponse:
        return FakeResponse(status_code=401, payload={"error": []})

    monkeypatch.setattr("tools.image._image_utils.requests.get", fake_get)

    with pytest.raises(ModelListRequestError) as raised:
        fetch_openai_model_ids("https://litellm.flyfus.com", "test-api-key")

    assert raised.value.status_code == 401
    assert "test-api-key" not in str(raised.value)


def test_fetch_openai_model_ids_drops_the_original_header_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(
        url: str,
        *,
        headers: dict[str, str],
        timeout: tuple[float, float],
        allow_redirects: bool,
    ) -> FakeResponse:
        raise requests.exceptions.InvalidHeader("Invalid header value Bearer test-api-key")

    monkeypatch.setattr("tools.image._image_utils.requests.get", fake_get)

    with pytest.raises(ModelListRequestError) as raised:
        fetch_openai_model_ids("https://litellm.flyfus.com", "test-api-key")

    assert raised.value.__cause__ is None
    assert "test-api-key" not in str(raised.value)
