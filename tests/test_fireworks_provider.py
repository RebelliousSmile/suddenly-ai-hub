"""Tests for FireworksProvider."""

import pytest
from unittest.mock import MagicMock, patch
from evaluation.providers.fireworks import FireworksProvider
from evaluation.providers.base import ChatMessage, CompletionRequest


class TestFireworksProviderValidation:
    def test_validate_with_valid_key(self):
        p = FireworksProvider(api_key="fw-key-12345")
        assert p.validate() is True

    def test_validate_with_empty_key(self):
        p = FireworksProvider(api_key="")
        assert p.validate() is False

    def test_validate_with_short_key(self):
        p = FireworksProvider(api_key="abc")
        assert p.validate() is False

    def test_validate_with_exactly_6_chars(self):
        p = FireworksProvider(api_key="123456")
        assert p.validate() is True

    def test_validate_with_long_key(self):
        p = FireworksProvider(api_key="a" * 50)
        assert p.validate() is True


class TestFireworksProviderChat:
    @patch("httpx.Client")
    def test_chat_completion_success(self, mock_client_cls):
        provider = FireworksProvider(api_key="fw-key-123")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Hello from fireworks"}}],
            "model": "accounts/fireworks/models/qwen2.5-7b-instruct",
            "usage": {"prompt_tokens": 8, "completion_tokens": 3},
        }
        mock_client_instance = MagicMock()
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_client_instance.post = MagicMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_client_instance

        req = CompletionRequest(
            model="accounts/fireworks/models/qwen2.5-7b-instruct",
            messages=[ChatMessage(role="user", content="Say hello")],
            temperature=0.5,
            max_tokens=64,
        )
        result = provider.chat_completion(req)

        assert result.content == "Hello from fireworks"
        assert result.model == "accounts/fireworks/models/qwen2.5-7b-instruct"
        assert result.usage["prompt_tokens"] == 8

    @patch("httpx.Client")
    def test_chat_completion_without_usage(self, mock_client_cls):
        provider = FireworksProvider(api_key="fw-key-123")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "no usage"}}],
            "model": "test",
        }
        mock_client_instance = MagicMock()
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_client_instance.post = MagicMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_client_instance

        req = CompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="test")],
        )
        result = provider.chat_completion(req)
        assert result.content == "no usage"
        assert result.usage is None

    @patch("httpx.Client")
    def test_chat_completion_raises_on_http_error(self, mock_client_cls):
        provider = FireworksProvider(api_key="fw-key-123")
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("500 Internal Server Error")
        mock_client_instance = MagicMock()
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_client_instance.post = MagicMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_client_instance

        req = CompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="Hi")],
        )
        with pytest.raises(Exception, match="500"):
            provider.chat_completion(req)

    @patch("httpx.Client")
    def test_chat_completion_payload_structure(self, mock_client_cls):
        """Verify the payload sent to Fireworks has the right structure."""
        provider = FireworksProvider(api_key="fw-key-123")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "model": "test",
        }
        mock_client_instance = MagicMock()
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_client_instance.post = MagicMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_client_instance

        req = CompletionRequest(
            model="accounts/fireworks/models/test",
            messages=[
                ChatMessage(role="system", content="be brief"),
                ChatMessage(role="user", content="test prompt"),
            ],
            temperature=0.8,
            max_tokens=50,
        )
        provider.chat_completion(req)

        call_args = mock_client_instance.post.call_args
        payload = call_args.kwargs["json"]
        headers = call_args.kwargs["headers"]

        assert payload["model"] == "accounts/fireworks/models/test"
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"
        assert payload["temperature"] == 0.8
        assert payload["max_tokens"] == 50
        assert "Bearer fw-key-123" in headers["Authorization"]
