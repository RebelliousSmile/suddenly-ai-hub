"""Tests for TogetherProvider."""

import pytest
from unittest.mock import MagicMock, patch
from pipelines.evaluation.providers.together import TogetherProvider
from pipelines.evaluation.providers.base import ChatMessage, CompletionRequest


class TestTogetherProviderValidation:
    def test_validate_with_valid_key(self):
        # New format keys start with "tgp_v1"
        p = TogetherProvider(api_key="tgp_v1-long-enough-key")
        assert p.validate() is True

    def test_validate_with_sk_prefix_key(self):
        # Old format should also work (length > 10)
        p = TogetherProvider(api_key="sk-test-key-123")
        assert p.validate() is True

    def test_validate_with_short_key(self):
        p = TogetherProvider(api_key="short")
        assert p.validate() is False

    def test_validate_with_empty_key(self):
        p = TogetherProvider(api_key="")
        assert p.validate() is False


class TestTogetherProviderChat:
    @patch("httpx.Client")
    def test_chat_completion_success(self, mock_client_cls):
        provider = TogetherProvider(api_key="sk-test")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Hello from together"}}],
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        mock_client_instance = MagicMock()
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_client_instance.post = MagicMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_client_instance

        req = CompletionRequest(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[ChatMessage(role="user", content="Hi there")],
            temperature=0.7,
            max_tokens=128,
        )
        result = provider.chat_completion(req)

        assert result.content == "Hello from together"
        assert result.model == "Qwen/Qwen2.5-7B-Instruct"
        assert result.usage["prompt_tokens"] == 10
        assert result.usage["completion_tokens"] == 5

    @patch("httpx.Client")
    def test_chat_completion_without_usage(self, mock_client_cls):
        provider = TogetherProvider(api_key="sk-test")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "no usage here"}}],
            "model": "test-model",
        }
        mock_client_instance = MagicMock()
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_client_instance.post = MagicMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_client_instance

        req = CompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="test")],
        )
        result = provider.chat_completion(req)

        assert result.content == "no usage here"
        assert result.usage is None

    @patch("httpx.Client")
    def test_chat_completion_raises_on_http_error(self, mock_client_cls):
        provider = TogetherProvider(api_key="sk-test")
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("429 Too Many Requests")
        mock_client_instance = MagicMock()
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_client_instance.post = MagicMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_client_instance

        req = CompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hi")],
        )
        with pytest.raises(Exception, match="429"):
            provider.chat_completion(req)

    @patch("httpx.Client")
    def test_chat_completion_multiple_messages(self, mock_client_cls):
        provider = TogetherProvider(api_key="sk-test")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "multi turn response"}}],
            "model": "test",
        }
        mock_client_instance = MagicMock()
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_client_instance.post = MagicMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_client_instance

        req = CompletionRequest(
            model="test",
            messages=[
                ChatMessage(role="system", content="You are helpful"),
                ChatMessage(role="user", content="Hello"),
            ],
        )
        result = provider.chat_completion(req)
        assert result.content == "multi turn response"

        # Verify the payload structure
        call_args = mock_client_instance.post.call_args
        payload = call_args.kwargs["json"]
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"
