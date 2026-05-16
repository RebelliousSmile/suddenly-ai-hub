"""Tests for evaluation.providers.base module."""

import pytest
from evaluation.providers.base import (
    BaseProvider,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
)


class TestChatMessage:
    def test_create_message(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_different_roles(self):
        for role in ("user", "assistant", "system"):
            msg = ChatMessage(role=role, content="test")
            assert msg.role == role

    def test_multiline_content(self):
        content = "line1\nline2\nline3"
        msg = ChatMessage(role="user", content=content)
        assert msg.content == content


class TestCompletionRequest:
    def test_defaults(self):
        req = CompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="hi")],
        )
        assert req.temperature == 0.7
        assert req.max_tokens == 1024
        assert req.model == "test-model"
        assert len(req.messages) == 1

    def test_custom_values(self):
        req = CompletionRequest(
            model="custom",
            messages=[ChatMessage(role="system", content="Be concise")],
            temperature=0.9,
            max_tokens=2048,
        )
        assert req.temperature == 0.9
        assert req.max_tokens == 2048
        assert req.messages[0].role == "system"

    def test_multiple_messages(self):
        msgs = [
            ChatMessage(role="system", content="You are a helper"),
            ChatMessage(role="user", content="Hi"),
        ]
        req = CompletionRequest(model="test", messages=msgs)
        assert len(req.messages) == 2


class TestCompletionResponse:
    def test_basic(self):
        resp = CompletionResponse(content="Hello world", model="test")
        assert resp.content == "Hello world"
        assert resp.model == "test"
        assert resp.usage is None

    def test_with_usage(self):
        resp = CompletionResponse(
            content="Hello",
            model="test",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )
        assert resp.usage["prompt_tokens"] == 10
        assert resp.usage["completion_tokens"] == 5

    def test_empty_content(self):
        resp = CompletionResponse(content="", model="test")
        assert resp.content == ""


class TestBaseProvider:
    def test_base_provider_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            BaseProvider()

    def test_abstract_methods_defined(self):
        """BaseProvider must define name, validate, chat_completion."""
        assert hasattr(BaseProvider, "name")
        assert hasattr(BaseProvider, "validate")
        assert hasattr(BaseProvider, "chat_completion")

    def test_is_abstract(self):
        """Attempting to call abstract methods raises TypeError."""
        with pytest.raises(TypeError):
            class Fake(BaseProvider):
                pass
            f = Fake()
            f.validate()
