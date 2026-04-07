"""Unit tests for ghost.ollama_client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost.ollama_client import OllamaClient


class TestOllamaClient:
    def test_chat_builds_messages_and_returns_content(self, monkeypatch) -> None:
        fake_response = {"message": {"content": "hello"}}
        fake_ollama = MagicMock()
        fake_ollama.chat.return_value = fake_response
        monkeypatch.setattr("ghost.ollama_client.ollama", fake_ollama)

        client = OllamaClient(model="llama3")
        result = client.chat("hi", system="system")

        assert result == "hello"
        fake_ollama.chat.assert_called_once()

    def test_chat_returns_error_string_on_exception(self, monkeypatch) -> None:
        fake_ollama = MagicMock()
        fake_ollama.chat.side_effect = RuntimeError("offline")
        monkeypatch.setattr("ghost.ollama_client.ollama", fake_ollama)

        client = OllamaClient()

        assert client.chat("hi") == "Error: offline"

    @pytest.mark.asyncio
    async def test_chat_async_uses_sync_chat(self, monkeypatch) -> None:
        client = OllamaClient()
        chat_mock = MagicMock(return_value="async-response")
        monkeypatch.setattr(client, "chat", chat_mock)

        result = await client.chat_async("hello")

        assert result == "async-response"
        chat_mock.assert_called_once_with("hello", None)

    @pytest.mark.asyncio
    async def test_get_recommendation_parses_json_payload(self, monkeypatch) -> None:
        client = OllamaClient()
        monkeypatch.setattr(
            client,
            "chat_async",
            AsyncMock(
                return_value='{"architecture": "mlp", "batch_size": 16, "tips": ["go"]}'
            ),
        )

        result = await client.get_recommendation("train classifier", "mnist")

        assert result["status"] == "success"
        assert result["recommendations"]["architecture"] == "mlp"

    @pytest.mark.asyncio
    async def test_get_recommendation_falls_back_to_text_on_bad_json(
        self, monkeypatch
    ) -> None:
        client = OllamaClient()
        monkeypatch.setattr(
            client,
            "chat_async",
            AsyncMock(return_value="plain english recommendation"),
        )

        result = await client.get_recommendation("train classifier")

        assert result["status"] == "success"
        assert result["recommendations"] == {
            "recommendation": "plain english recommendation"
        }

    @pytest.mark.asyncio
    async def test_get_recommendation_returns_error_on_chat_failure(
        self, monkeypatch
    ) -> None:
        client = OllamaClient()
        monkeypatch.setattr(
            client,
            "chat_async",
            AsyncMock(side_effect=RuntimeError("timeout")),
        )

        result = await client.get_recommendation("train classifier")

        assert result == {"status": "error", "message": "timeout"}

    @pytest.mark.asyncio
    async def test_analyze_training_progress_parses_json(self, monkeypatch) -> None:
        client = OllamaClient()
        monkeypatch.setattr(
            client,
            "chat_async",
            AsyncMock(
                return_value='{"status": "good", "analysis": "stable", "suggestions": ["continue"]}'
            ),
        )

        result = await client.analyze_training_progress(
            [
                {"step": 1, "loss": 0.2, "accuracy": 0.9},
            ]
        )

        assert result["status"] == "success"
        assert result["analysis"]["status"] == "good"

    @pytest.mark.asyncio
    async def test_analyze_training_progress_falls_back_to_text(
        self, monkeypatch
    ) -> None:
        client = OllamaClient()
        monkeypatch.setattr(
            client,
            "chat_async",
            AsyncMock(return_value="training is stable"),
        )

        result = await client.analyze_training_progress([{"step": 1, "loss": 0.5}])

        assert result["status"] == "success"
        assert result["analysis"] == {"analysis": "training is stable"}

    @pytest.mark.asyncio
    async def test_analyze_training_progress_returns_error_on_failure(
        self, monkeypatch
    ) -> None:
        client = OllamaClient()
        monkeypatch.setattr(
            client,
            "chat_async",
            AsyncMock(side_effect=RuntimeError("unreachable")),
        )

        result = await client.analyze_training_progress([{"step": 1, "loss": 0.5}])

        assert result == {"status": "error", "message": "unreachable"}

    def test_generate_training_code_delegates_to_chat(self, monkeypatch) -> None:
        client = OllamaClient()
        chat_mock = MagicMock(return_value="print('train')")
        monkeypatch.setattr(client, "chat", chat_mock)

        result = client.generate_training_code(
            "image classification", framework="tensorflow"
        )

        assert result == "print('train')"
        chat_mock.assert_called_once()

    def test_check_connection_returns_true_when_ollama_is_available(
        self, monkeypatch
    ) -> None:
        fake_ollama = MagicMock()
        fake_ollama.ps.return_value = {}
        monkeypatch.setattr("ghost.ollama_client.ollama", fake_ollama)

        client = OllamaClient()

        assert client.check_connection() is True

    def test_check_connection_returns_false_on_exception(self, monkeypatch) -> None:
        fake_ollama = MagicMock()
        fake_ollama.ps.side_effect = RuntimeError("offline")
        monkeypatch.setattr("ghost.ollama_client.ollama", fake_ollama)

        client = OllamaClient()

        assert client.check_connection() is False
