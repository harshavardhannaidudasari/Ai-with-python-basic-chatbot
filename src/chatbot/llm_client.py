"""Thin wrapper around the Anthropic Claude Messages API.

Handles system-prompt assembly (including RAG context injection), streaming,
and the small per-model quirks (e.g. `effort` isn't accepted on Haiku).
"""

from __future__ import annotations

from collections.abc import Iterator

import anthropic
import ollama

from .config import settings


class LLMError(RuntimeError):
    """Raised when the LLM API call fails or is misconfigured."""


_NO_EFFORT_MODELS = ("haiku", "sonnet-4-5", "sonnet-4-0")


def _supports_effort(model: str) -> bool:
    # `output_config.effort` errors on Haiku and Sonnet-4.5-and-older tiers.
    return not any(marker in model for marker in _NO_EFFORT_MODELS)


def _build_system_prompt(context_chunks: list[str] | None) -> str:
    if not context_chunks:
        return settings.system_prompt

    context_block = "\n\n---\n\n".join(context_chunks)
    return (
        f"{settings.system_prompt}\n\n"
        "Use the following retrieved context to answer the user's question "
        "when it is relevant. If the context does not contain the answer, "
        "say so and answer from your general knowledge instead.\n\n"
        f"<context>\n{context_block}\n</context>"
    )


class ClaudeClient:
    """Wraps `anthropic.Anthropic` with sane chatbot defaults."""

    def __init__(self) -> None:
        try:
            # Zero-arg client resolves ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN
            # / an `ant auth login` profile automatically.
            self.client = anthropic.Anthropic()
        except Exception as exc:  # pragma: no cover - defensive
            raise LLMError(f"Failed to initialize Anthropic client: {exc}") from exc

    def _build_kwargs(self, system: str, messages: list[dict]) -> dict:
        kwargs: dict = {
            "model": settings.model,
            "max_tokens": settings.max_tokens,
            "system": system,
            "messages": messages,
        }
        if _supports_effort(settings.model):
            kwargs["output_config"] = {"effort": settings.effort}
        return kwargs

    def build_system_prompt(self, context_chunks: list[str] | None) -> str:
        return _build_system_prompt(context_chunks)

    def stream_reply(
        self, messages: list[dict], context_chunks: list[str] | None = None
    ) -> Iterator[str]:
        """Yield text chunks as they arrive from the model."""
        system = self.build_system_prompt(context_chunks)
        kwargs = self._build_kwargs(system, messages)
        try:
            with self.client.messages.stream(**kwargs) as stream:
                yield from stream.text_stream
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Claude API error ({exc.status_code}): {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"Network error contacting Claude API: {exc}") from exc
        except Exception as exc:
            # Covers e.g. missing/invalid credentials, which the SDK raises
            # as a plain TypeError rather than an APIError subclass.
            raise LLMError(f"Could not get a response from Claude: {exc}") from exc

    def reply(self, messages: list[dict], context_chunks: list[str] | None = None) -> str:
        """Return the full response text in one call (used by the web API)."""
        system = self.build_system_prompt(context_chunks)
        kwargs = self._build_kwargs(system, messages)
        try:
            with self.client.messages.stream(**kwargs) as stream:
                final = stream.get_final_message()
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Claude API error ({exc.status_code}): {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"Network error contacting Claude API: {exc}") from exc
        except Exception as exc:
            # Covers e.g. missing/invalid credentials, which the SDK raises
            # as a plain TypeError rather than an APIError subclass.
            raise LLMError(f"Could not get a response from Claude: {exc}") from exc

        if final.stop_reason == "refusal":
            return "I'm not able to help with that request."
        return "".join(block.text for block in final.content if block.type == "text")

    def stream_reply_with_image(
        self, prompt: str, image_b64: str, media_type: str = "image/jpeg"
    ) -> Iterator[str]:
        """Yield text chunks describing/answering about a single attached image.

        Single-turn only (no conversation history, no RAG context), matching
        `OllamaClient.stream_reply_with_image`'s contract.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": image_b64},
                    },
                    {"type": "text", "text": prompt or "Describe this image."},
                ],
            }
        ]
        kwargs = self._build_kwargs(settings.system_prompt, messages)
        try:
            with self.client.messages.stream(**kwargs) as stream:
                yield from stream.text_stream
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Claude API error ({exc.status_code}): {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"Network error contacting Claude API: {exc}") from exc
        except Exception as exc:
            raise LLMError(f"Could not get a response from Claude: {exc}") from exc


class OllamaClient:
    """Wraps a local Ollama server (https://ollama.com) with the same interface as `ClaudeClient`."""

    def __init__(self) -> None:
        self.client = ollama.Client(host=settings.ollama_host)

    def build_system_prompt(self, context_chunks: list[str] | None) -> str:
        return _build_system_prompt(context_chunks)

    def _chat_messages(self, system: str, messages: list[dict]) -> list[dict]:
        return [{"role": "system", "content": system}, *messages]

    def stream_reply(
        self, messages: list[dict], context_chunks: list[str] | None = None
    ) -> Iterator[str]:
        """Yield text chunks as they arrive from the model."""
        system = self.build_system_prompt(context_chunks)
        try:
            for part in self.client.chat(
                model=settings.ollama_model,
                messages=self._chat_messages(system, messages),
                stream=True,
            ):
                yield part["message"]["content"]
        except Exception as exc:
            raise LLMError(
                f"Could not get a response from Ollama ({settings.ollama_host}): {exc}"
            ) from exc

    def reply(self, messages: list[dict], context_chunks: list[str] | None = None) -> str:
        """Return the full response text in one call (used by the web API)."""
        system = self.build_system_prompt(context_chunks)
        try:
            response = self.client.chat(
                model=settings.ollama_model,
                messages=self._chat_messages(system, messages),
                stream=False,
            )
        except Exception as exc:
            raise LLMError(
                f"Could not get a response from Ollama ({settings.ollama_host}): {exc}"
            ) from exc
        return response["message"]["content"]

    def stream_reply_with_image(
        self, prompt: str, image_b64: str, media_type: str = "image/jpeg"
    ) -> Iterator[str]:
        """Yield text chunks describing/answering about a single attached image.

        Single-turn only (no conversation history, no RAG context) — kept
        simple since vision models are used for one-off image questions.
        `media_type` is accepted for interface parity with `ClaudeClient` but
        unused here — Ollama infers the image format from the raw bytes.
        """
        message = {"role": "user", "content": prompt or "Describe this image.", "images": [image_b64]}
        try:
            for part in self.client.chat(
                model=settings.ollama_vision_model,
                messages=[message],
                stream=True,
            ):
                yield part["message"]["content"]
        except Exception as exc:
            raise LLMError(
                f"Could not get a response from Ollama vision model "
                f"'{settings.ollama_vision_model}' ({settings.ollama_host}): {exc}"
            ) from exc


def build_llm_client() -> ClaudeClient | OllamaClient:
    """Return the configured LLM client (`LLM_PROVIDER=claude|ollama`, default `claude`)."""
    provider = settings.llm_provider.strip().lower()
    if provider == "ollama":
        return OllamaClient()
    if provider == "claude":
        return ClaudeClient()
    raise LLMError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r} (expected 'claude' or 'ollama')")
