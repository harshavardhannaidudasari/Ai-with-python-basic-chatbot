"""Thin wrapper around the Anthropic Claude Messages API.

Handles system-prompt assembly (including RAG context injection), streaming,
and the small per-model quirks (e.g. `effort` isn't accepted on Haiku).
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
import threading
import time
import wave
from collections.abc import Iterator

import anthropic
import groq
import ollama
from google import genai
from google.genai import types as genai_types
from google.genai import errors as genai_errors
from sarvamai import SarvamAI

from .config import settings


class LLMError(RuntimeError):
    """Raised when the LLM API call fails or is misconfigured."""


_NO_EFFORT_MODELS = ("haiku", "sonnet-4-5", "sonnet-4-0")


def _supports_effort(model: str) -> bool:
    # `output_config.effort` errors on Haiku and Sonnet-4.5-and-older tiers.
    return not any(marker in model for marker in _NO_EFFORT_MODELS)


def _groq_reasoning_kwargs(model: str) -> dict:
    """Suppress chain-of-thought on Groq's reasoning-capable models.

    Without this, a reasoning model (e.g. qwen/qwen3.6-27b) streams a raw
    `<think>...</think>` block as ordinary content: it leaks internal
    reasoning into the chat UI, and on a bounded max_tokens budget can
    consume the whole budget before the actual answer is ever generated
    (observed live: image analysis returning nothing but an unfinished
    chain of thought, cut off mid-sentence).
    `reasoning_format="hidden"` alone isn't enough — the model still spends
    tokens reasoning, just doesn't show it, so it can still exhaust the
    budget with no visible answer left. `reasoning_effort="none"` actually
    turns reasoning off for the qwen3 family, so no budget is spent on it
    at all. gpt-oss doesn't support disabling reasoning, only hiding it,
    so it uses `include_reasoning` instead — see
    https://console.groq.com/docs/reasoning.
    """
    if "qwen" in model:
        return {"reasoning_effort": "none"}
    if "gpt-oss" in model:
        return {"include_reasoning": False}
    return {}


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

    @staticmethod
    def _options() -> dict | None:
        # keep_alive/num_predict are set here rather than left at Ollama's
        # defaults: on slow/CPU-only hardware, an idle model gets unloaded
        # between messages (paying a full reload on the next one), and an
        # uncapped reply can run for tens of seconds longer than needed.
        return {"num_predict": settings.ollama_num_predict} if settings.ollama_num_predict > 0 else None

    def _unload(self, model: str) -> None:
        # Best-effort eviction of a model we're *not* about to use. Keeping
        # the text model warm (see OLLAMA_KEEP_ALIVE) speeds up back-to-back
        # text messages, but on machines without enough RAM for both the text
        # and vision models at once, Ollama's own automatic eviction can fail
        # outright (observed: an out-of-memory crash trying to load the
        # vision model while the text model was still resident) rather than
        # just evicting the old one. Explicitly unloading the model we're
        # switching away from avoids that collision.
        if settings.ollama_model == settings.ollama_vision_model:
            return  # same model serves both roles — nothing to evict
        try:
            self.client.generate(model=model, keep_alive=0)
        except Exception:
            pass

    def stream_reply(
        self, messages: list[dict], context_chunks: list[str] | None = None
    ) -> Iterator[str]:
        """Yield text chunks as they arrive from the model."""
        system = self.build_system_prompt(context_chunks)
        self._unload(settings.ollama_vision_model)
        try:
            for part in self.client.chat(
                model=settings.ollama_model,
                messages=self._chat_messages(system, messages),
                stream=True,
                keep_alive=settings.ollama_keep_alive,
                options=self._options(),
            ):
                yield part["message"]["content"]
        except Exception as exc:
            raise LLMError(
                f"Could not get a response from Ollama ({settings.ollama_host}): {exc}"
            ) from exc

    def reply(self, messages: list[dict], context_chunks: list[str] | None = None) -> str:
        """Return the full response text in one call (used by the web API)."""
        system = self.build_system_prompt(context_chunks)
        self._unload(settings.ollama_vision_model)
        try:
            response = self.client.chat(
                model=settings.ollama_model,
                messages=self._chat_messages(system, messages),
                stream=False,
                keep_alive=settings.ollama_keep_alive,
                options=self._options(),
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
        self._unload(settings.ollama_model)
        try:
            for part in self.client.chat(
                model=settings.ollama_vision_model,
                messages=[message],
                stream=True,
                keep_alive=settings.ollama_keep_alive,
                options=self._options(),
            ):
                yield part["message"]["content"]
        except Exception as exc:
            raise LLMError(
                f"Could not get a response from Ollama vision model "
                f"'{settings.ollama_vision_model}' ({settings.ollama_host}): {exc}"
            ) from exc


class GroqClient:
    """Wraps `groq.Groq` (OpenAI-compatible) with the same interface as `ClaudeClient`.

    Free-tier alternative to Claude for deployments where an Anthropic API
    key isn't an option (e.g. no billing set up).
    """

    def __init__(self) -> None:
        try:
            # Zero-arg client resolves GROQ_API_KEY from the environment.
            self.client = groq.Groq()
        except Exception as exc:  # pragma: no cover - defensive
            raise LLMError(f"Failed to initialize Groq client: {exc}") from exc

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
            stream = self.client.chat.completions.create(
                model=settings.groq_model,
                messages=self._chat_messages(system, messages),
                max_tokens=settings.max_tokens,
                stream=True,
                **_groq_reasoning_kwargs(settings.groq_model),
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except groq.APIStatusError as exc:
            raise LLMError(f"Groq API error ({exc.status_code}): {exc.message}") from exc
        except groq.APIConnectionError as exc:
            raise LLMError(f"Network error contacting Groq API: {exc}") from exc
        except Exception as exc:
            raise LLMError(f"Could not get a response from Groq: {exc}") from exc

    def reply(self, messages: list[dict], context_chunks: list[str] | None = None) -> str:
        """Return the full response text in one call (used by the web API)."""
        system = self.build_system_prompt(context_chunks)
        try:
            completion = self.client.chat.completions.create(
                model=settings.groq_model,
                messages=self._chat_messages(system, messages),
                max_tokens=settings.max_tokens,
                stream=False,
                **_groq_reasoning_kwargs(settings.groq_model),
            )
        except groq.APIStatusError as exc:
            raise LLMError(f"Groq API error ({exc.status_code}): {exc.message}") from exc
        except groq.APIConnectionError as exc:
            raise LLMError(f"Network error contacting Groq API: {exc}") from exc
        except Exception as exc:
            raise LLMError(f"Could not get a response from Groq: {exc}") from exc
        return completion.choices[0].message.content or ""

    def stream_reply_with_image(
        self, prompt: str, image_b64: str, media_type: str = "image/jpeg"
    ) -> Iterator[str]:
        """Yield text chunks describing/answering about a single attached image.

        Single-turn only (no conversation history, no RAG context), matching
        `ClaudeClient.stream_reply_with_image`'s contract. Uses a separate
        vision-capable model since not every Groq-hosted model accepts images.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt or "Describe this image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{image_b64}"},
                    },
                ],
            }
        ]
        try:
            stream = self.client.chat.completions.create(
                model=settings.groq_vision_model,
                messages=messages,
                max_tokens=settings.max_tokens,
                stream=True,
                **_groq_reasoning_kwargs(settings.groq_vision_model),
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except groq.APIStatusError as exc:
            raise LLMError(f"Groq API error ({exc.status_code}): {exc.message}") from exc
        except groq.APIConnectionError as exc:
            raise LLMError(f"Network error contacting Groq API: {exc}") from exc
        except Exception as exc:
            raise LLMError(f"Could not get a response from Groq: {exc}") from exc


def build_llm_client() -> ClaudeClient | OllamaClient | GroqClient:
    """Return the configured LLM client (`LLM_PROVIDER=claude|ollama|groq`, default `claude`)."""
    provider = settings.llm_provider.strip().lower()
    if provider == "ollama":
        return OllamaClient()
    if provider == "claude":
        return ClaudeClient()
    if provider == "groq":
        return GroqClient()
    raise LLMError(
        f"Unknown LLM_PROVIDER: {settings.llm_provider!r} (expected 'claude', 'ollama', or 'groq')"
    )


_tts_client: groq.Groq | None = None


def synthesize_speech(text: str) -> bytes:
    """Return WAV audio bytes for `text`, spoken by voice mode's chosen voice.

    Tries each configured/enabled engine in turn — see _TTS_STAGES (bottom
    of this file, defined after the engine functions it references) for the
    default order, or settings.tts_provider to pin a specific one first
    regardless of that order (e.g. "sarvam" for Indian-language replies even
    with GEMINI_API_KEY also set). Each stage falls through to the next on
    any failure, including a stage's dependency simply not being installed
    (e.g. Kokoro enabled but requirements-kokoro.txt not installed) or
    configured (e.g. GEMINI_API_KEY left unset). If every configured stage
    fails — or none is configured at all — the caller (web_app's
    /api/speak) surfaces the error and the frontend falls back further, to
    the browser's own voice.
    """
    stages = list(_TTS_STAGES)
    if settings.tts_provider:
        # Stable sort: the pinned provider moves to front, everything else
        # keeps its default relative order.
        stages.sort(key=lambda stage: stage[0] != settings.tts_provider)

    last_exc: Exception | None = None
    for name, flag_attr, fn in stages:
        if not getattr(settings, flag_attr):
            continue
        try:
            return fn(text)
        except Exception as exc:
            print(f"[tts:{name}] falling back: {exc}")
            last_exc = exc

    if last_exc is not None:
        raise last_exc if isinstance(last_exc, LLMError) else LLMError(str(last_exc))
    raise LLMError(
        "No text-to-speech engine is configured — set GEMINI_API_KEY, "
        "SARVAM_API_KEY, GROQ_API_KEY, KOKORO_TTS_ENABLED=true, or "
        "VOICE_CLONE_REFERENCE in .env."
    )


_gemini_tts_client: genai.Client | None = None


def synthesize_speech_gemini(text: str) -> bytes:
    """Return WAV audio bytes for `text`, spoken by Gemini's TTS voice.

    Free via a Google AI Studio API key (aistudio.google.com/apikey), and
    preferred over Groq's Orpheus (see synthesize_speech()) since its free
    tier is far less likely to be exhausted by normal use. The API itself
    returns raw 24kHz/16-bit/mono PCM, not a WAV file, so this wraps it in a
    WAV header before returning — the rest of the app (web_app's
    /api/speak, the frontend's <audio> playback) expects WAV either way.
    """
    global _gemini_tts_client
    if _gemini_tts_client is None:
        try:
            # Zero-arg client resolves GEMINI_API_KEY from the environment.
            _gemini_tts_client = genai.Client()
        except Exception as exc:  # pragma: no cover - defensive
            raise LLMError(f"Failed to initialize Gemini TTS client: {exc}") from exc

    # Mirrors synthesize_speech_groq's retry: a transient rate/server error
    # plausibly clears within a short backoff; anything else (bad key,
    # decommissioned model) won't, so only one retry is attempted.
    for attempt in range(2):
        try:
            response = _gemini_tts_client.models.generate_content(
                model=settings.gemini_tts_model,
                contents=text,
                config=genai_types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=genai_types.SpeechConfig(
                        voice_config=genai_types.VoiceConfig(
                            prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                                voice_name=settings.gemini_tts_voice,
                            )
                        )
                    ),
                ),
            )
            break
        except genai_errors.APIError as exc:
            retryable = exc.code == 429 or exc.code >= 500
            if retryable and attempt == 0:
                time.sleep(0.6)
                continue
            raise LLMError(f"Gemini TTS error ({exc.code}): {exc.message}") from exc
        except Exception as exc:
            raise LLMError(f"Could not synthesize speech via Gemini: {exc}") from exc

    candidates = response.candidates or []
    if not candidates or not candidates[0].content.parts:
        raise LLMError("Gemini TTS returned no audio.")
    pcm = candidates[0].content.parts[0].inline_data.data

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit samples
        wf.setframerate(24000)
        wf.writeframes(pcm)
    return buffer.getvalue()


def synthesize_speech_groq(text: str) -> bytes:
    """Return WAV audio bytes for `text`, spoken by Groq's Orpheus TTS voice.

    Independent of LLM_PROVIDER: voice mode's spoken replies use this
    whenever GROQ_API_KEY is set, even when chat itself is served by Claude
    or Ollama, since Orpheus is a real, natural-sounding voice — unlike any
    browser's built-in TTS, which is what this falls back to on failure.
    """
    global _tts_client
    if _tts_client is None:
        try:
            # Zero-arg client resolves GROQ_API_KEY from the environment.
            _tts_client = groq.Groq()
        except Exception as exc:  # pragma: no cover - defensive
            raise LLMError(f"Failed to initialize Groq TTS client: {exc}") from exc

    # One sentence = one request, and a longer reply fires several of these
    # in quick succession — enough to trip Groq's per-minute rate limit or
    # hit an occasional transient blip mid-conversation. Each such failure
    # used to fall straight back to the browser's robotic voice for that one
    # sentence, so a reply would audibly flip between the human Orpheus
    # voice and the robotic fallback. Retrying once after a short backoff
    # absorbs those transient cases so the human voice stays consistent;
    # only a genuinely persistent failure still falls back.
    for attempt in range(2):
        try:
            response = _tts_client.audio.speech.create(
                model=settings.groq_tts_model,
                voice=settings.groq_tts_voice,
                input=text,
                # Orpheus only accepts "wav" — other Groq TTS models accept
                # more formats (flac/mp3/mulaw/ogg/wav), but this app is
                # pinned to Orpheus specifically, so keep it simple rather
                # than probing.
                response_format="wav",
            )
            break
        except groq.APIStatusError as exc:
            # A per-minute/request-rate 429, or a transient 5xx, plausibly
            # clears within the 0.6s backoff below. A per-day quota 429
            # (e.g. "tokens per day (TPD)") won't — Groq's free tier resets
            # on the order of minutes to hours, so retrying immediately just
            # adds a guaranteed-useless 0.6s of latency before falling back
            # to the browser voice anyway. Skip straight to the fallback in
            # that case instead of stalling every sentence for nothing.
            quota_exhausted = "per day" in (exc.message or "").lower()
            retryable = not quota_exhausted and (exc.status_code == 429 or exc.status_code >= 500)
            if retryable and attempt == 0:
                time.sleep(0.6)
                continue
            raise LLMError(f"Groq TTS error ({exc.status_code}): {exc.message}") from exc
        except groq.APIConnectionError as exc:
            if attempt == 0:
                time.sleep(0.6)
                continue
            raise LLMError(f"Network error contacting Groq TTS: {exc}") from exc
        except Exception as exc:
            raise LLMError(f"Could not synthesize speech via Groq: {exc}") from exc

    # The SDK's response object only exposes write_to_file(path), not raw
    # bytes directly, so round-trip through a temp file.
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        response.write_to_file(tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        os.remove(tmp_path)


_clone_tts_client = None
_clone_tts_lock = threading.Lock()


def synthesize_speech_clone(text: str) -> bytes:
    """Return WAV audio bytes for `text`, spoken in a cloned reference voice.

    Zero-shot voice cloning via Coqui XTTS-v2, running entirely locally from
    settings.voice_clone_reference (a short sample of the target voice) —
    no API, no per-request cost, unlike Groq/ElevenLabs/etc. The tradeoff is
    resource cost instead of money: ~2GB of model weights plus a heavy
    ML dependency chain (torch/transformers/coqui-tts — see
    requirements-voice-clone.txt, not part of the default install), and
    CPU inference that takes several seconds per sentence rather than
    Groq's near-instant API response. Not viable on Render's free tier.
    """
    global _clone_tts_client
    if _clone_tts_client is None:
        # XTTS-v2 is released under Coqui's non-commercial Public Model
        # License; loading it prompts for interactive agreement unless this
        # is set. Setting it here (rather than requiring it in the
        # environment) keeps the feature a one-line opt-in via
        # VOICE_CLONE_REFERENCE alone.
        os.environ.setdefault("COQUI_TOS_AGREED", "1")
        try:
            import torchaudio
            from TTS.api import TTS as CoquiTTS
        except ImportError as exc:
            raise LLMError(
                "Local voice cloning isn't installed — run "
                "`pip install -r requirements-voice-clone.txt` first."
            ) from exc

        # torchaudio>=2.9 routes torchaudio.load() through TorchCodec, which
        # needs FFmpeg's actual shared libraries installed on the system (not
        # just the torchcodec Python package) — absent here, and not
        # something to install unilaterally (a native binary from outside
        # the Python package index). XTTS internally calls torchaudio.load()
        # to read the reference voice sample, so patch it to use `soundfile`
        # instead: same PCM decoding, already a hard dependency of
        # coqui-tts, verified to read both wav and mp3 without FFmpeg.
        def _load_audio_via_soundfile(path, *args, **kwargs):
            import soundfile as sf
            import torch

            data, sr = sf.read(path, dtype="float32", always_2d=True)
            return torch.from_numpy(data.T), sr

        torchaudio.load = _load_audio_via_soundfile

        try:
            _clone_tts_client = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2")
        except Exception as exc:
            raise LLMError(f"Failed to load the voice-clone model: {exc}") from exc

    try:
        # Coqui's TTS object isn't documented as thread-safe for concurrent
        # .tts() calls on one instance; this app is single-user, but the
        # lock keeps back-to-back sentence requests from racing.
        with _clone_tts_lock:
            samples = _clone_tts_client.tts(
                text=text,
                speaker_wav=settings.voice_clone_reference,
                language=settings.voice_clone_language,
            )
    except Exception as exc:
        raise LLMError(f"Voice cloning failed: {exc}") from exc

    import io

    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, samples, samplerate=24000, format="WAV")
    return buffer.getvalue()


_sarvam_client: SarvamAI | None = None


def synthesize_speech_sarvam(text: str) -> bytes:
    """Return WAV audio bytes for `text`, spoken by Sarvam AI's TTS voice.

    Sarvam is an Indian AI platform — free credits on signup, no card
    required. Its edge over Gemini/Groq isn't English quality, it's native
    Hindi/Tamil/Telugu/etc. and Hinglish code-switching support, which
    neither of the others handle natively. Get a key at
    https://dashboard.sarvam.ai
    """
    global _sarvam_client
    if _sarvam_client is None:
        try:
            _sarvam_client = SarvamAI(api_subscription_key=settings.sarvam_api_key)
        except Exception as exc:  # pragma: no cover - defensive
            raise LLMError(f"Failed to initialize Sarvam TTS client: {exc}") from exc

    # Mirrors synthesize_speech_groq's retry: a transient rate/server error
    # plausibly clears within a short backoff; anything else won't.
    for attempt in range(2):
        try:
            response = _sarvam_client.text_to_speech.convert(
                text=text,
                language_code=settings.sarvam_tts_language,
                model=settings.sarvam_tts_model,
                speaker=settings.sarvam_tts_speaker,
            )
            break
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            retryable = status == 429 or (isinstance(status, int) and status >= 500)
            if retryable and attempt == 0:
                time.sleep(0.6)
                continue
            raise LLMError(f"Could not synthesize speech via Sarvam: {exc}") from exc

    if not response.audios:
        raise LLMError("Sarvam TTS returned no audio.")
    # Sarvam's audio chunks are already complete WAV files, base64-encoded —
    # unlike Gemini's raw PCM, no header wrapping needed.
    return base64.b64decode("".join(response.audios))


_kokoro_pipeline = None
_kokoro_lock = threading.Lock()


def synthesize_speech_kokoro(text: str) -> bytes:
    """Return WAV audio bytes for `text`, spoken by the local Kokoro-82M model.

    Kokoro is a small (82M parameter), Apache-2.0 open-weight TTS model —
    free and unlimited since it runs entirely locally: no API, no
    per-request cost, no daily quota to exhaust like Groq/Gemini/Sarvam's
    free tiers. The tradeoff is the same shape as voice cloning above: a
    torch/transformers dependency chain (see requirements-kokoro.txt, not
    part of the default install) and CPU inference — much lighter than
    XTTS-v2's ~2GB, but still not viable on Render's free tier.
    """
    global _kokoro_pipeline
    if _kokoro_pipeline is None:
        try:
            from kokoro import KPipeline
        except ImportError as exc:
            raise LLMError(
                "Kokoro TTS isn't installed — run "
                "`pip install -r requirements-kokoro.txt` first."
            ) from exc
        try:
            _kokoro_pipeline = KPipeline(lang_code=settings.kokoro_tts_lang)
        except Exception as exc:
            raise LLMError(f"Failed to load the Kokoro TTS model: {exc}") from exc

    try:
        # Not documented as thread-safe for concurrent calls; this app is
        # single-user, but the lock keeps back-to-back sentence requests
        # from racing (mirrors synthesize_speech_clone's _clone_tts_lock).
        with _kokoro_lock:
            generator = _kokoro_pipeline(text, voice=settings.kokoro_tts_voice, speed=1)
            # A short sentence is normally one segment, but the pipeline can
            # still split a long one internally — concatenate every
            # segment's audio so nothing gets dropped.
            segments = [audio for _graphemes, _phonemes, audio in generator]
    except Exception as exc:
        raise LLMError(f"Kokoro TTS failed: {exc}") from exc

    if not segments:
        raise LLMError("Kokoro TTS returned no audio.")

    import numpy as np
    import soundfile as sf

    samples = segments[0] if len(segments) == 1 else np.concatenate(segments)
    buffer = io.BytesIO()
    sf.write(buffer, samples, samplerate=24000, format="WAV")
    return buffer.getvalue()


# Default priority for synthesize_speech() when settings.tts_provider
# doesn't pin one explicitly: the local clone (a specific chosen voice, when
# configured) first, then Gemini (currently the best general-purpose free
# voice — Groq's Orpheus free tier caps out at just 3,600 tokens/day,
# confirmed exhausted live on 2026-09-02), then Sarvam (best choice
# specifically for Indian-language replies), then local Kokoro (free/
# unlimited but needs its own install), then Groq last. Defined here, after
# the engine functions above, so it can reference them directly.
_TTS_STAGES: list[tuple[str, str, object]] = [
    ("clone", "voice_clone_reference", synthesize_speech_clone),
    ("gemini", "gemini_api_key", synthesize_speech_gemini),
    ("sarvam", "sarvam_api_key", synthesize_speech_sarvam),
    ("kokoro", "kokoro_tts_enabled", synthesize_speech_kokoro),
    ("groq", "groq_api_key", synthesize_speech_groq),
]
