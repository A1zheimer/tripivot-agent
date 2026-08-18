from __future__ import annotations

import http.client
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any, Protocol

from .models import BackendResult, Language, TranslationRequest


class TranslationBackendError(RuntimeError):
    """Raised when a backend call fails after retries or returns a bad payload."""


_RETRYABLE_HTTP = {429, 500, 502, 503, 504}


class TranslationBackend(Protocol):
    def translate(self, request: TranslationRequest) -> BackendResult: ...


class OpenAICompatibleBackend:
    """Backend for vLLM, OpenAI, or any compatible chat-completions server."""

    revision_capable = True
    supports_glossary = True

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: int = 180,
        temperature: float = 0.1,
        max_retries: int = 2,
    ) -> None:
        self.model = model
        self.base_url = (base_url or os.getenv("TRANSLATION_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("TRANSLATION_API_KEY", "")
        self.timeout = timeout
        self.temperature = temperature
        self.max_retries = max_retries
        if not self.base_url:
            raise ValueError("base_url or TRANSLATION_BASE_URL is required")

    def translate(self, request: TranslationRequest) -> BackendResult:
        endpoint = (
            self.base_url
            if self.base_url.endswith("/chat/completions")
            else f"{self.base_url}/chat/completions"
        )
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": max(1024, 6 * len(request.text)),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a professional document translator. Preserve meaning, formatting, "
                        "numbers, named entities, and cross-paragraph coherence. Use every "
                        "applicable required glossary translation exactly. Return only the "
                        "translation of the final \"Text to translate\" section, with no "
                        "commentary and no code fences."
                    ),
                },
                {"role": "user", "content": _translation_prompt(request)},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload, duration_ms = self._post_with_retry(endpoint, body, headers)
        choices = payload.get("choices") or []
        if not choices:
            raise TranslationBackendError(
                f"backend {self.model} returned no choices (usage={payload.get('usage')})"
            )
        choice = choices[0]
        finish_reason = choice.get("finish_reason")
        content = (choice.get("message") or {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise TranslationBackendError(f"backend {self.model} returned empty content")
        if finish_reason == "length":
            raise TranslationBackendError(
                f"backend {self.model} truncated the translation (finish_reason=length); "
                "increase the chunk size budget or model context"
            )
        return BackendResult(
            text=content.strip(),
            metadata={
                "backend": "openai-compatible",
                "model": self.model,
                "usage": payload.get("usage", {}),
                "finish_reason": finish_reason,
                "duration_ms": duration_ms,
            },
        )

    def _post_with_retry(
        self,
        endpoint: str,
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> tuple[dict[str, Any], int]:
        encoded = json.dumps(body, ensure_ascii=False).encode()
        started = time.monotonic()
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                endpoint, data=encoded, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.load(response)
                return payload, round((time.monotonic() - started) * 1000)
            except urllib.error.HTTPError as error:
                detail = error.read().decode(errors="replace")
                last_error = error
                if error.code in _RETRYABLE_HTTP and attempt < self.max_retries:
                    time.sleep(min(8.0, 2.0**attempt))
                    continue
                raise TranslationBackendError(
                    f"translation backend returned HTTP {error.code}: {detail}"
                ) from error
            except (
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
                OSError,
                http.client.HTTPException,
                json.JSONDecodeError,
            ) as error:
                # Transient transport failures (resets, incomplete reads,
                # gateway HTML with HTTP 200) are retried; the rest is wrapped
                # into TranslationBackendError with request context.
                last_error = error
                if attempt < self.max_retries:
                    time.sleep(min(8.0, 2.0**attempt))
                    continue
                raise TranslationBackendError(
                    f"translation backend unreachable/invalid after {attempt + 1} "
                    f"attempt(s): {type(error).__name__}: {error}"
                ) from error
        raise TranslationBackendError(f"translation backend failed: {last_error}")


class NllbBackend:
    """Optional local baseline; install the ``nllb`` project extra."""

    # Greedy seq2seq decoding cannot follow glossary or revision instructions,
    # so the agent skips the revision loop for this backend.
    revision_capable = False
    supports_glossary = False

    LANGUAGE_CODES = {
        Language.ENGLISH: "eng_Latn",
        Language.CHINESE: "zho_Hans",
        Language.BURMESE: "mya_Mymr",
    }

    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M") -> None:
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("install the project with the [nllb] extra") from error
        self.model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def translate(self, request: TranslationRequest) -> BackendResult:
        source_code = self.LANGUAGE_CODES[request.direction.source]
        target_code = self.LANGUAGE_CODES[request.direction.target]
        self._tokenizer.src_lang = source_code
        encoded = self._tokenizer(request.text, return_tensors="pt", truncation=True)
        generated = self._model.generate(
            **encoded,
            forced_bos_token_id=self._tokenizer.convert_tokens_to_ids(target_code),
            max_new_tokens=max(64, int(encoded["input_ids"].shape[-1] * 2.5)),
        )
        text = self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        return BackendResult(
            text=text.strip(),
            metadata={"backend": "nllb", "model": self.model_name},
        )


class MappingBackend:
    """Deterministic backend for smoke tests and integration development."""

    # Deterministic: re-issuing a failed request returns the same output, so
    # revision rounds are pointless. Glossary terms are still applied.
    revision_capable = False
    supports_glossary = True

    def __init__(self, translations: Mapping[str, str]) -> None:
        self.translations = translations

    def translate(self, request: TranslationRequest) -> BackendResult:
        text = self.translations.get(request.text, request.text)
        for entry in request.glossary:
            source = entry.term(request.direction.source)
            target = entry.term(request.direction.target)
            if source and target:
                text = text.replace(source, target)
        return BackendResult(text=text, confidence=1.0, metadata={"backend": "mapping"})


def _translation_prompt(request: TranslationRequest) -> str:
    direction = request.direction
    lines = [
        f"Translate from {direction.source.label} ({direction.source.value}) "
        f"to {direction.target.label} ({direction.target.value}).",
        f"Domain: {request.style.domain}",
        f"Register: {request.style.register}",
        f"Audience: {request.style.audience}",
    ]
    if request.style.instructions:
        rules = "\n".join(f"- {item}" for item in request.style.instructions)
        lines.append(f"Style rules:\n{rules}")
    if request.glossary:
        pairs = []
        for entry in request.glossary:
            source = entry.term(direction.source)
            target = entry.term(direction.target)
            if source and target:
                pairs.append(f"- {source} => {target}")
        if pairs:
            lines.append("Required terminology:\n" + "\n".join(pairs))
    if request.previous_context:
        context = "\n".join(
            f"SOURCE: {source}\nTRANSLATION: {translation}"
            for source, translation in request.previous_context
        )
        lines.append(
            "Previous context (for coherence only; do not retranslate and never copy "
            f"it into the output):\n{context}"
        )
    if request.previous_translation:
        lines.append(
            "Previous draft (fix only the listed problems, keep the rest):\n"
            f"{request.previous_translation}"
        )
    if request.revision_notes:
        lines.append(
            "Revision requirements:\n" + "\n".join(f"- {note}" for note in request.revision_notes)
        )
    lines.append(f"Text to translate:\n{request.text}")
    return "\n\n".join(lines)
