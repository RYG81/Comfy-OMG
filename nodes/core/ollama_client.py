"""
ollama_client.py
─────────────────────────────────────────────────────────────────────────
Shared Ollama HTTP client used by every node in the pack.

Features
• Singleton connection pool  (one requests.Session per base_url)
• Dynamic model listing      → GET  /api/tags
• Streaming generation       → POST /api/generate  (stream=True)
• Streaming chat             → POST /api/chat       (stream=True)
• Non-streaming variants for embeddings / structured output
• Retry logic with exponential back-off
• Friendly error messages surfaced to ComfyUI console
• Session lifecycle management (close / evict)
• Support for num_ctx parameter to control context window size
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Generator, List, Optional, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ── Logger ─────────────────────────────────────────────────────────
_log = logging.getLogger(__name__)

# ── Public API ─────────────────────────────────────────────────────
__all__ = [
    "OllamaConnectionError",
    "check_ollama_alive",
    "get_installed_models",
    "get_model_names",
    "get_running_models",
    "get_model_info",
    "generate_stream",
    "generate",
    "chat_stream",
    "chat",
    "embed",
    "pull_model_stream",
    "delete_model",
    "copy_model",
    "close_session",
    "clear_session_cache",
]

# Type alias for the `think` parameter
Think = Union[bool, str, None]  # False | True | "low"/"medium"/"high" | None
FormatType = Union[str, dict[str, Any], None]

# ── Module-level session cache (one per base_url) ──────────────────
_session_cache: dict[str, requests.Session] = {}
_session_lock = threading.Lock()


def _url(base: str, path: str) -> str:
    """Build a full URL from base and a path, stripping any trailing slash."""
    return base.rstrip("/") + path


def _make_session(base_url: str) -> requests.Session:
    """Return a cached (or freshly-created) requests.Session for base_url.

    Uses double-checked locking so concurrent callers never race to
    create duplicate sessions.
    """
    if base_url in _session_cache:
        return _session_cache[base_url]

    with _session_lock:
        if base_url not in _session_cache:
            session = requests.Session()
            retry = Retry(
                total=3,
                backoff_factor=0.5,
                status_forcelist=[500, 502, 503, 504],
                allowed_methods=["GET", "POST", "DELETE"],
            )
            adapter = HTTPAdapter(
                max_retries=retry,
                pool_connections=4,
                pool_maxsize=8,
            )
            session.mount("http://",  adapter)
            session.mount("https://", adapter)
            _session_cache[base_url] = session
            _log.debug("[OllamaNodes] Created new session for %s", base_url)
        return _session_cache[base_url]


def close_session(base_url: str) -> None:
    """Close and evict the cached session for *base_url*."""
    with _session_lock:
        session = _session_cache.pop(base_url, None)
    if session is not None:
        session.close()
        _log.debug("[OllamaNodes] Closed session for %s", base_url)


def clear_session_cache() -> None:
    """Close *all* cached sessions (call on ComfyUI shutdown)."""
    with _session_lock:
        sessions = list(_session_cache.values())
        _session_cache.clear()
    for s in sessions:
        s.close()
    _log.debug("[OllamaNodes] All sessions closed.")


# ── Public helpers ─────────────────────────────────────────────────

class OllamaConnectionError(RuntimeError):
    """Raised when Ollama cannot be reached or returns an unexpected error."""


def _extract_ollama_error(resp: requests.Response) -> str:
    """Best-effort extraction of Ollama error message from an HTTP response."""
    try:
        data = resp.json()
        if isinstance(data, dict):
            err = data.get("error")
            if err:
                return str(err)
    except Exception:
        pass
    txt = (resp.text or "").strip()
    return txt if txt else f"HTTP {resp.status_code}"


def check_ollama_alive(base_url: str, timeout: float = 5.0) -> bool:
    """Return True if Ollama responds on *base_url*."""
    try:
        r = _make_session(base_url).get(_url(base_url, "/"), timeout=timeout)
        return r.ok
    except Exception:
        return False


def get_installed_models(base_url: str, timeout: float = 10.0) -> List[dict]:
    """
    Fetch the list of locally installed Ollama models.

    Returns a list of dicts with at least:
        name, size, modified_at, details (family, parameter_size, quantization_level)
    """
    try:
        resp = _make_session(base_url).get(
            _url(base_url, "/api/tags"), timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("models", [])
    except requests.exceptions.ConnectionError:
        raise OllamaConnectionError(
            f"[OllamaNodes] Cannot connect to Ollama at {base_url}. "
            "Is the Ollama server running?"
        )
    except Exception as exc:
        raise OllamaConnectionError(
            f"[OllamaNodes] Failed to list models: {exc}"
        ) from exc


def get_model_names(base_url: str) -> List[str]:
    """Return a sorted list of installed model name strings."""
    models = get_installed_models(base_url)
    return sorted(m["name"] for m in models) if models else ["(no models found)"]


def get_running_models(base_url: str, timeout: float = 10.0) -> List[dict]:
    """Return currently-loaded models (GET /api/ps)."""
    try:
        resp = _make_session(base_url).get(
            _url(base_url, "/api/ps"), timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("models", [])
    except Exception as exc:
        _log.warning("[OllamaNodes] get_running_models failed: %s", exc)
        return []


def get_model_info(base_url: str, model: str, timeout: float = 10.0) -> dict:
    """Fetch detailed model metadata (POST /api/show)."""
    try:
        resp = _make_session(base_url).post(
            _url(base_url, "/api/show"),
            json={"name": model},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        _log.warning("[OllamaNodes] get_model_info %r failed: %s", model, exc)
        return {"error": str(exc)}


# ── Internal streaming helper ──────────────────────────────────────

def _stream_post(
    base_url: str,
    endpoint: str,
    payload: dict[str, Any],
    timeout: float,
    label: str,
) -> Generator[dict, None, None]:
    """
    POST *payload* to *endpoint* and yield each parsed NDJSON chunk.

    Handles:
    • HTTP error extraction with friendly messages
    • Per-line JSON decode errors (skips corrupt chunks gracefully)
    """
    with _make_session(base_url).post(
        _url(base_url, endpoint),
        json=payload,
        stream=True,
        timeout=timeout,
    ) as resp:
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            msg = _extract_ollama_error(resp)
            raise OllamaConnectionError(
                f"[OllamaNodes] {label} failed ({resp.status_code}): {msg}"
            ) from exc

        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            try:
                yield json.loads(raw_line)
            except json.JSONDecodeError:
                _log.debug(
                    "[OllamaNodes] %s — skipping non-JSON line: %r",
                    label,
                    raw_line[:120],
                )
                continue


def _build_options(
    temperature: float,
    top_p: float,
    top_k: int,
    repeat_penalty: float,
    num_predict: int,
    num_ctx: int,          # 👈 ADDED: context window size
    seed: int,
) -> dict[str, Any]:
    """Build the shared `options` sub-dict for generate / chat payloads."""
    opts: dict[str, Any] = {
        "temperature":    temperature,
        "top_p":          top_p,
        "top_k":          top_k,
        "repeat_penalty": repeat_penalty,
        "num_predict":    num_predict,
        "num_ctx":        num_ctx,   # 👈 ADDED: forward context window
    }
    if seed >= 0:
        opts["seed"] = seed
    return opts


def _normalize_format(response_format: FormatType, format_: FormatType) -> FormatType:
    """Prefer response_format, then format; accepts 'json' or JSON-schema objects."""
    resolved = response_format if response_format not in ("", None) else format_
    if resolved in ("", None):
        return None
    if isinstance(resolved, str):
        return "json" if resolved.strip().lower() == "json" else None
    if isinstance(resolved, dict):
        return resolved
    return None


# ── Streaming generate ─────────────────────────────────────────────

def generate_stream(
    base_url: str,
    model: str,
    prompt: str,
    system: str = "",
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    repeat_penalty: float = 1.1,
    num_predict: int = 512,
    num_ctx: int = 8192,     # 👈 ADDED: context window control
    seed: int = -1,
    response_format: FormatType = "",
    format: FormatType = "",
    think: Think = False,
    keep_alive: str = "5m",
    timeout: float = 120.0,
    images: list[str] | None = None,
) -> Generator[str, None, None]:
    """
    Stream text generation tokens from /api/generate.
    Yields each token string as it arrives.
    """
    resolved_format = _normalize_format(response_format, format)

    payload: dict[str, Any] = {
        "model":      model,
        "prompt":     prompt,
        "stream":     True,
        "keep_alive": keep_alive,
        "options":    _build_options(temperature, top_p, top_k, repeat_penalty, num_predict, num_ctx, seed),
    }
    if system:
        payload["system"] = system
    if resolved_format is not None:
        payload["format"] = resolved_format
    if think is not None:
        payload["think"] = think
    if images:
        payload["images"] = images

    saw_response = False
    thinking_fallback: list[str] = []

    for chunk in _stream_post(base_url, "/api/generate", payload, timeout, "/api/generate"):
        token = chunk.get("response", "")
        if token:
            saw_response = True
            yield token
        elif not saw_response:
            tkn = chunk.get("thinking", "")
            if tkn:
                thinking_fallback.append(tkn)
        if chunk.get("done"):
            if not saw_response and thinking_fallback:
                yield "".join(thinking_fallback)
            return


def generate(
    base_url: str,
    model: str,
    prompt: str,
    **kwargs: Any,
) -> str:
    """Non-streaming wrapper around generate_stream. Returns full text."""
    return "".join(generate_stream(base_url, model, prompt, **kwargs))


# ── Streaming chat ─────────────────────────────────────────────────

def chat_stream(
    base_url: str,
    model: str,
    messages: list[dict],
    system: str = "",
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    repeat_penalty: float = 1.1,
    num_predict: int = 512,
    num_ctx: int = 8192,     # 👈 ADDED: context window control
    seed: int = -1,
    response_format: FormatType = "",
    format: FormatType = "",
    think: Think = False,
    keep_alive: str = "5m",
    timeout: float = 180.0,
) -> Generator[str, None, None]:
    """Stream chat completion tokens from /api/chat."""
    resolved_format = _normalize_format(response_format, format)

    full_messages: list[dict] = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    payload: dict[str, Any] = {
        "model":      model,
        "messages":   full_messages,
        "stream":     True,
        "keep_alive": keep_alive,
        "options":    _build_options(temperature, top_p, top_k, repeat_penalty, num_predict, num_ctx, seed),
    }
    if resolved_format is not None:
        payload["format"] = resolved_format
    if think is not None:
        payload["think"] = think

    saw_response = False
    thinking_fallback: list[str] = []

    for chunk in _stream_post(base_url, "/api/chat", payload, timeout, "/api/chat"):
        token = chunk.get("message", {}).get("content", "")
        if token:
            saw_response = True
            yield token
        elif not saw_response:
            tkn = chunk.get("message", {}).get("thinking", "")
            if tkn:
                thinking_fallback.append(tkn)
        if chunk.get("done"):
            if not saw_response and thinking_fallback:
                yield "".join(thinking_fallback)
            return


def chat(
    base_url: str,
    model: str,
    messages: list[dict],
    **kwargs: Any,
) -> str:
    """Non-streaming chat. Returns full assistant reply."""
    return "".join(chat_stream(base_url, model, messages, **kwargs))


# ── Embeddings ─────────────────────────────────────────────────────

def embed(
    base_url: str,
    model: str,
    texts: list[str],
    timeout: float = 60.0,
) -> list[list[float]]:
    """
    Generate embeddings via POST /api/embed.
    Returns a list of float vectors (one per input text).
    """
    try:
        resp = _make_session(base_url).post(
            _url(base_url, "/api/embed"),
            json={"model": model, "input": texts},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("embeddings", [])
    except Exception as exc:
        raise OllamaConnectionError(
            f"[OllamaNodes] Embedding failed: {exc}"
        ) from exc


# ── Model management ───────────────────────────────────────────────

def pull_model_stream(
    base_url: str,
    model: str,
    timeout: float = 600.0,
) -> Generator[dict, None, None]:
    """Stream pull progress events from POST /api/pull."""
    payload = {"name": model, "stream": True}
    yield from _stream_post(base_url, "/api/pull", payload, timeout, "/api/pull")


def delete_model(base_url: str, model: str, timeout: float = 30.0) -> bool:
    """Delete a model via DELETE /api/delete. Returns True on success."""
    try:
        resp = _make_session(base_url).delete(
            _url(base_url, "/api/delete"),
            json={"name": model},
            timeout=timeout,
        )
        return resp.status_code in (200, 204)
    except requests.exceptions.HTTPError as exc:
        _log.warning(
            "[OllamaNodes] delete_model %r: HTTP %s",
            model, exc.response.status_code if exc.response else "?",
        )
        return False
    except Exception as exc:
        _log.warning("[OllamaNodes] delete_model %r failed: %s", model, exc)
        return False


def copy_model(
    base_url: str,
    source: str,
    destination: str,
    timeout: float = 30.0,
) -> bool:
    """Copy / alias a model via POST /api/copy."""
    try:
        resp = _make_session(base_url).post(
            _url(base_url, "/api/copy"),
            json={"source": source, "destination": destination},
            timeout=timeout,
        )
        return resp.status_code in (200, 204)
    except Exception as exc:
        _log.warning(
            "[OllamaNodes] copy_model %r -> %r failed: %s",
            source, destination, exc,
        )
        return False
