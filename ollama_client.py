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

Think = Union[bool, str, None]

_session_cache: dict[str, requests.Session] = {}
_session_lock = threading.Lock()


def _url(base: str, path: str) -> str:
    return base.rstrip("/") + path


def _make_session(base_url: str) -> requests.Session:
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
            adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=8)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            _session_cache[base_url] = session
            _log.debug("[OllamaNodes] Created new session for %s", base_url)
        return _session_cache[base_url]


def close_session(base_url: str) -> None:
    with _session_lock:
        session = _session_cache.pop(base_url, None)
    if session is not None:
        session.close()


def clear_session_cache() -> None:
    with _session_lock:
        sessions = list(_session_cache.values())
        _session_cache.clear()
    for s in sessions:
        s.close()


class OllamaConnectionError(RuntimeError):
    pass


def _extract_ollama_error(resp: requests.Response) -> str:
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
    try:
        r = _make_session(base_url).get(_url(base_url, "/"), timeout=timeout)
        return r.ok
    except Exception:
        return False


def get_installed_models(base_url: str, timeout: float = 10.0) -> List[dict]:
    try:
        resp = _make_session(base_url).get(_url(base_url, "/api/tags"), timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("models", [])
    except requests.exceptions.ConnectionError:
        raise OllamaConnectionError(
            f"[OllamaNodes] Cannot connect to Ollama at {base_url}. Is the Ollama server running?"
        )
    except Exception as exc:
        raise OllamaConnectionError(f"[OllamaNodes] Failed to list models: {exc}") from exc


def get_model_names(base_url: str) -> List[str]:
    models = get_installed_models(base_url)
    return sorted(m["name"] for m in models) if models else ["(no models found)"]


def get_running_models(base_url: str, timeout: float = 10.0) -> List[dict]:
    try:
        resp = _make_session(base_url).get(_url(base_url, "/api/ps"), timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("models", [])
    except Exception as exc:
        _log.warning("[OllamaNodes] get_running_models failed: %s", exc)
        return []


def get_model_info(base_url: str, model: str, timeout: float = 10.0) -> dict:
    try:
        resp = _make_session(base_url).post(
            _url(base_url, "/api/show"), json={"name": model}, timeout=timeout
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        _log.warning("[OllamaNodes] get_model_info %r failed: %s", model, exc)
        return {"error": str(exc)}


def _stream_post(
    base_url: str, endpoint: str, payload: dict[str, Any], timeout: float, label: str,
) -> Generator[dict, None, None]:
    with _make_session(base_url).post(
        _url(base_url, endpoint), json=payload, stream=True, timeout=timeout,
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
                continue


def _build_options(
    temperature: float, top_p: float, top_k: int,
    repeat_penalty: float, num_predict: int, num_ctx: int, seed: int,
) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "temperature": temperature, "top_p": top_p, "top_k": top_k,
        "repeat_penalty": repeat_penalty, "num_predict": num_predict, "num_ctx": num_ctx,
    }
    if seed >= 0:
        opts["seed"] = seed
    return opts


def generate_stream(
    base_url: str, model: str, prompt: str, system: str = "",
    temperature: float = 0.7, top_p: float = 0.9, top_k: int = 40,
    repeat_penalty: float = 1.1, num_predict: int = 512, num_ctx: int = 8192,
    seed: int = -1, response_format: str = "", format: str = "",
    think: Think = False, keep_alive: str = "5m", timeout: float = 120.0,
    images: list[str] | None = None,
) -> Generator[str, None, None]:
    resolved_format = response_format or format
    payload: dict[str, Any] = {
        "model": model, "prompt": prompt, "stream": True, "keep_alive": keep_alive,
        "options": _build_options(temperature, top_p, top_k, repeat_penalty, num_predict, num_ctx, seed),
    }
    if system:
        payload["system"] = system
    if resolved_format == "json":
        payload["format"] = "json"
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


def generate(base_url: str, model: str, prompt: str, **kwargs: Any) -> str:
    return "".join(generate_stream(base_url, model, prompt, **kwargs))


def chat_stream(
    base_url: str, model: str, messages: list[dict], system: str = "",
    temperature: float = 0.7, top_p: float = 0.9, top_k: int = 40,
    repeat_penalty: float = 1.1, num_predict: int = 512, num_ctx: int = 8192,
    seed: int = -1, response_format: str = "", format: str = "",
    think: Think = False, keep_alive: str = "5m", timeout: float = 180.0,
) -> Generator[str, None, None]:
    resolved_format = response_format or format
    full_messages: list[dict] = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)
    payload: dict[str, Any] = {
        "model": model, "messages": full_messages, "stream": True, "keep_alive": keep_alive,
        "options": _build_options(temperature, top_p, top_k, repeat_penalty, num_predict, num_ctx, seed),
    }
    if resolved_format == "json":
        payload["format"] = "json"
    if think is not None:
        payload["think"] = think
    for chunk in _stream_post(base_url, "/api/chat", payload, timeout, "/api/chat"):
        token = chunk.get("message", {}).get("content", "")
        if token:
            yield token
        if chunk.get("done"):
            return


def chat(base_url: str, model: str, messages: list[dict], **kwargs: Any) -> str:
    return "".join(chat_stream(base_url, model, messages, **kwargs))


def embed(base_url: str, model: str, texts: list[str], timeout: float = 60.0) -> list[list[float]]:
    try:
        resp = _make_session(base_url).post(
            _url(base_url, "/api/embed"), json={"model": model, "input": texts}, timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("embeddings", [])
    except Exception as exc:
        raise OllamaConnectionError(f"[OllamaNodes] Embedding failed: {exc}") from exc


def pull_model_stream(base_url: str, model: str, timeout: float = 600.0) -> Generator[dict, None, None]:
    yield from _stream_post(base_url, "/api/pull", {"name": model, "stream": True}, timeout, "/api/pull")


def delete_model(base_url: str, model: str, timeout: float = 30.0) -> bool:
    try:
        resp = _make_session(base_url).delete(
            _url(base_url, "/api/delete"), json={"name": model}, timeout=timeout,
        )
        return resp.status_code in (200, 204)
    except Exception as exc:
        _log.warning("[OllamaNodes] delete_model %r failed: %s", model, exc)
        return False


def copy_model(base_url: str, source: str, destination: str, timeout: float = 30.0) -> bool:
    try:
        resp = _make_session(base_url).post(
            _url(base_url, "/api/copy"),
            json={"source": source, "destination": destination}, timeout=timeout,
        )
        return resp.status_code in (200, 204)
    except Exception as exc:
        _log.warning("[OllamaNodes] copy_model %r -> %r failed: %s", source, destination, exc)
        return False
