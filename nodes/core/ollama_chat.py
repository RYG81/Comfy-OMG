"""
ollama_chat.py
─────────────────────────────────────────────────────────────────────────
🦙 Ollama Chat

Stateful multi-turn chat via /api/chat.
Maintains a conversation history list (CHAT_HISTORY type) that is passed
between nodes so you can build branching dialogue graphs in ComfyUI.

Inputs
  ollama_model   – OLLAMA_MODEL pipe
  user_message   – STRING  (current user turn)
  chat_history   – CHAT_HISTORY  (optional; omit for a fresh conversation)
  system_prompt  – STRING  (applied once at the head of history)

Outputs
  STRING        – assistant reply
  CHAT_HISTORY  – updated history (pass to the next Chat node)
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import copy
import sys
import threading

from .ollama_client import chat_stream, OllamaConnectionError

# Type alias registered with ComfyUI (arbitrary string names are fine)
CHAT_HISTORY_TYPE = "CHAT_HISTORY"


class OllamaChat:
    _history_store: dict[str, list] = {}
    _history_lock = threading.Lock()

    CATEGORY = "Ollama-Magic-Nodes/Core"
    FUNCTION      = "chat"
    RETURN_TYPES  = ("STRING", CHAT_HISTORY_TYPE)
    RETURN_NAMES  = ("assistant_reply", "chat_history")
    OUTPUT_NODE   = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL", {}),
                "user_message": (
                    "STRING",
                    {
                        "multiline":   True,
                        "default":     "Hello! What can you help me with?",
                        "description": "The user's latest message",
                    },
                ),
            },
            "optional": {
                "conversation_id": (
                    "STRING",
                    {
                        "default": "default",
                        "description": "Memory slot for automatic chat history",
                    },
                ),
                "auto_use_history": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "description": "Automatically continue saved history for this conversation_id",
                    },
                ),
                "chat_history": (
                    CHAT_HISTORY_TYPE,
                    {"description": "Optional manual history override"},
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default":   (
                            "You are a helpful, concise assistant. "
                            "Respond clearly and accurately."
                        ),
                        "description": "System persona (only applied when starting a new conversation)",
                    },
                ),
                "num_predict": (
                    "INT",
                    {"default": 1024, "min": 1, "max": 8192, "step": 1},
                ),
                "temperature_override": (
                    "FLOAT",
                    {"default": -1.0, "min": -1.0, "max": 2.0, "step": 0.01},
                ),
                "output_format": (
                    ["text", "json"],
                    {"default": "text"},
                ),
                "stream_to_console": (
                    "BOOLEAN",
                    {"default": True},
                ),
                "max_history_turns": (
                    "INT",
                    {
                        "default": 20, "min": 1, "max": 200, "step": 1,
                        "description": "Truncate history to the last N message pairs (saves context)",
                    },
                ),
                "reset_history": (
                    "BOOLEAN",
                    {"default": False,
                     "description": "Clear saved/manual history and start fresh"},
                ),
                "think_mode": (
                    ["off", "on", "low", "medium", "high"],
                    {"default": "off"},
                ),
            },
        }

    # ── Helpers ───────────────────────────────────────────────────

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Chat is intentionally stateful; always execute when queued.
        return float("nan")
    @staticmethod
    def _trim_history(history: list, max_turns: int) -> list:
        """Keep only the last *max_turns* assistant/user pairs."""
        if not history:
            return []
        # Separate system message(s) from conversational turns
        system_msgs  = [m for m in history if m["role"] == "system"]
        convo_msgs   = [m for m in history if m["role"] != "system"]
        # Each "turn" = 1 user + 1 assistant message → 2 items
        max_items = max_turns * 2
        if len(convo_msgs) > max_items:
            convo_msgs = convo_msgs[-max_items:]
        return system_msgs + convo_msgs

    @staticmethod
    def _normalize_history(history) -> list:
        if not isinstance(history, list):
            return []
        normalized = []
        for message in history:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content")
            if role in {"system", "user", "assistant"} and isinstance(content, str):
                normalized.append({"role": role, "content": content})
        return normalized

    @staticmethod
    def _store_key(base_url: str, model: str, conversation_id: str) -> str:
        conversation = (conversation_id or "default").strip() or "default"
        return f"{base_url.rstrip('/')}|{model}|{conversation}"

    # ── Main execution ────────────────────────────────────────────

    def chat(
        self,
        ollama_model:         dict,
        user_message:         str,
        conversation_id:      str         = "default",
        auto_use_history:     bool        = True,
        chat_history:         list | None = None,
        system_prompt:        str         = "",
        num_predict:          int         = 1024,
        temperature_override: float       = -1.0,
        output_format:        str         = "text",
        stream_to_console:    bool        = True,
        max_history_turns:    int         = 20,
        reset_history:        bool        = False,
        think_mode:           str         = "off",
    ):
        base_url   = ollama_model["base_url"]
        model      = ollama_model["model"]
        keep_alive = ollama_model.get("keep_alive", "5m")

        temperature    = (
            temperature_override
            if temperature_override >= 0
            else ollama_model.get("temperature", 0.7)
        )
        top_p          = ollama_model.get("top_p",          0.9)
        top_k          = ollama_model.get("top_k",          40)
        repeat_penalty = ollama_model.get("repeat_penalty", 1.1)
        seed           = ollama_model.get("seed",           -1)
        fmt            = "json" if output_format == "json" else ""
        think = False if think_mode == "off" else (True if think_mode == "on" else think_mode)
        history_key = self._store_key(base_url, model, conversation_id)

        # ── Build / restore history ───────────────────────────────
        if reset_history:
            with self._history_lock:
                self._history_store.pop(history_key, None)
            history = []
            if system_prompt.strip():
                history.append({"role": "system", "content": system_prompt.strip()})
        elif chat_history is not None:
            history = self._normalize_history(copy.deepcopy(chat_history))
        elif auto_use_history:
            with self._history_lock:
                history = copy.deepcopy(self._history_store.get(history_key, []))
            if not history and system_prompt.strip():
                history.append({"role": "system", "content": system_prompt.strip()})
        else:
            history: list = []
            if system_prompt.strip():
                history.append({"role": "system", "content": system_prompt.strip()})

        # Trim old turns to keep context window manageable
        history = self._trim_history(history, max_history_turns)

        # Append current user turn
        history.append({"role": "user", "content": user_message})

        # The client helper separates system messages automatically,
        # so pass the full history (minus embedded system msgs) as messages
        non_system = [m for m in history if m["role"] != "system"]
        system_str = next(
            (m["content"] for m in history if m["role"] == "system"), ""
        )

        if stream_to_console:
            print(f"\n[OllamaChat] ▶ {model} responding…\n", flush=True)

        reply_tokens: list[str] = []
        try:
            for token in chat_stream(
                base_url       = base_url,
                model          = model,
                messages       = non_system,
                system         = system_str,
                temperature    = temperature,
                top_p          = top_p,
                top_k          = top_k,
                repeat_penalty = repeat_penalty,
                num_predict    = num_predict,
                seed           = seed,
                format         = fmt,
                think          = think,
                keep_alive     = keep_alive,
            ):
                reply_tokens.append(token)
                if stream_to_console:
                    sys.stdout.write(token)
                    sys.stdout.flush()

        except OllamaConnectionError as exc:
            raise RuntimeError(str(exc)) from exc

        if stream_to_console:
            print("\n[OllamaChat] ✅ Done.", flush=True)

        assistant_reply = "".join(reply_tokens)

        # Append assistant reply to history
        history.append({"role": "assistant", "content": assistant_reply})
        history = self._trim_history(history, max_history_turns)

        if auto_use_history:
            with self._history_lock:
                self._history_store[history_key] = copy.deepcopy(history)

        return (assistant_reply, history)
