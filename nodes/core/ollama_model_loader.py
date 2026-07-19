"""
ollama_model_loader.py
─────────────────────────────────────────────────────────────────────────
🦙 Ollama Model Loader

Connects to a running Ollama instance, enumerates every locally-installed
model at execution time (no hard-coded list), and outputs an OLLAMA_MODEL
pipe that the other nodes consume.

Outputs
  OLLAMA_MODEL  – dict with {model, base_url, keep_alive, …options}
  STRING        – model name for display / routing
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import time
from typing import Any

from .ollama_client import (
    get_model_names,
    get_model_info,
    get_running_models,
    check_ollama_alive,
    OllamaConnectionError,
)

# Default Ollama base URL
_DEFAULT_URL = "http://localhost:11434"


class OllamaModelLoader:
    """
    Fetches the live list of installed Ollama models and produces an
    OLLAMA_MODEL connection object for downstream nodes.
    """

    CATEGORY = "Ollama-Magic-Nodes/Core"
    FUNCTION = "load_model"
    RETURN_TYPES = ("OLLAMA_MODEL", "STRING")
    RETURN_NAMES = ("ollama_model", "model_name")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        # Attempt to list models; fall back to a placeholder if Ollama is offline
        try:
            model_list = get_model_names(_DEFAULT_URL)
        except OllamaConnectionError:
            model_list = ["(Ollama offline – start server)"]

        return {
            "required": {
                "ollama_url": (
                    "STRING",
                    {
                        "default":     _DEFAULT_URL,
                        "multiline":   False,
                        "description": "Ollama server base URL",
                    },
                ),
                "model": (
                    model_list,
                    {
                        "description": "Installed model (auto-loaded from Ollama)",
                    },
                ),
                "keep_alive": (
                    ["5m", "10m", "30m", "1h", "0"],
                    {
                        "default":     "0",
                        "description": "How long to keep model in VRAM after last use (0 = once)",
                    },
                ),
            },
            "optional": {
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01,
                     "description": "Sampling temperature (0 = deterministic, 2 = very random)"},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "top_k": (
                    "INT",
                    {"default": 40, "min": 1, "max": 200, "step": 1},
                ),
                "repeat_penalty": (
                    "FLOAT",
                    {"default": 1.1, "min": 0.5, "max": 2.0, "step": 0.01},
                ),
                "seed": (
                    "INT",
                    {"default": -1, "min": -1, "max": 2**31 - 1, "step": 1,
                     "description": "Random seed (-1 = random each run)"},
                ),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        relevant = {
            "ollama_url": kwargs.get("ollama_url", _DEFAULT_URL),
            "model": kwargs.get("model", ""),
            "keep_alive": kwargs.get("keep_alive", "5m"),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "top_k": kwargs.get("top_k", 40),
            "repeat_penalty": kwargs.get("repeat_penalty", 1.1),
            "seed": kwargs.get("seed", -1),
        }
        return tuple(sorted(relevant.items()))

    def load_model(
        self,
        ollama_url:     str,
        model:          str,
        keep_alive:     str = "5m",
        temperature:    float = 0.7,
        top_p:          float = 0.9,
        top_k:          int = 40,
        repeat_penalty: float = 1.1,
        seed:           int = -1,
    ):
        base_url = ollama_url.rstrip("/")

        # ── Connectivity check ────────────────────────────────────
        if not check_ollama_alive(base_url):
            raise ConnectionError(
                f"[OllamaModelLoader] Cannot reach Ollama at {base_url}.\n"
                "  • Make sure Ollama is running  (`ollama serve`)\n"
                "  • Check the URL in the node settings"
            )

        # ── Verify model is actually installed ───────────────────
        installed = get_model_names(base_url)
        if model not in installed and model != "(Ollama offline – start server)":
            print(
                f"[OllamaModelLoader] ⚠  Model '{model}' not found in installed list. "
                f"Proceeding anyway – Ollama may still load it."
            )

        # ── Fetch metadata ────────────────────────────────────────
        info = get_model_info(base_url, model)
        running = get_running_models(base_url)
        is_loaded = any(m.get("name") == model for m in running)

        details = info.get("details", {})
        param_size = details.get("parameter_size", "unknown")
        quant = details.get("quantization_level", "unknown")
        family = details.get("family", "unknown")

        print(
            f"[OllamaModelLoader] ✅ Model      : {model}\n"
            f"                    Family     : {family}\n"
            f"                    Params     : {param_size}\n"
            f"                    Quant      : {quant}\n"
            f"                    In VRAM    : {'yes' if is_loaded else 'no (will load on first use)'}\n"
            f"                    Server     : {base_url}"
        )

        ollama_model: dict[str, Any] = {
            "model":          model,
            "base_url":       base_url,
            "keep_alive":     keep_alive,
            "temperature":    temperature,
            "top_p":          top_p,
            "top_k":          top_k,
            "repeat_penalty": repeat_penalty,
            "seed":           seed,
            # metadata (read-only)
            "_info": {
                "family":     family,
                "param_size": param_size,
                "quant":      quant,
                "is_loaded":  is_loaded,
            },
        }

        return (ollama_model, model)
