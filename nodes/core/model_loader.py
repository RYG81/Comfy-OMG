"""
OllamaModelLoader — Connect to an Ollama server and select a model.
This node is the entry point that other nodes connect to.
"""
from __future__ import annotations
import logging
from .ollama_client import get_model_names, check_ollama_alive

_log = logging.getLogger(__name__)


class OllamaModelLoader:
    """Connects to an Ollama server and exposes the selected model for downstream nodes."""

    CATEGORY = "Ollama-Magic-Nodes/Core"
    FUNCTION = "load_model"
    RETURN_TYPES = ("OLLAMA_MODEL",)
    RETURN_NAMES = ("ollama_model",)
    OUTPUT_TOOLTIPS = ("Ollama model connection object — connect to any Ollama-powered node",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_url": ("STRING", {
                    "default": "http://127.0.0.1:11434",
                    "tooltip": "Ollama server URL",
                }),
                "model": ((), {
                    "tooltip": "Select the Ollama model to use",
                }),
                "keep_alive": ("STRING", {
                    "default": "5m",
                    "tooltip": "How long to keep model in memory (e.g., '5m', '1h', '0' for unload immediately)",
                }),
                "num_ctx": ("INT", {
                    "default": 8192,
                    "min": 512,
                    "max": 131072,
                    "step": 512,
                    "tooltip": "Context window size (tokens). Larger = more memory, better for long outputs",
                }),
            },
            "optional": {
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Sampling temperature (0=deterministic, 2=very random)",
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1, "max": 2**31,
                    "tooltip": "Random seed (-1 for random)",
                }),
            },
        }

    def load_model(self, base_url: str, model: str, keep_alive: str = "5m",
                    num_ctx: int = 8192, temperature: float = 0.7, seed: int = -1):
        alive = check_ollama_alive(base_url)
        if not alive:
            _log.warning("[OllamaNodes] Server at %s not responding", base_url)

        model_config = {
            "base_url": base_url,
            "model": model,
            "keep_alive": keep_alive,
            "num_ctx": num_ctx,
            "temperature": temperature,
            "seed": seed,
        }
        _log.info("[OllamaNodes] Model loaded: %s @ %s (ctx=%d)", model, base_url, num_ctx)
        return (model_config,)
