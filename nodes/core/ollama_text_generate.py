"""
ollama_text_generate.py
─────────────────────────────────────────────────────────────────────────
🦙 Ollama Text Generate

Raw prompt → completion using /api/generate.
Supports streaming (tokens printed to console in real-time),
system prompts, JSON-mode, and per-node option overrides.

Inputs
  ollama_model  – OLLAMA_MODEL pipe from OllamaModelLoader
  prompt        – STRING
  system_prompt – STRING  (optional)
  …options…     – override loader defaults

Outputs
  STRING  – generated text
  STRING  – raw prompt (pass-through, useful for chaining)
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import sys

from .ollama_client import generate_stream, OllamaConnectionError


class OllamaTextGenerate:

    CATEGORY = "Ollama-Magic-Nodes/Core"
    FUNCTION      = "generate"
    RETURN_TYPES  = ("STRING", "STRING")
    RETURN_NAMES  = ("generated_text", "prompt_passthrough")
    OUTPUT_NODE   = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL", {}),
                "prompt": (
                    "STRING",
                    {
                        "multiline":   True,
                        "default":     "Write a short poem about neural networks.",
                        "description": "The user prompt sent to the model",
                    },
                ),
            },
            "optional": {
                "system_prompt": (
                    "STRING",
                    {
                        "multiline":   True,
                        "default":     "",
                        "description": "System / persona instructions (leave blank to skip)",
                    },
                ),
                "num_predict": (
                    "INT",
                    {
                        "default": 512, "min": 1, "max": 8192, "step": 1,
                        "description": "Max tokens to generate",
                    },
                ),
                "temperature_override": (
                    "FLOAT",
                    {"default": -1.0, "min": -1.0, "max": 2.0, "step": 0.01,
                     "description": "Override model temperature (-1 = use loader value)"},
                ),
                "output_format": (
                    ["text", "json"],
                    {"default": "text"},
                ),
                "stream_to_console": (
                    "BOOLEAN",
                    {"default": True,
                     "description": "Print tokens to console as they arrive"},
                ),
                "prefix_text": (
                    "STRING",
                    {"multiline": False, "default": "",
                     "description": "Optional text prepended to the final output"},
                ),
                "suffix_text": (
                    "STRING",
                    {"multiline": False, "default": "",
                     "description": "Optional text appended to the final output"},
                ),
                "think_mode": (
                    ["off", "on", "low", "medium", "high"],
                    {"default": "off",
                     "description": "Use Ollama thinking. Some models support only boolean or only levels."},
                ),
            },
        }

    def generate(
        self,
        ollama_model:        dict,
        prompt:              str,
        system_prompt:       str   = "",
        num_predict:         int   = 512,
        temperature_override: float = -1.0,
        output_format:       str   = "text",
        stream_to_console:   bool  = True,
        prefix_text:         str   = "",
        suffix_text:         str   = "",
        think_mode:          str   = "off",
    ):
        base_url   = ollama_model["base_url"]
        model      = ollama_model["model"]
        keep_alive = ollama_model.get("keep_alive", "5m")

        temperature = (
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

        if stream_to_console:
            print(f"\n[OllamaTextGenerate] ▶ {model} generating…\n", flush=True)

        tokens: list[str] = []
        try:
            for token in generate_stream(
                base_url       = base_url,
                model          = model,
                prompt         = prompt,
                system         = system_prompt,
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
                tokens.append(token)
                if stream_to_console:
                    sys.stdout.write(token)
                    sys.stdout.flush()

        except OllamaConnectionError as exc:
            raise RuntimeError(str(exc)) from exc

        if stream_to_console:
            print("\n[OllamaTextGenerate] ✅ Done.", flush=True)

        result = prefix_text + "".join(tokens) + suffix_text
        return (result, prompt)
