"""
OllamaPromptVariations — Batch Prompt Variations
Generates multiple creative variations of a prompt.
"""
from __future__ import annotations
import json
import logging

from ...ollama_client import generate
from ...prompts.prompt_variations import build_system_prompt, build_user_prompt
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)


class OllamaPromptVariations:
    """Generates N variations of a prompt while maintaining the core concept."""

    CATEGORY = "Ollama-Magic-Nodes/Prompt"
    FUNCTION = "generate_variations"
    RETURN_TYPES = ("STRING",) * 11
    RETURN_NAMES = (
        "variation_1", "variation_2", "variation_3", "variation_4", "variation_5",
        "variation_6", "variation_7", "variation_8", "variation_9", "variation_10",
        "all_variations",
    )
    OUTPUT_TOOLTIPS = tuple([f"Variation {i}" for i in range(1, 11)] + ["All variations newline-separated"])

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "base_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Original prompt to create variations of",
                }),
                "num_variations": ("INT", {
                    "default": 5,
                    "min": 2, "max": 10, "step": 1,
                    "tooltip": "Number of variations to generate",
                }),
                "variation_strength": (["subtle", "moderate", "wild"], {
                    "default": "moderate",
                    "tooltip": "How different the variations should be",
                }),
            },
            "optional": {
                "lock_elements": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Elements to keep unchanged (comma-separated)",
                }),
            },
        }

    def generate_variations(self, ollama_model: dict, base_prompt: str,
                           num_variations: int = 5, variation_strength: str = "moderate",
                           lock_elements: str = ""):
        cfg = ollama_model
        system = build_system_prompt(num_variations, variation_strength, lock_elements)
        user_prompt = build_user_prompt(base_prompt)

        _log.info("[OllamaNodes] Generating %d variations (strength=%s)", num_variations, variation_strength)

        raw = generate(
            base_url=cfg["base_url"],
            model=cfg["model"],
            prompt=user_prompt,
            system=system,
            temperature=cfg.get("temperature", 0.8),
            num_ctx=cfg.get("num_ctx", 8192),
            num_predict=4096,
            seed=cfg.get("seed", -1),
            keep_alive=cfg.get("keep_alive", "5m"),
            response_format="json",
        )

        parsed = extract_json_block(raw)
        if parsed is None:
            _log.warning("[OllamaNodes] Failed to parse JSON for variations")
            return (raw,) + ("",) * 10

        variations = []
        for i in range(1, 11):
            variations.append(parsed.get(f"variation_{i}", ""))

        all_vars = parsed.get("all_variations", "\n".join([v for v in variations if v]))

        return tuple(variations) + (all_vars,)
