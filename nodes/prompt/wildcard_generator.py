"""
OllamaWildcardGenerator — Generate prompts with randomizable variations.
"""
from __future__ import annotations
import json
import logging

from ...ollama_client import generate
from ...prompts.wildcard_generator import (
    VARIATION_STYLE, CATEGORIES_TO_VARY,
    build_system_prompt, build_user_prompt,
)
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)


class OllamaWildcardGenerator:
    """Generate prompts with {option1|option2|option3} wildcard syntax."""

    CATEGORY = "Ollama-Magic-Nodes/Prompt"
    FUNCTION = "generate_wildcards"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "wildcard_prompt", "static_elements", "varied_elements",
        "total_combinations", "sample_1", "sample_2", "sample_3",
    )
    OUTPUT_TOOLTIPS = (
        "Prompt with {wildcard|syntax}",
        "Elements that stay constant",
        "What was made into wildcards",
        "Number of possible combinations",
        "Sample resolved prompt 1",
        "Sample resolved prompt 2",
        "Sample resolved prompt 3",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "base_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Base prompt to add variations to",
                }),
                "variation_style": (VARIATION_STYLE, {
                    "default": "moderate",
                    "tooltip": "How different the variations should be",
                }),
                "categories": (CATEGORIES_TO_VARY, {
                    "default": "all",
                    "tooltip": "What to vary",
                }),
                "num_options": ("INT", {
                    "default": 3,
                    "min": 2, "max": 8, "step": 1,
                    "tooltip": "Number of options per wildcard",
                }),
            },
        }

    def generate_wildcards(self, ollama_model: dict, base_prompt: str,
                          variation_style: str = "moderate",
                          categories: str = "all", num_options: int = 3):
        cfg = ollama_model
        system = build_system_prompt(variation_style, categories, num_options)
        user_prompt = build_user_prompt(base_prompt)

        _log.info("[OllamaNodes] Generating wildcards (style=%s, options=%d)", variation_style, num_options)

        raw = generate(
            base_url=cfg["base_url"],
            model=cfg["model"],
            prompt=user_prompt,
            system=system,
            temperature=cfg.get("temperature", 0.8),
            num_ctx=cfg.get("num_ctx", 8192),
            num_predict=2500,
            seed=cfg.get("seed", -1),
            keep_alive=cfg.get("keep_alive", "5m"),
            response_format="json",
        )

        parsed = extract_json_block(raw)
        if parsed is None:
            return (raw,) + ("",) * 6

        samples = parsed.get("sample_outputs", ["", "", ""])
        while len(samples) < 3:
            samples.append("")

        return (
            parsed.get("wildcard_prompt", ""),
            parsed.get("static_elements", ""),
            parsed.get("varied_elements", ""),
            parsed.get("total_combinations", ""),
            samples[0],
            samples[1],
            samples[2],
        )
