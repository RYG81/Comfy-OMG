"""
OllamaNegativePrompt — Negative Prompt Generator
Analyzes a positive prompt and generates optimal negative prompts.
"""
from __future__ import annotations
import json
import logging

from ...ollama_client import generate
from ...prompts.negative_prompt import build_system_prompt, build_user_prompt
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)


class OllamaNegativePrompt:
    """Generates optimized negative prompts based on the positive prompt."""

    CATEGORY = "Ollama-Magic-Nodes/Prompt"
    FUNCTION = "generate_negative"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("negative_prompt", "quality_negatives", "content_negatives", "anatomy_negatives")
    OUTPUT_TOOLTIPS = (
        "Complete negative prompt ready to use",
        "Quality-focused negatives only",
        "Content-focused negatives specific to the prompt",
        "Anatomy-focused negatives",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "positive_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "The positive prompt to generate negatives for",
                }),
                "target_model": (["SDXL", "SD1.5", "Flux", "Pony", "Anime"], {
                    "default": "SDXL",
                    "tooltip": "Target AI model",
                }),
                "strictness": (["light", "balanced", "strict"], {
                    "default": "balanced",
                    "tooltip": "How comprehensive the negative prompt should be",
                }),
            },
        }

    def generate_negative(self, ollama_model: dict, positive_prompt: str,
                         target_model: str = "SDXL", strictness: str = "balanced"):
        cfg = ollama_model
        system = build_system_prompt(target_model, strictness)
        user_prompt = build_user_prompt(positive_prompt)

        _log.info("[OllamaNodes] Generating negative prompt (model=%s, strictness=%s)", target_model, strictness)

        raw = generate(
            base_url=cfg["base_url"],
            model=cfg["model"],
            prompt=user_prompt,
            system=system,
            temperature=cfg.get("temperature", 0.5),
            num_ctx=cfg.get("num_ctx", 4096),
            num_predict=1024,
            seed=cfg.get("seed", -1),
            keep_alive=cfg.get("keep_alive", "5m"),
            response_format="json",
        )

        parsed = extract_json_block(raw)
        if parsed is None:
            _log.warning("[OllamaNodes] Failed to parse JSON for negative prompt")
            return (raw, "", "", "")

        return (
            parsed.get("negative_prompt", ""),
            parsed.get("quality_negatives", ""),
            parsed.get("content_negatives", ""),
            parsed.get("anatomy_negatives", ""),
        )
