"""
OllamaPromptEnhancer — Prompt Enhancer
Takes a simple prompt and enhances it with rich details.
"""
from __future__ import annotations
import json
import logging

from ...ollama_client import generate
from ...prompts.prompt_enhance import SYSTEM_PROMPT_PRESETS, build_system_prompt, build_user_prompt
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)


class OllamaPromptEnhancer:
    """Enhances simple prompts into detailed, optimized prompts for AI image generation."""

    CATEGORY = "Ollama-Magic-Nodes/Prompt"
    FUNCTION = "enhance"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "negative_prompt", "tags", "changes_summary")
    OUTPUT_TOOLTIPS = (
        "Enhanced, optimized prompt ready for use",
        "Suggested negative prompt",
        "Extracted tags/keywords",
        "Summary of changes made",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "simple_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Your basic prompt to enhance",
                }),
                "system_prompt_preset": (list(SYSTEM_PROMPT_PRESETS), {
                    "default": "default",
                    "tooltip": "System prompt preset to use for enhancement",
                }),
                "enhancement_level": (["subtle", "moderate", "dramatic"], {
                    "default": "moderate",
                    "tooltip": "How much to enhance the prompt",
                }),
                "target_model": (["SDXL", "SD1.5", "Flux", "Midjourney"], {
                    "default": "SDXL",
                    "tooltip": "Target AI model to optimize the prompt for",
                }),
                "aspect_focus": (["balanced", "detail", "mood", "technical"], {
                    "default": "balanced",
                    "tooltip": "What aspect to focus the enhancement on",
                }),
            },
        }

    def enhance(self, ollama_model: dict, simple_prompt: str,
                system_prompt_preset: str = "default", enhancement_level: str = "moderate",
                target_model: str = "SDXL", aspect_focus: str = "balanced"):
        cfg = ollama_model

        system = build_system_prompt(enhancement_level, target_model, aspect_focus, system_prompt_preset)
        user_prompt = build_user_prompt(simple_prompt, system_prompt_preset)

        _log.info("[OllamaNodes] Enhancing prompt with %s (level=%s, target=%s)",
                  cfg["model"], enhancement_level, target_model)

        raw = generate(
            base_url=cfg["base_url"],
            model=cfg["model"],
            prompt=user_prompt,
            system=system,
            temperature=cfg.get("temperature", 0.7),
            num_ctx=cfg.get("num_ctx", 8192),
            num_predict=2048,
            seed=cfg.get("seed", -1),
            keep_alive=cfg.get("keep_alive", "5m"),
            response_format="" if system_prompt_preset == "2026U" else "json",
        )

        parsed = extract_json_block(raw)
        if parsed is None:
            _log.warning("[OllamaNodes] Failed to parse JSON for prompt enhancement")
            return (raw, "", "", "")

        return (
            parsed.get("enhanced_prompt", ""),
            parsed.get("negative_prompt", ""),
            parsed.get("tags", ""),
            parsed.get("changes_summary", ""),
        )
