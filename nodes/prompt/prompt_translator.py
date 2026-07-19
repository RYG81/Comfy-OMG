"""
OllamaPromptTranslator — Convert prompts between model formats.
"""
from __future__ import annotations
import json
import logging

from ...ollama_client import generate
from ...prompts.prompt_translator import FORMATS, build_system_prompt, build_user_prompt
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)


class OllamaPromptTranslator:
    """Translate prompts between different AI model formats."""

    CATEGORY = "Ollama-Magic-Nodes/Prompt"
    FUNCTION = "translate"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "translated_prompt", "source_analysis", "translation_notes",
        "format_additions", "potential_issues", "negative_prompt",
    )
    OUTPUT_TOOLTIPS = (
        "Prompt in target format",
        "Analysis of source prompt",
        "Changes made during translation",
        "Format-specific elements added",
        "Potential translation issues",
        "Negative prompt for target format",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Prompt to translate",
                }),
                "source_format": (list(FORMATS.keys()), {
                    "default": "booru_tags",
                    "tooltip": "Source prompt format",
                }),
                "target_format": (list(FORMATS.keys()), {
                    "default": "natural_flux",
                    "tooltip": "Target prompt format",
                }),
            },
        }

    def translate(self, ollama_model: dict, prompt: str,
                 source_format: str = "booru_tags", target_format: str = "natural_flux"):
        cfg = ollama_model
        
        if source_format == target_format:
            return (prompt, "Same format, no translation needed", "", "", "", "")
        
        system = build_system_prompt(source_format, target_format)
        user_prompt = build_user_prompt(prompt)

        _log.info("[OllamaNodes] Translating %s → %s", source_format, target_format)

        raw = generate(
            base_url=cfg["base_url"],
            model=cfg["model"],
            prompt=user_prompt,
            system=system,
            temperature=cfg.get("temperature", 0.5),
            num_ctx=cfg.get("num_ctx", 8192),
            num_predict=2000,
            seed=cfg.get("seed", -1),
            keep_alive=cfg.get("keep_alive", "5m"),
            response_format="json",
        )

        parsed = extract_json_block(raw)
        if parsed is None:
            return (raw, "", "", "", "", "")

        return (
            parsed.get("translated_prompt", ""),
            parsed.get("source_analysis", ""),
            parsed.get("translation_notes", ""),
            parsed.get("format_additions", ""),
            parsed.get("potential_issues", ""),
            parsed.get("negative_prompt", ""),
        )
