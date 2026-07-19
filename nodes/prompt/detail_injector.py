"""
OllamaDetailInjector — Add specific types of details to prompts.
"""
from __future__ import annotations
import json
import logging

from ...ollama_client import generate
from ...prompts.detail_injector import DETAIL_TYPE, DETAIL_INTENSITY, build_system_prompt, build_user_prompt
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)


class OllamaDetailInjector:
    """Inject specific types of details into existing prompts."""

    CATEGORY = "Ollama-Magic-Nodes/Prompt"
    FUNCTION = "inject_details"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "enhanced_prompt", "added_details", "detail_tags",
        "quality_boost", "negative_additions",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "intensity": (DETAIL_INTENSITY, {"default": "moderate - noticeable but balanced"}),
            },
            "optional": {
                "texture": ("BOOLEAN", {"default": True}),
                "material": ("BOOLEAN", {"default": True}),
                "lighting_details": ("BOOLEAN", {"default": False}),
                "micro_details": ("BOOLEAN", {"default": False}),
                "wear_and_tear": ("BOOLEAN", {"default": False}),
                "reflections": ("BOOLEAN", {"default": False}),
                "atmospheric": ("BOOLEAN", {"default": False}),
                "custom_detail_type": ("STRING", {"default": ""}),
            },
        }

    def inject_details(self, ollama_model: dict, prompt: str, intensity: str, **kwargs):
        cfg = ollama_model
        
        detail_types = []
        if kwargs.get("texture", True): detail_types.append("texture")
        if kwargs.get("material", True): detail_types.append("material")
        if kwargs.get("lighting_details"): detail_types.append("lighting_details")
        if kwargs.get("micro_details"): detail_types.append("micro_details")
        if kwargs.get("wear_and_tear"): detail_types.append("wear_and_tear")
        if kwargs.get("reflections"): detail_types.append("reflections")
        if kwargs.get("atmospheric"): detail_types.append("atmospheric")
        if kwargs.get("custom_detail_type", "").strip():
            detail_types.append(kwargs["custom_detail_type"])

        system = build_system_prompt(detail_types, intensity)
        user_prompt = build_user_prompt(prompt)
        
        _log.info("[OllamaNodes] Injecting details: %s", detail_types)
        
        raw = generate(
            base_url=cfg["base_url"], model=cfg["model"],
            prompt=user_prompt, system=system,
            temperature=cfg.get("temperature", 0.6),
            num_ctx=cfg.get("num_ctx", 8192),
            num_predict=2000,
            seed=cfg.get("seed", -1),
            keep_alive=cfg.get("keep_alive", "5m"),
            response_format="json",
        )

        parsed = extract_json_block(raw)
        if parsed is None:
            return (raw, "", "", "", "")

        return (
            parsed.get("enhanced_prompt", ""),
            parsed.get("added_details", ""),
            parsed.get("detail_tags", ""),
            parsed.get("quality_boost", ""),
            parsed.get("negative_additions", ""),
        )
