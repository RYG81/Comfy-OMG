"""
OllamaOutfitGenerator — Outfit Generator
Generates detailed outfit/costume descriptions.
"""
from __future__ import annotations
import json
import logging

from ...ollama_client import generate
from ...prompts.outfit_generator import build_system_prompt, build_user_prompt
from ...utils.image_utils import tensor_to_base64
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)

OUTFIT_TYPES = ["casual", "formal", "fantasy", "sci-fi", "historical", "uniform", "costume", "streetwear", "haute_couture"]
SEASONS = ["spring", "summer", "fall", "winter", "any"]


class OllamaOutfitGenerator:
    """Generates detailed outfit descriptions for characters."""

    CATEGORY = "Ollama-Magic-Nodes/Character"
    FUNCTION = "generate_outfit"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "outfit_prompt", "top_description", "bottom_description",
        "accessories", "footwear", "outfit_tags",
    )
    OUTPUT_TOOLTIPS = (
        "Complete outfit description prompt",
        "Upper body clothing",
        "Lower body clothing",
        "Accessories and jewelry",
        "Shoes/boots description",
        "Comma-separated clothing tags",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "character_description": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Who is wearing the outfit",
                }),
                "outfit_type": (OUTFIT_TYPES, {
                    "default": "casual",
                    "tooltip": "Type of outfit to generate",
                }),
                "season": (SEASONS, {
                    "default": "any",
                    "tooltip": "Season/weather the outfit is for",
                }),
            },
            "optional": {
                "color_scheme": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Preferred colors (optional)",
                }),
                "reference_image": ("IMAGE", {"tooltip": "Optional outfit reference"}),
            },
        }

    def generate_outfit(self, ollama_model: dict, character_description: str,
                       outfit_type: str = "casual", season: str = "any",
                       color_scheme: str = "", reference_image=None):
        cfg = ollama_model
        has_reference = reference_image is not None
        system = build_system_prompt(character_description, outfit_type, season, color_scheme, has_reference)
        user_prompt = build_user_prompt(has_reference)

        _log.info("[OllamaNodes] Generating outfit (type=%s, season=%s)", outfit_type, season)

        images = []
        if has_reference:
            images.append(tensor_to_base64(reference_image))

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
            images=images if images else None,
            response_format="json",
        )

        parsed = extract_json_block(raw)
        if parsed is None:
            _log.warning("[OllamaNodes] Failed to parse JSON for outfit")
            return (raw, "", "", "", "", "")

        return (
            parsed.get("outfit_prompt", ""),
            parsed.get("top_description", ""),
            parsed.get("bottom_description", ""),
            parsed.get("accessories", ""),
            parsed.get("footwear", ""),
            parsed.get("outfit_tags", ""),
        )
