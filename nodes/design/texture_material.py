"""
OllamaTextureMaterial — Create detailed material and texture descriptions.
"""
from __future__ import annotations
import json
import logging

from ...ollama_client import generate
from ...prompts.texture_material import (
    build_system_prompt, build_user_prompt,
    MATERIAL_CATEGORY, METAL_TYPE, FABRIC_TYPE, STONE_TYPE, WOOD_TYPE,
    SURFACE_FINISH, CONDITION,
)
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)


class OllamaTextureMaterial:
    """Design detailed materials and textures."""

    CATEGORY = "Ollama-Magic-Nodes/Design"
    FUNCTION = "design_material"
    RETURN_TYPES = ("STRING",) * 11
    RETURN_NAMES = (
        "material_prompt", "surface_texture", "color_description",
        "reflectivity", "transparency", "pattern_details",
        "wear_details", "tactile_impression", "close_up_details",
        "material_tags", "negative_prompt",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "category": (MATERIAL_CATEGORY, {"default": "metal"}),
            },
            "optional": {
                "material_description": ("STRING", {"default": "", "multiline": True}),
                "category_custom": ("STRING", {"default": ""}),
                "metal_type": (METAL_TYPE, {"default": "custom"}),
                "fabric_type": (FABRIC_TYPE, {"default": "custom"}),
                "stone_type": (STONE_TYPE, {"default": "custom"}),
                "wood_type": (WOOD_TYPE, {"default": "custom"}),
                "finish": (SURFACE_FINISH, {"default": "custom"}),
                "finish_custom": ("STRING", {"default": ""}),
                "condition": (CONDITION, {"default": "good - light use"}),
                "color": ("STRING", {"default": ""}),
                "pattern": ("STRING", {"default": ""}),
                "lighting": ("STRING", {"default": ""}),
                "special": ("STRING", {"default": ""}),
            },
        }

    def design_material(self, ollama_model: dict, category: str = "metal", **kwargs):
        cfg = ollama_model
        
        # Get specific type based on category
        specific_type = "custom"
        if category == "metal":
            specific_type = kwargs.get("metal_type", "custom")
        elif category == "fabric" or category == "leather":
            specific_type = kwargs.get("fabric_type", "custom")
        elif category == "stone" or category == "ceramic":
            specific_type = kwargs.get("stone_type", "custom")
        elif category == "wood":
            specific_type = kwargs.get("wood_type", "custom")

        system = build_system_prompt(
            category=kwargs.get("category_custom") or category,
            specific_type=specific_type,
            finish=kwargs.get("finish_custom") or kwargs.get("finish", "custom"),
            condition=kwargs.get("condition", "good - light use"),
            color=kwargs.get("color", "custom"),
            pattern=kwargs.get("pattern", "custom"),
            lighting=kwargs.get("lighting", "custom"),
            special=kwargs.get("special", "none"),
        )
        user_prompt = build_user_prompt(kwargs.get("material_description", ""))
        
        _log.info("[OllamaNodes] Designing material: %s", category)
        
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
            return (raw,) + ("",) * 10

        return (
            parsed.get("material_prompt", ""),
            parsed.get("surface_texture", ""),
            parsed.get("color_description", ""),
            parsed.get("reflectivity", ""),
            parsed.get("transparency", ""),
            parsed.get("pattern_details", ""),
            parsed.get("wear_details", ""),
            parsed.get("tactile_impression", ""),
            parsed.get("close_up_details", ""),
            parsed.get("material_tags", ""),
            parsed.get("negative_prompt", ""),
        )
