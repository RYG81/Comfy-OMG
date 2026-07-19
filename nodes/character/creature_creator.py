"""
OllamaCreatureCreator — Design fantasy, sci-fi, or hybrid creatures.
"""
from __future__ import annotations
import json
import logging

from ...ollama_client import generate
from ...prompts.creature_creator import (
    build_system_prompt, build_user_prompt,
    BASE_CREATURE, SIZE_SCALE, BODY_PLAN, SURFACE_COVERING, SPECIAL_FEATURES, TEMPERAMENT,
)
from ...prompts.prompt_builder import ENVIRONMENT, COLOR_PALETTE, ART_STYLE
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)


class OllamaCreatureCreator:
    """Design unique fantasy, sci-fi, or hybrid creatures."""

    CATEGORY = "Ollama-Magic-Nodes/Character"
    FUNCTION = "create_creature"
    RETURN_TYPES = ("STRING",) * 10
    RETURN_NAMES = (
        "creature_prompt", "anatomy_description", "surface_details",
        "feature_details", "coloration", "eye_description",
        "movement_impression", "size_reference", "style_tags", "negative_prompt",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
            },
            "optional": {
                "creature_concept": ("STRING", {"default": "", "multiline": True}),
                "base": (BASE_CREATURE, {"default": "original design"}),
                "base_custom": ("STRING", {"default": ""}),
                "size": (SIZE_SCALE, {"default": "large - horse sized"}),
                "body_plan": (BODY_PLAN, {"default": "custom"}),
                "body_custom": ("STRING", {"default": ""}),
                "surface": (SURFACE_COVERING, {"default": "custom"}),
                "surface_custom": ("STRING", {"default": ""}),
                "features": (SPECIAL_FEATURES, {"default": "custom"}),
                "features_custom": ("STRING", {"default": ""}),
                "temperament": (TEMPERAMENT, {"default": "custom"}),
                "temperament_custom": ("STRING", {"default": ""}),
                "habitat": (ENVIRONMENT, {"default": "custom"}),
                "habitat_custom": ("STRING", {"default": ""}),
                "colors": (COLOR_PALETTE, {"default": "custom"}),
                "colors_custom": ("STRING", {"default": ""}),
                "art_style": (ART_STYLE, {"default": "custom"}),
                "style_custom": ("STRING", {"default": ""}),
                "additional_details": ("STRING", {"default": "", "multiline": True}),
            },
        }

    def create_creature(self, ollama_model: dict, **kwargs):
        cfg = ollama_model
        
        def get_val(key, custom_key):
            val = kwargs.get(key, "custom")
            custom = kwargs.get(custom_key, "")
            return custom if val == "custom" and custom.strip() else val

        system = build_system_prompt(
            base=get_val("base", "base_custom"),
            size=kwargs.get("size", "large - horse sized"),
            body_plan=get_val("body_plan", "body_custom"),
            surface=get_val("surface", "surface_custom"),
            features=get_val("features", "features_custom"),
            temperament=get_val("temperament", "temperament_custom"),
            habitat=get_val("habitat", "habitat_custom"),
            colors=get_val("colors", "colors_custom"),
            art_style=get_val("art_style", "style_custom"),
            custom_details=kwargs.get("additional_details", ""),
        )
        user_prompt = build_user_prompt(kwargs.get("creature_concept", ""))
        
        _log.info("[OllamaNodes] Creating creature")
        
        raw = generate(
            base_url=cfg["base_url"], model=cfg["model"],
            prompt=user_prompt, system=system,
            temperature=cfg.get("temperature", 0.8),
            num_ctx=cfg.get("num_ctx", 8192),
            num_predict=2500,
            seed=cfg.get("seed", -1),
            keep_alive=cfg.get("keep_alive", "5m"),
            response_format="json",
        )

        parsed = extract_json_block(raw)
        if parsed is None:
            return (raw,) + ("",) * 9

        return (
            parsed.get("creature_prompt", ""),
            parsed.get("anatomy_description", ""),
            parsed.get("surface_details", ""),
            parsed.get("feature_details", ""),
            parsed.get("coloration", ""),
            parsed.get("eye_description", ""),
            parsed.get("movement_impression", ""),
            parsed.get("size_reference", ""),
            parsed.get("style_tags", ""),
            parsed.get("negative_prompt", ""),
        )
