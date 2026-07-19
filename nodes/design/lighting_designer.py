"""
OllamaLightingDesigner — Design detailed lighting setups.
"""
from __future__ import annotations
import json
import logging

from ...ollama_client import generate
from ...prompts.lighting_designer import (
    build_system_prompt, build_user_prompt,
    LIGHTING_SCENARIO, KEY_LIGHT_POSITION, KEY_LIGHT_QUALITY,
    FILL_LIGHT, RIM_LIGHT, PRACTICAL_LIGHTS, COLOR_TEMPERATURE,
    SHADOW_CHARACTER, VOLUMETRIC,
)
from ...prompts.prompt_builder import MOOD_ATMOSPHERE
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)


class OllamaLightingDesigner:
    """Design detailed, production-quality lighting setups."""

    CATEGORY = "Ollama-Magic-Nodes/Design"
    FUNCTION = "design_lighting"
    RETURN_TYPES = ("STRING",) * 8
    RETURN_NAMES = (
        "lighting_prompt", "technical_setup", "color_palette",
        "shadow_description", "atmosphere_effects", "mood_contribution",
        "style_tags", "avoid",
    )
    OUTPUT_TOOLTIPS = (
        "Complete lighting description for prompts",
        "Technical lighting breakdown",
        "Light colors and temperatures",
        "Shadow description",
        "Atmospheric effects",
        "How lighting affects mood",
        "Lighting tags for prompts",
        "What to avoid",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "scenario": (LIGHTING_SCENARIO, {"default": "cinematic", "tooltip": "Lighting scenario"}),
            },
            "optional": {
                "scenario_custom": ("STRING", {"default": "", "tooltip": "Custom scenario"}),
                "subject": ("STRING", {"default": "", "multiline": False, "tooltip": "What's being lit"}),
                
                "key_position": (KEY_LIGHT_POSITION, {"default": "front left 45°", "tooltip": "Key light position"}),
                "key_custom": ("STRING", {"default": "", "tooltip": "Custom key position"}),
                
                "key_quality": (KEY_LIGHT_QUALITY, {"default": "soft (large diffused)", "tooltip": "Key light quality"}),
                "quality_custom": ("STRING", {"default": "", "tooltip": "Custom quality"}),
                
                "fill_light": (FILL_LIGHT, {"default": "moderate (1:2 ratio)", "tooltip": "Fill light"}),
                "fill_custom": ("STRING", {"default": "", "tooltip": "Custom fill"}),
                
                "rim_light": (RIM_LIGHT, {"default": "subtle edge", "tooltip": "Rim/back light"}),
                "rim_custom": ("STRING", {"default": "", "tooltip": "Custom rim"}),
                
                "practical": (PRACTICAL_LIGHTS, {"default": "none", "tooltip": "Practical lights"}),
                "practical_custom": ("STRING", {"default": "", "tooltip": "Custom practicals"}),
                
                "color_temp": (COLOR_TEMPERATURE, {"default": "daylight (5600K)", "tooltip": "Color temperature"}),
                "temp_custom": ("STRING", {"default": "", "tooltip": "Custom temperature"}),
                
                "shadow_char": (SHADOW_CHARACTER, {"default": "soft gradual falloff", "tooltip": "Shadow character"}),
                "shadow_custom": ("STRING", {"default": "", "tooltip": "Custom shadows"}),
                
                "volumetric": (VOLUMETRIC, {"default": "none", "tooltip": "Volumetric effects"}),
                "volumetric_custom": ("STRING", {"default": "", "tooltip": "Custom volumetric"}),
                
                "mood": (MOOD_ATMOSPHERE, {"default": "custom", "tooltip": "Mood"}),
                "mood_custom": ("STRING", {"default": "", "tooltip": "Custom mood"}),
                
                "additional": ("STRING", {"default": "", "multiline": True, "tooltip": "Additional instructions"}),
            },
        }

    def design_lighting(self, ollama_model: dict, scenario: str = "cinematic", **kwargs):
        cfg = ollama_model
        
        def get_val(dropdown_key, custom_key, default_dropdown="custom"):
            dropdown_val = kwargs.get(dropdown_key, default_dropdown)
            custom_val = kwargs.get(custom_key, "")
            if dropdown_val == "custom" and custom_val.strip():
                return custom_val.strip()
            elif dropdown_val != "custom":
                return dropdown_val
            return default_dropdown
        
        scenario_val = scenario if scenario != "custom" else kwargs.get("scenario_custom", "cinematic")
        
        system = build_system_prompt(
            scenario=scenario_val,
            key_position=get_val("key_position", "key_custom", "front left 45°"),
            key_quality=get_val("key_quality", "quality_custom", "soft (large diffused)"),
            fill_light=get_val("fill_light", "fill_custom", "moderate (1:2 ratio)"),
            rim_light=get_val("rim_light", "rim_custom", "subtle edge"),
            practical=get_val("practical", "practical_custom", "none"),
            color_temp=get_val("color_temp", "temp_custom", "daylight (5600K)"),
            shadow_char=get_val("shadow_char", "shadow_custom", "soft gradual falloff"),
            volumetric=get_val("volumetric", "volumetric_custom", "none"),
            mood=get_val("mood", "mood_custom", "cinematic"),
            custom_instructions=kwargs.get("additional", ""),
        )
        
        user_prompt = build_user_prompt(kwargs.get("subject", ""))

        _log.info("[OllamaNodes] Designing lighting for %s", scenario_val)

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
            return (raw,) + ("",) * 7

        return (
            parsed.get("lighting_prompt", ""),
            parsed.get("technical_setup", ""),
            parsed.get("color_palette", ""),
            parsed.get("shadow_description", ""),
            parsed.get("atmosphere_effects", ""),
            parsed.get("mood_contribution", ""),
            parsed.get("style_tags", ""),
            parsed.get("avoid", ""),
        )
