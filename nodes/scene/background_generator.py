"""
OllamaBackgroundGenerator — Generate detailed backgrounds separately from subjects.
"""
from __future__ import annotations
import json
import logging

from ...ollama_client import generate
from ...prompts.background_generator import (
    build_system_prompt, build_user_prompt,
    BACKGROUND_TYPE, ARCHITECTURE_STYLE, NATURE_TYPE, URBAN_TYPE, ROOM_TYPE,
    DEPTH_COMPLEXITY, POPULATION,
)
from ...prompts.prompt_builder import TIME_OF_DAY, WEATHER, SEASON, MOOD_ATMOSPHERE, COLOR_PALETTE
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)


class OllamaBackgroundGenerator:
    """Generate detailed background/environment descriptions."""

    CATEGORY = "Ollama-Magic-Nodes/Scene"
    FUNCTION = "generate_background"
    RETURN_TYPES = ("STRING",) * 9
    RETURN_NAMES = (
        "background_prompt", "foreground_elements", "midground_elements",
        "background_elements", "atmospheric_effects", "color_description",
        "depth_cues", "integration_tips", "negative_prompt",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
            },
            "optional": {
                "description": ("STRING", {"default": "", "multiline": True, "tooltip": "Custom description"}),
                "bg_type": (BACKGROUND_TYPE, {"default": "custom"}),
                "bg_type_custom": ("STRING", {"default": ""}),
                "architecture": (ARCHITECTURE_STYLE, {"default": "custom"}),
                "architecture_custom": ("STRING", {"default": ""}),
                "nature": (NATURE_TYPE, {"default": "custom"}),
                "nature_custom": ("STRING", {"default": ""}),
                "urban": (URBAN_TYPE, {"default": "custom"}),
                "urban_custom": ("STRING", {"default": ""}),
                "room": (ROOM_TYPE, {"default": "custom"}),
                "room_custom": ("STRING", {"default": ""}),
                "depth": (DEPTH_COMPLEXITY, {"default": "medium - foreground and background"}),
                "population": (POPULATION, {"default": "empty - no one"}),
                "time_of_day": (TIME_OF_DAY, {"default": "custom"}),
                "time_custom": ("STRING", {"default": ""}),
                "weather": (WEATHER, {"default": "custom"}),
                "weather_custom": ("STRING", {"default": ""}),
                "season": (SEASON, {"default": "custom"}),
                "season_custom": ("STRING", {"default": ""}),
                "mood": (MOOD_ATMOSPHERE, {"default": "custom"}),
                "mood_custom": ("STRING", {"default": ""}),
                "colors": (COLOR_PALETTE, {"default": "custom"}),
                "colors_custom": ("STRING", {"default": ""}),
            },
        }

    def generate_background(self, ollama_model: dict, **kwargs):
        cfg = ollama_model
        
        def get_val(key, custom_key):
            val = kwargs.get(key, "custom")
            custom = kwargs.get(custom_key, "")
            return custom if val == "custom" and custom.strip() else val

        system = build_system_prompt(
            bg_type=get_val("bg_type", "bg_type_custom"),
            architecture=get_val("architecture", "architecture_custom"),
            nature=get_val("nature", "nature_custom"),
            urban=get_val("urban", "urban_custom"),
            room=get_val("room", "room_custom"),
            depth=kwargs.get("depth", "medium - foreground and background"),
            population=kwargs.get("population", "empty - no one"),
            time_of_day=get_val("time_of_day", "time_custom"),
            weather=get_val("weather", "weather_custom"),
            season=get_val("season", "season_custom"),
            mood=get_val("mood", "mood_custom"),
            colors=get_val("colors", "colors_custom"),
        )

        user_prompt = build_user_prompt(kwargs.get("description", ""))
        
        _log.info("[OllamaNodes] Generating background")
        
        raw = generate(
            base_url=cfg["base_url"], model=cfg["model"],
            prompt=user_prompt, system=system,
            temperature=cfg.get("temperature", 0.7),
            num_ctx=cfg.get("num_ctx", 8192),
            num_predict=2500,
            seed=cfg.get("seed", -1),
            keep_alive=cfg.get("keep_alive", "5m"),
            response_format="json",
        )

        parsed = extract_json_block(raw)
        if parsed is None:
            return (raw,) + ("",) * 8

        return (
            parsed.get("background_prompt", ""),
            parsed.get("foreground_elements", ""),
            parsed.get("midground_elements", ""),
            parsed.get("background_elements", ""),
            parsed.get("atmospheric_effects", ""),
            parsed.get("color_description", ""),
            parsed.get("depth_cues", ""),
            parsed.get("integration_tips", ""),
            parsed.get("negative_prompt", ""),
        )
