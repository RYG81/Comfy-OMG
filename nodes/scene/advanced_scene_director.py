"""
OllamaAdvancedSceneDirector — Advanced Scene Director
Full cinematic control with comprehensive dropdown options.
"""
from __future__ import annotations
import json
import logging

from ...ollama_client import generate
from ...prompts.advanced_scene_director import (
    build_system_prompt, build_user_prompt,
    SCENE_TYPE, CHARACTER_RELATIONSHIP, SCENE_ACTION, EMOTIONAL_TENSION,
    SCENE_COMPOSITION, ENVIRONMENT_SCALE, ENVIRONMENT_CONDITION,
    FOREGROUND_ELEMENTS, BACKGROUND_ELEMENTS,
    LIGHTING_SETUP, LIGHT_COLOR, SHADOW_INTENSITY,
    CAMERA_MOVEMENT_FEEL, LENS_EFFECTS, POST_PROCESSING,
)
from ...prompts.prompt_builder import (
    ENVIRONMENT, TIME_OF_DAY, WEATHER, CAMERA_SHOT, CAMERA_ANGLE,
    MOOD_ATMOSPHERE, ART_STYLE,
)
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)


class OllamaAdvancedSceneDirector:
    """Advanced scene director with comprehensive dropdown options for every aspect."""

    CATEGORY = "Ollama-Magic-Nodes/Scene"
    FUNCTION = "direct_scene"
    RETURN_TYPES = ("STRING",) * 10
    RETURN_NAMES = (
        "scene_prompt", "character_breakdown", "spatial_layout",
        "lighting_description", "camera_technical", "atmosphere_details",
        "composition_notes", "color_script", "negative_prompt", "director_vision",
    )
    OUTPUT_TOOLTIPS = (
        "Complete scene prompt",
        "Individual character descriptions",
        "Spatial arrangement details",
        "Lighting breakdown",
        "Camera specifications",
        "Atmospheric elements",
        "Composition analysis",
        "Color palette and grading",
        "Negative prompt",
        "Director's narrative intent",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "scene_description": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Describe what's happening in the scene",
                }),
                "num_characters": ("INT", {
                    "default": 1,
                    "min": 0, "max": 10, "step": 1,
                    "tooltip": "Number of characters (0 for landscape/still life)",
                }),
            },
            "optional": {
                # ═══ SCENE ═══
                "scene_type": (SCENE_TYPE, {"default": "custom", "tooltip": "Type of scene"}),
                "scene_type_custom": ("STRING", {"default": "", "tooltip": "Custom scene type"}),
                
                "relationship": (CHARACTER_RELATIONSHIP, {"default": "custom", "tooltip": "Character relationship"}),
                "relationship_custom": ("STRING", {"default": "", "tooltip": "Custom relationship"}),
                
                "scene_action": (SCENE_ACTION, {"default": "custom", "tooltip": "What's happening"}),
                "action_custom": ("STRING", {"default": "", "tooltip": "Custom action"}),
                
                "emotional_tension": (EMOTIONAL_TENSION, {"default": "custom", "tooltip": "Emotional tension level"}),
                "tension_custom": ("STRING", {"default": "", "tooltip": "Custom tension"}),
                
                # ═══ ENVIRONMENT ═══
                "environment": (ENVIRONMENT, {"default": "custom", "tooltip": "Environment/setting"}),
                "environment_custom": ("STRING", {"default": "", "tooltip": "Custom environment"}),
                
                "env_scale": (ENVIRONMENT_SCALE, {"default": "custom", "tooltip": "Environment scale"}),
                "scale_custom": ("STRING", {"default": "", "tooltip": "Custom scale"}),
                
                "env_condition": (ENVIRONMENT_CONDITION, {"default": "custom", "tooltip": "Environment condition"}),
                "condition_custom": ("STRING", {"default": "", "tooltip": "Custom condition"}),
                
                "time_of_day": (TIME_OF_DAY, {"default": "custom", "tooltip": "Time of day"}),
                "time_custom": ("STRING", {"default": "", "tooltip": "Custom time"}),
                
                "weather": (WEATHER, {"default": "custom", "tooltip": "Weather"}),
                "weather_custom": ("STRING", {"default": "", "tooltip": "Custom weather"}),
                
                "foreground": (FOREGROUND_ELEMENTS, {"default": "custom", "tooltip": "Foreground elements"}),
                "foreground_custom": ("STRING", {"default": "", "tooltip": "Custom foreground"}),
                
                "background": (BACKGROUND_ELEMENTS, {"default": "custom", "tooltip": "Background elements"}),
                "background_custom": ("STRING", {"default": "", "tooltip": "Custom background"}),
                
                # ═══ LIGHTING ═══
                "lighting_setup": (LIGHTING_SETUP, {"default": "custom", "tooltip": "Lighting setup"}),
                "lighting_custom": ("STRING", {"default": "", "tooltip": "Custom lighting setup"}),
                
                "light_color": (LIGHT_COLOR, {"default": "custom", "tooltip": "Light color"}),
                "light_color_custom": ("STRING", {"default": "", "tooltip": "Custom light color"}),
                
                "shadow_intensity": (SHADOW_INTENSITY, {"default": "custom", "tooltip": "Shadow intensity"}),
                "shadow_custom": ("STRING", {"default": "", "tooltip": "Custom shadows"}),
                
                # ═══ CAMERA ═══
                "camera_shot": (CAMERA_SHOT, {"default": "custom", "tooltip": "Camera shot type"}),
                "shot_custom": ("STRING", {"default": "", "tooltip": "Custom shot"}),
                
                "camera_angle": (CAMERA_ANGLE, {"default": "custom", "tooltip": "Camera angle"}),
                "angle_custom": ("STRING", {"default": "", "tooltip": "Custom angle"}),
                
                "camera_movement": (CAMERA_MOVEMENT_FEEL, {"default": "custom", "tooltip": "Camera movement feel"}),
                "movement_custom": ("STRING", {"default": "", "tooltip": "Custom movement"}),
                
                "lens_effects": (LENS_EFFECTS, {"default": "custom", "tooltip": "Lens effects"}),
                "lens_custom": ("STRING", {"default": "", "tooltip": "Custom lens effects"}),
                
                "post_processing": (POST_PROCESSING, {"default": "custom", "tooltip": "Post-processing/color grade"}),
                "post_custom": ("STRING", {"default": "", "tooltip": "Custom post-processing"}),
                
                # ═══ STYLE & MOOD ═══
                "mood": (MOOD_ATMOSPHERE, {"default": "custom", "tooltip": "Mood/atmosphere"}),
                "mood_custom": ("STRING", {"default": "", "tooltip": "Custom mood"}),
                
                "art_style": (ART_STYLE, {"default": "custom", "tooltip": "Art style"}),
                "style_custom": ("STRING", {"default": "", "tooltip": "Custom art style"}),
                
                "composition": (SCENE_COMPOSITION, {"default": "custom", "tooltip": "Composition style"}),
                "composition_custom": ("STRING", {"default": "", "tooltip": "Custom composition"}),
                
                # ═══ ADDITIONAL ═══
                "additional_instructions": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Any additional instructions or details",
                }),
            },
        }

    def direct_scene(self, ollama_model: dict, scene_description: str,
                    num_characters: int = 1, **kwargs):
        cfg = ollama_model
        
        # Helper to get value (custom or dropdown)
        def get_val(dropdown_key, custom_key):
            dropdown_val = kwargs.get(dropdown_key, "custom")
            custom_val = kwargs.get(custom_key, "")
            if dropdown_val == "custom" and custom_val.strip():
                return custom_val.strip()
            elif dropdown_val != "custom":
                return dropdown_val
            return "custom"
        
        system = build_system_prompt(
            scene_type=get_val("scene_type", "scene_type_custom"),
            num_characters=num_characters,
            relationship=get_val("relationship", "relationship_custom"),
            scene_action=get_val("scene_action", "action_custom"),
            emotional_tension=get_val("emotional_tension", "tension_custom"),
            environment=get_val("environment", "environment_custom"),
            env_scale=get_val("env_scale", "scale_custom"),
            env_condition=get_val("env_condition", "condition_custom"),
            time_of_day=get_val("time_of_day", "time_custom"),
            weather=get_val("weather", "weather_custom"),
            lighting_setup=get_val("lighting_setup", "lighting_custom"),
            light_color=get_val("light_color", "light_color_custom"),
            shadow_intensity=get_val("shadow_intensity", "shadow_custom"),
            camera_shot=get_val("camera_shot", "shot_custom"),
            camera_angle=get_val("camera_angle", "angle_custom"),
            camera_movement=get_val("camera_movement", "movement_custom"),
            lens_effects=get_val("lens_effects", "lens_custom"),
            post_processing=get_val("post_processing", "post_custom"),
            mood=get_val("mood", "mood_custom"),
            art_style=get_val("art_style", "style_custom"),
            composition=get_val("composition", "composition_custom"),
            foreground=get_val("foreground", "foreground_custom"),
            background=get_val("background", "background_custom"),
            custom_instructions=kwargs.get("additional_instructions", ""),
        )
        
        user_prompt = build_user_prompt(scene_description)

        _log.info("[OllamaNodes] Advanced scene direction (%d characters)", num_characters)

        raw = generate(
            base_url=cfg["base_url"],
            model=cfg["model"],
            prompt=user_prompt,
            system=system,
            temperature=cfg.get("temperature", 0.7),
            num_ctx=cfg.get("num_ctx", 16384),
            num_predict=4096,
            seed=cfg.get("seed", -1),
            keep_alive=cfg.get("keep_alive", "5m"),
            response_format="json",
        )

        parsed = extract_json_block(raw)
        if parsed is None:
            _log.warning("[OllamaNodes] Failed to parse JSON for advanced scene")
            return (raw,) + ("",) * 9

        return (
            parsed.get("scene_prompt", ""),
            parsed.get("character_breakdown", ""),
            parsed.get("spatial_layout", ""),
            parsed.get("lighting_description", ""),
            parsed.get("camera_technical", ""),
            parsed.get("atmosphere_details", ""),
            parsed.get("composition_notes", ""),
            parsed.get("color_script", ""),
            parsed.get("negative_prompt", ""),
            parsed.get("director_vision", ""),
        )
