"""
OllamaSceneDirector - director-grade scene composition node.
"""
from __future__ import annotations
import json
import logging

from ...ollama_client import generate
from ...prompts.scene_director import (
    ASPECT_RATIOS,
    ART_STYLES,
    BLOCKING,
    CAMERA_ANGLES,
    CAMERA_SHOTS,
    COLOR_GRADES,
    COMPOSITION_STYLES,
    DEPTH_OF_FIELD,
    DIRECTOR_INTENT,
    LENS_CHOICES,
    LIGHT_COLOR,
    LIGHTING_SETUPS,
    MOODS,
    OUTPUT_PROFILES,
    POSE_DIRECTION,
    PROMPT_LENGTH,
    SETTINGS,
    SHADOW_STYLE,
    SUBJECT_FOCUS,
    SUBJECT_TYPES,
    TIME_OF_DAY,
    WEATHER,
    build_director_brief,
    build_system_prompt,
    build_user_prompt,
)
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)


def _as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


class OllamaSceneDirector:
    """Directs scenes with subject, pose, blocking, camera, lighting, and style control."""

    CATEGORY = "Ollama-Magic-Nodes/Scene"
    FUNCTION = "direct_scene"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "scene_prompt", "composition_guide", "lighting_setup",
        "character_descriptions", "negative_prompt", "director_notes",
    )
    OUTPUT_TOOLTIPS = (
        "Complete image prompt ready for generation",
        "Spatial blocking and composition notes",
        "Production lighting breakdown",
        "Character-by-character performance and position notes",
        "Suggested negative prompt",
        "Narrative and cinematic intent",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "scene_description": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "What is happening in the scene",
                }),
                "num_characters": ("INT", {
                    "default": 2,
                    "min": 0, "max": 12, "step": 1,
                    "tooltip": "0 for landscape/object scenes",
                }),
            },
            "optional": {
                "output_profile": (OUTPUT_PROFILES, {
                    "default": "balanced image prompt",
                    "tooltip": "Prompt style for the target generator",
                }),
                "aspect_ratio": (ASPECT_RATIOS, {
                    "default": "custom",
                    "tooltip": "Framing target to guide composition",
                }),
                "aspect_custom": ("STRING", {"default": "", "tooltip": "Custom aspect ratio or framing"}),
                "prompt_length": (PROMPT_LENGTH, {
                    "default": "standard",
                    "tooltip": "How dense the final prompt should be",
                }),
                "subject_type": (SUBJECT_TYPES, {"default": "custom", "tooltip": "Primary subject type"}),
                "subject_custom": ("STRING", {"default": "", "tooltip": "Custom subject type"}),
                "subject_focus": (SUBJECT_FOCUS, {"default": "clear main subject", "tooltip": "What the viewer should read first"}),
                "subject_focus_custom": ("STRING", {"default": "", "tooltip": "Custom focal priority"}),
                "pose_direction": (POSE_DIRECTION, {"default": "custom", "tooltip": "Pose or performance direction"}),
                "pose_custom": ("STRING", {"default": "", "tooltip": "Custom pose/performance"}),
                "blocking": (BLOCKING, {"default": "custom", "tooltip": "Where subjects sit in the frame"}),
                "blocking_custom": ("STRING", {"default": "", "tooltip": "Custom blocking"}),
                "setting": (SETTINGS, {"default": "custom", "tooltip": "Scene location"}),
                "setting_custom": ("STRING", {"default": "", "tooltip": "Custom setting"}),
                "time_of_day": (TIME_OF_DAY, {"default": "custom", "tooltip": "Time of day"}),
                "time_custom": ("STRING", {"default": "", "tooltip": "Custom time"}),
                "weather": (WEATHER, {"default": "custom", "tooltip": "Weather/atmosphere"}),
                "weather_custom": ("STRING", {"default": "", "tooltip": "Custom weather"}),
                "mood": (MOODS, {"default": "dramatic", "tooltip": "Emotional tone"}),
                "mood_custom": ("STRING", {"default": "", "tooltip": "Custom mood"}),
                "director_intent": (DIRECTOR_INTENT, {"default": "custom", "tooltip": "What the direction should emphasize"}),
                "intent_custom": ("STRING", {"default": "", "tooltip": "Custom director intent"}),
                "lighting_setup": (LIGHTING_SETUPS, {"default": "cinematic lighting", "tooltip": "Lighting design"}),
                "lighting_custom": ("STRING", {"default": "", "tooltip": "Custom lighting setup"}),
                "light_color": (LIGHT_COLOR, {"default": "custom", "tooltip": "Light color"}),
                "light_color_custom": ("STRING", {"default": "", "tooltip": "Custom light color"}),
                "shadow_style": (SHADOW_STYLE, {"default": "custom", "tooltip": "Shadow behavior"}),
                "shadow_custom": ("STRING", {"default": "", "tooltip": "Custom shadow style"}),
                "camera_shot": (CAMERA_SHOTS, {"default": "wide shot", "tooltip": "Shot size"}),
                "shot_custom": ("STRING", {"default": "", "tooltip": "Custom shot"}),
                "camera_angle": (CAMERA_ANGLES, {"default": "eye level", "tooltip": "Camera angle"}),
                "angle_custom": ("STRING", {"default": "", "tooltip": "Custom camera angle"}),
                "lens": (LENS_CHOICES, {"default": "custom", "tooltip": "Lens character"}),
                "lens_custom": ("STRING", {"default": "", "tooltip": "Custom lens"}),
                "depth_of_field": (DEPTH_OF_FIELD, {"default": "custom", "tooltip": "Focus behavior"}),
                "dof_custom": ("STRING", {"default": "", "tooltip": "Custom focus/DoF"}),
                "composition": (COMPOSITION_STYLES, {"default": "custom", "tooltip": "Composition style"}),
                "composition_custom": ("STRING", {"default": "", "tooltip": "Custom composition"}),
                "art_style": (ART_STYLES, {"default": "custom", "tooltip": "Visual style"}),
                "style_custom": ("STRING", {"default": "", "tooltip": "Custom visual style"}),
                "color_grade": (COLOR_GRADES, {"default": "custom", "tooltip": "Color grade"}),
                "color_grade_custom": ("STRING", {"default": "", "tooltip": "Custom color grade"}),
                "additional_details": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Props, wardrobe, continuity, environment, or other details",
                }),
                "custom_system_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "High-priority guidance that steers how the LLM directs the scene",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        cfg = kwargs.get("ollama_model") or {}
        relevant = {
            "scene_description": kwargs.get("scene_description", ""),
            "num_characters": kwargs.get("num_characters", 2),
            "model": cfg.get("model", ""),
            "base_url": cfg.get("base_url", ""),
            "temperature": cfg.get("temperature", 0.7),
            "num_ctx": cfg.get("num_ctx", 16384),
            "seed": cfg.get("seed", -1),
            "keep_alive": cfg.get("keep_alive", "5m"),
        }
        for key, value in kwargs.items():
            if key != "ollama_model":
                relevant[key] = value
        return json.dumps(relevant, sort_keys=True, separators=(",", ":"), default=str)

    def direct_scene(
        self,
        ollama_model: dict,
        scene_description: str,
        num_characters: int = 2,
        output_profile: str = "balanced image prompt",
        aspect_ratio: str = "custom",
        aspect_custom: str = "",
        prompt_length: str = "standard",
        subject_type: str = "custom",
        subject_custom: str = "",
        subject_focus: str = "clear main subject",
        subject_focus_custom: str = "",
        pose_direction: str = "custom",
        pose_custom: str = "",
        blocking: str = "custom",
        blocking_custom: str = "",
        setting: str = "custom",
        setting_custom: str = "",
        time_of_day: str = "custom",
        time_custom: str = "",
        weather: str = "custom",
        weather_custom: str = "",
        mood: str = "dramatic",
        mood_custom: str = "",
        director_intent: str = "custom",
        intent_custom: str = "",
        lighting_setup: str = "cinematic lighting",
        lighting_custom: str = "",
        light_color: str = "custom",
        light_color_custom: str = "",
        shadow_style: str = "custom",
        shadow_custom: str = "",
        camera_shot: str = "wide shot",
        shot_custom: str = "",
        camera_angle: str = "eye level",
        angle_custom: str = "",
        lens: str = "custom",
        lens_custom: str = "",
        depth_of_field: str = "custom",
        dof_custom: str = "",
        composition: str = "custom",
        composition_custom: str = "",
        art_style: str = "custom",
        style_custom: str = "",
        color_grade: str = "custom",
        color_grade_custom: str = "",
        additional_details: str = "",
        custom_system_prompt: str = "",
    ):
        cfg = ollama_model

        def get_val(dropdown_val: str, custom_val: str) -> str:
            custom_val = str(custom_val or "").strip()
            if dropdown_val == "custom" and custom_val:
                return custom_val
            if dropdown_val != "custom":
                return dropdown_val
            return "custom"

        controls = {
            "output_profile": output_profile,
            "aspect_ratio": get_val(aspect_ratio, aspect_custom),
            "prompt_length": prompt_length,
            "num_characters": num_characters,
            "subject_type": get_val(subject_type, subject_custom),
            "subject_focus": get_val(subject_focus, subject_focus_custom),
            "pose_direction": get_val(pose_direction, pose_custom),
            "blocking": get_val(blocking, blocking_custom),
            "setting": get_val(setting, setting_custom),
            "time_of_day": get_val(time_of_day, time_custom),
            "weather": get_val(weather, weather_custom),
            "mood": get_val(mood, mood_custom),
            "director_intent": get_val(director_intent, intent_custom),
            "lighting_setup": get_val(lighting_setup, lighting_custom),
            "light_color": get_val(light_color, light_color_custom),
            "shadow_style": get_val(shadow_style, shadow_custom),
            "camera_shot": get_val(camera_shot, shot_custom),
            "camera_angle": get_val(camera_angle, angle_custom),
            "lens": get_val(lens, lens_custom),
            "depth_of_field": get_val(depth_of_field, dof_custom),
            "composition": get_val(composition, composition_custom),
            "art_style": get_val(art_style, style_custom),
            "color_grade": get_val(color_grade, color_grade_custom),
            "additional_details": additional_details,
            "custom_system_prompt": custom_system_prompt,
        }

        system = build_system_prompt(**controls)
        director_brief = build_director_brief(**controls)
        user_prompt = build_user_prompt(scene_description, director_brief)

        _log.info(
            "[OllamaNodes] Scene Director with %s (%d characters, controls=%s)",
            cfg["model"],
            num_characters,
            len(director_brief.splitlines()),
        )

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
            _log.warning("[OllamaNodes] Failed to parse JSON for scene direction")
            return (raw, "", "", "", "", "")

        return (
            _as_text(parsed.get("scene_prompt", "")),
            _as_text(parsed.get("composition_guide", "")),
            _as_text(parsed.get("lighting_setup", "")),
            _as_text(parsed.get("character_descriptions", "")),
            _as_text(parsed.get("negative_prompt", "")),
            _as_text(parsed.get("director_notes", "")),
        )
