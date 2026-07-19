"""
OllamaEnvironmentTransform — Environment Transformer
Transforms scene weather, time, and season.
"""
from __future__ import annotations
import json
import logging

from ...ollama_client import generate
from ...prompts.environment_transform import build_system_prompt, USER_PROMPT
from ...utils.image_utils import tensor_to_base64
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)

WEATHER_OPTIONS = ["sunny", "cloudy", "rainy", "stormy", "snowy", "foggy", "windy"]
TIME_OPTIONS = ["dawn", "morning", "noon", "afternoon", "sunset", "dusk", "night", "midnight"]
SEASON_OPTIONS = ["spring", "summer", "autumn", "winter", "keep_original"]


class OllamaEnvironmentTransform:
    """Transforms scene environment — weather, time, and season."""

    CATEGORY = "Ollama-Magic-Nodes/Scene"
    FUNCTION = "transform"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "transformed_prompt", "original_analysis", "lighting_description",
        "atmosphere_description", "negative_prompt",
    )
    OUTPUT_TOOLTIPS = (
        "Prompt for the transformed scene",
        "Analysis of original environment",
        "New lighting setup",
        "New atmosphere/mood",
        "What to avoid",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Original scene image"}),
                "ollama_model": ("OLLAMA_MODEL",),
                "target_weather": (WEATHER_OPTIONS, {
                    "default": "sunny",
                    "tooltip": "Target weather condition",
                }),
                "target_time": (TIME_OPTIONS, {
                    "default": "noon",
                    "tooltip": "Target time of day",
                }),
                "target_season": (SEASON_OPTIONS, {
                    "default": "keep_original",
                    "tooltip": "Target season",
                }),
                "transformation_strength": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.3, "max": 1.0, "step": 0.1,
                    "tooltip": "How dramatically to transform",
                }),
            },
        }

    def transform(self, image, ollama_model: dict, target_weather: str = "sunny",
                 target_time: str = "noon", target_season: str = "keep_original",
                 transformation_strength: float = 0.7):
        cfg = ollama_model
        img_b64 = tensor_to_base64(image)
        system = build_system_prompt(target_weather, target_time, target_season, transformation_strength)

        _log.info("[OllamaNodes] Environment transform (weather=%s, time=%s, season=%s)",
                  target_weather, target_time, target_season)

        raw = generate(
            base_url=cfg["base_url"],
            model=cfg["model"],
            prompt=USER_PROMPT,
            system=system,
            temperature=cfg.get("temperature", 0.5),
            num_ctx=cfg.get("num_ctx", 8192),
            num_predict=2048,
            seed=cfg.get("seed", -1),
            keep_alive=cfg.get("keep_alive", "5m"),
            images=[img_b64],
            response_format="json",
        )

        parsed = extract_json_block(raw)
        if parsed is None:
            _log.warning("[OllamaNodes] Failed to parse JSON for environment transform")
            return (raw, "", "", "", "")

        return (
            parsed.get("transformed_prompt", ""),
            parsed.get("original_analysis", ""),
            parsed.get("lighting_description", ""),
            parsed.get("atmosphere_description", ""),
            parsed.get("negative_prompt", ""),
        )
