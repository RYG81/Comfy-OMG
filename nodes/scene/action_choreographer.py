"""
OllamaActionChoreographer — Design dynamic action sequences.
"""
from __future__ import annotations
import json
import logging

from ...ollama_client import generate
from ...prompts.action_choreographer import (
    build_system_prompt, build_user_prompt,
    ACTION_TYPE, INTENSITY_LEVEL, MOTION_PHASE, CAMERA_STYLE, PARTICIPANT_COUNT,
)
from ...prompts.prompt_builder import ENVIRONMENT, SPECIAL_EFFECTS
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)


class OllamaActionChoreographer:
    """Choreograph dynamic action sequences and poses."""

    CATEGORY = "Ollama-Magic-Nodes/Scene"
    FUNCTION = "choreograph"
    RETURN_TYPES = ("STRING",) * 11
    RETURN_NAMES = (
        "action_prompt", "pose_description", "motion_direction",
        "impact_point", "facial_expressions", "clothing_dynamics",
        "environmental_interaction", "energy_visualization",
        "camera_position", "composition_tips", "negative_prompt",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
            },
            "optional": {
                "action_description": ("STRING", {"default": "", "multiline": True}),
                "action_type": (ACTION_TYPE, {"default": "custom"}),
                "action_custom": ("STRING", {"default": ""}),
                "intensity": (INTENSITY_LEVEL, {"default": "energetic - dynamic motion"}),
                "motion_phase": (MOTION_PHASE, {"default": "action - mid-motion"}),
                "motion_custom": ("STRING", {"default": ""}),
                "camera_style": (CAMERA_STYLE, {"default": "dynamic angle - dramatic perspective"}),
                "camera_custom": ("STRING", {"default": ""}),
                "participants": (PARTICIPANT_COUNT, {"default": "solo - single figure"}),
                "environment": (ENVIRONMENT, {"default": "custom"}),
                "env_custom": ("STRING", {"default": ""}),
                "weapons": ("STRING", {"default": "", "tooltip": "Weapons or props involved"}),
                "effects": (SPECIAL_EFFECTS, {"default": "custom"}),
                "effects_custom": ("STRING", {"default": ""}),
                "additional": ("STRING", {"default": "", "multiline": True}),
            },
        }

    def choreograph(self, ollama_model: dict, **kwargs):
        cfg = ollama_model
        
        def get_val(key, custom_key):
            val = kwargs.get(key, "custom")
            custom = kwargs.get(custom_key, "")
            return custom if val == "custom" and custom.strip() else val

        system = build_system_prompt(
            action_type=get_val("action_type", "action_custom"),
            intensity=kwargs.get("intensity", "energetic - dynamic motion"),
            motion_phase=get_val("motion_phase", "motion_custom"),
            camera_style=get_val("camera_style", "camera_custom"),
            participants=kwargs.get("participants", "solo - single figure"),
            environment=get_val("environment", "env_custom"),
            weapons=kwargs.get("weapons", "none"),
            effects=get_val("effects", "effects_custom"),
            custom_instructions=kwargs.get("additional", ""),
        )
        user_prompt = build_user_prompt(kwargs.get("action_description", ""))
        
        _log.info("[OllamaNodes] Choreographing action")
        
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
            return (raw,) + ("",) * 10

        return (
            parsed.get("action_prompt", ""),
            parsed.get("pose_description", ""),
            parsed.get("motion_direction", ""),
            parsed.get("impact_point", ""),
            parsed.get("facial_expressions", ""),
            parsed.get("clothing_dynamics", ""),
            parsed.get("environmental_interaction", ""),
            parsed.get("energy_visualization", ""),
            parsed.get("camera_position", ""),
            parsed.get("composition_tips", ""),
            parsed.get("negative_prompt", ""),
        )
