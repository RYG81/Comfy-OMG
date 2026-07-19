"""
OllamaHandPoseHelper — Generate AI-friendly hand pose descriptions.
"""
from __future__ import annotations
import json
import logging

from ...ollama_client import generate
from ...prompts.hand_pose_helper import (
    build_system_prompt, build_user_prompt,
    HAND_ACTION, HAND_POSITION, WHICH_HANDS, DETAIL_LEVEL,
)
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)


class OllamaHandPoseHelper:
    """Generate detailed, AI-optimized hand pose descriptions."""

    CATEGORY = "Ollama-Magic-Nodes/Character"
    FUNCTION = "describe_hands"
    RETURN_TYPES = ("STRING",) * 11
    RETURN_NAMES = (
        "hand_prompt", "right_hand", "left_hand", "finger_positions",
        "palm_orientation", "wrist_angle", "hand_interaction",
        "simplification_tips", "hand_tags", "negative_prompt", "alternative_pose",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "action": (HAND_ACTION, {"default": "relaxed open"}),
            },
            "optional": {
                "description": ("STRING", {"default": "", "multiline": True}),
                "action_custom": ("STRING", {"default": ""}),
                "position": (HAND_POSITION, {"default": "at sides"}),
                "position_custom": ("STRING", {"default": ""}),
                "which_hands": (WHICH_HANDS, {"default": "both hands same pose"}),
                "detail_level": (DETAIL_LEVEL, {"default": "standard - clear finger positions"}),
                "holding": ("STRING", {"default": "", "tooltip": "Object being held"}),
                "interaction": ("STRING", {"default": "", "tooltip": "Hand interaction"}),
            },
        }

    def describe_hands(self, ollama_model: dict, action: str = "relaxed open", **kwargs):
        cfg = ollama_model
        
        action_val = kwargs.get("action_custom") or action
        position_val = kwargs.get("position_custom") or kwargs.get("position", "at sides")
        
        system = build_system_prompt(
            action=action_val,
            position=position_val,
            which_hands=kwargs.get("which_hands", "both hands same pose"),
            detail_level=kwargs.get("detail_level", "standard - clear finger positions"),
            holding=kwargs.get("holding", "nothing"),
            interaction=kwargs.get("interaction", "none"),
        )
        user_prompt = build_user_prompt(kwargs.get("description", ""))
        
        _log.info("[OllamaNodes] Generating hand pose: %s", action_val)
        
        raw = generate(
            base_url=cfg["base_url"], model=cfg["model"],
            prompt=user_prompt, system=system,
            temperature=cfg.get("temperature", 0.4),
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
            parsed.get("hand_prompt", ""),
            parsed.get("right_hand", ""),
            parsed.get("left_hand", ""),
            parsed.get("finger_positions", ""),
            parsed.get("palm_orientation", ""),
            parsed.get("wrist_angle", ""),
            parsed.get("hand_interaction", ""),
            parsed.get("simplification_tips", ""),
            parsed.get("hand_tags", ""),
            parsed.get("negative_prompt", ""),
            parsed.get("alternative_pose", ""),
        )
