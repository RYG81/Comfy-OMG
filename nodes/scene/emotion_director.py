"""
OllamaEmotionDirector — Direct emotional content in prompts.
"""
from __future__ import annotations
import json
import logging

from ...ollama_client import generate
from ...prompts.emotion_director import (
    PRIMARY_EMOTIONS, EMOTION_INTENSITY, EXPRESSION_STYLE, BODY_LANGUAGE_INTENSITY,
    build_system_prompt, build_user_prompt,
)
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)


class OllamaEmotionDirector:
    """Direct and intensify emotional content in prompts."""

    CATEGORY = "Ollama-Magic-Nodes/Scene"
    FUNCTION = "direct_emotion"
    RETURN_TYPES = ("STRING",) * 9
    RETURN_NAMES = (
        "emotional_prompt", "facial_direction", "eye_description",
        "body_language", "color_mood", "lighting_mood",
        "environmental_mood", "emotional_tags", "avoid",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "primary_emotion": (PRIMARY_EMOTIONS, {"default": "joy"}),
            },
            "optional": {
                "secondary_emotion": (["none"] + PRIMARY_EMOTIONS, {"default": "none"}),
                "intensity": (EMOTION_INTENSITY, {"default": "present - clearly visible, balanced"}),
                "expression_style": (EXPRESSION_STYLE, {"default": "realistic - natural human expression"}),
                "body_language": (BODY_LANGUAGE_INTENSITY, {"default": "moderate - clear body language"}),
            },
        }

    def direct_emotion(self, ollama_model: dict, prompt: str,
                      primary_emotion: str = "joy", **kwargs):
        cfg = ollama_model
        
        system = build_system_prompt(
            primary_emotion=primary_emotion,
            secondary_emotion=kwargs.get("secondary_emotion", "none"),
            intensity=kwargs.get("intensity", "present - clearly visible, balanced"),
            expression_style=kwargs.get("expression_style", "realistic - natural human expression"),
            body_language=kwargs.get("body_language", "moderate - clear body language"),
        )
        user_prompt = build_user_prompt(prompt)
        
        _log.info("[OllamaNodes] Directing emotion: %s", primary_emotion)
        
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
            parsed.get("emotional_prompt", ""),
            parsed.get("facial_direction", ""),
            parsed.get("eye_description", ""),
            parsed.get("body_language", ""),
            parsed.get("color_mood", ""),
            parsed.get("lighting_mood", ""),
            parsed.get("environmental_mood", ""),
            parsed.get("emotional_tags", ""),
            parsed.get("avoid", ""),
        )
