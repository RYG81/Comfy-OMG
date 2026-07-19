"""
OllamaImageToStory — Generate narrative content from images.
"""
from __future__ import annotations
import json
import logging

from ...ollama_client import generate
from ...prompts.image_to_story import (
    build_system_prompt, USER_PROMPT,
    STORY_TYPE, TONE, PERSPECTIVE, LENGTH,
)
from ...utils.image_utils import tensor_to_base64
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)


class OllamaImageToStory:
    """Generate stories and narratives from images."""

    CATEGORY = "Ollama-Magic-Nodes/Image"
    FUNCTION = "generate_story"
    RETURN_TYPES = ("STRING",) * 9
    RETURN_NAMES = (
        "story", "title", "setting_description", "character_profiles",
        "mood_analysis", "dialogue_sample", "before_scene",
        "after_scene", "themes",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "ollama_model": ("OLLAMA_MODEL",),
            },
            "optional": {
                "story_type": (STORY_TYPE, {"default": "narrative description"}),
                "tone": (TONE, {"default": "neutral"}),
                "perspective": (PERSPECTIVE, {"default": "third person omniscient"}),
                "length": (LENGTH, {"default": "medium - 2-3 paragraphs"}),
            },
        }

    def generate_story(self, image, ollama_model: dict, **kwargs):
        cfg = ollama_model
        img_b64 = tensor_to_base64(image)
        
        system = build_system_prompt(
            story_type=kwargs.get("story_type", "narrative description"),
            tone=kwargs.get("tone", "neutral"),
            perspective=kwargs.get("perspective", "third person omniscient"),
            length=kwargs.get("length", "medium - 2-3 paragraphs"),
        )
        
        _log.info("[OllamaNodes] Generating story from image")
        
        raw = generate(
            base_url=cfg["base_url"], model=cfg["model"],
            prompt=USER_PROMPT, system=system,
            temperature=cfg.get("temperature", 0.8),
            num_ctx=cfg.get("num_ctx", 8192),
            num_predict=3000,
            seed=cfg.get("seed", -1),
            keep_alive=cfg.get("keep_alive", "5m"),
            images=[img_b64],
            response_format="json",
        )

        parsed = extract_json_block(raw)
        if parsed is None:
            return (raw,) + ("",) * 8

        return (
            parsed.get("story", ""),
            parsed.get("title", ""),
            parsed.get("setting_description", ""),
            parsed.get("character_profiles", ""),
            parsed.get("mood_analysis", ""),
            parsed.get("dialogue_sample", ""),
            parsed.get("before_scene", ""),
            parsed.get("after_scene", ""),
            parsed.get("themes", ""),
        )
