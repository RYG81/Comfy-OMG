"""
OllamaAutoTagger — Auto Tagger
Generates comprehensive tags from images.
"""
from __future__ import annotations
import json
import logging

from ...ollama_client import generate
from ...prompts.auto_tagger import build_system_prompt, USER_PROMPT
from ...utils.image_utils import tensor_to_base64
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)

FORMAT_OPTIONS = ["booru", "natural", "weighted", "all"]


class OllamaAutoTagger:
    """Generates comprehensive tags from images in various formats."""

    CATEGORY = "Ollama-Magic-Nodes/Image"
    FUNCTION = "tag"
    RETURN_TYPES = ("STRING",) * 7
    RETURN_NAMES = (
        "booru_tags", "natural_tags", "weighted_tags",
        "character_tags", "style_tags", "quality_tags", "meta_tags",
    )
    OUTPUT_TOOLTIPS = (
        "Danbooru-style tags",
        "Natural language tags",
        "Tags with weights",
        "Character-specific tags",
        "Style/aesthetic tags",
        "Quality/technical tags",
        "Meta information tags",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to tag"}),
                "ollama_model": ("OLLAMA_MODEL",),
                "tag_format": (FORMAT_OPTIONS, {
                    "default": "all",
                    "tooltip": "Tag format to generate",
                }),
                "max_tags": ("INT", {
                    "default": 50,
                    "min": 10, "max": 100, "step": 5,
                    "tooltip": "Maximum number of tags",
                }),
            },
            "optional": {
                "include_nsfw_tags": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Include content rating tags",
                }),
            },
        }

    def tag(self, image, ollama_model: dict, tag_format: str = "all",
           max_tags: int = 50, include_nsfw_tags: bool = False):
        cfg = ollama_model
        img_b64 = tensor_to_base64(image)
        system = build_system_prompt(tag_format, max_tags, include_nsfw_tags)

        _log.info("[OllamaNodes] Auto-tagging (format=%s, max=%d)", tag_format, max_tags)

        raw = generate(
            base_url=cfg["base_url"],
            model=cfg["model"],
            prompt=USER_PROMPT,
            system=system,
            temperature=cfg.get("temperature", 0.4),
            num_ctx=cfg.get("num_ctx", 8192),
            num_predict=2500,
            seed=cfg.get("seed", -1),
            keep_alive=cfg.get("keep_alive", "5m"),
            images=[img_b64],
            response_format="json",
        )

        parsed = extract_json_block(raw)
        if parsed is None:
            _log.warning("[OllamaNodes] Failed to parse JSON for auto-tagger")
            return (raw,) + ("",) * 6

        return (
            parsed.get("booru_tags", ""),
            parsed.get("natural_tags", ""),
            parsed.get("weighted_tags", ""),
            parsed.get("character_tags", ""),
            parsed.get("style_tags", ""),
            parsed.get("quality_tags", ""),
            parsed.get("meta_tags", ""),
        )
