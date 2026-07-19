"""
OllamaStyleIdentifier — Art Style Identifier
Identifies art styles, movements, and techniques in images.
"""
from __future__ import annotations
import json
import logging

from ...ollama_client import generate
from ...prompts.style_identifier import build_system_prompt, USER_PROMPT
from ...utils.image_utils import tensor_to_base64
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)

DEPTH_OPTIONS = ["quick", "detailed", "comprehensive"]


class OllamaStyleIdentifier:
    """Identifies artistic styles, movements, and techniques."""

    CATEGORY = "Ollama-Magic-Nodes/Image"
    FUNCTION = "identify_style"
    RETURN_TYPES = ("STRING",) * 8
    RETURN_NAMES = (
        "primary_style", "style_movement", "techniques", "possible_influences",
        "medium", "style_tags", "similar_artists", "replication_prompt",
    )
    OUTPUT_TOOLTIPS = (
        "Main art style identified",
        "Art movement/era",
        "Artistic techniques used",
        "Artist/work influences",
        "Apparent medium",
        "Tags for replication",
        "Artists with similar style",
        "Prompt to recreate this style",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to analyze"}),
                "ollama_model": ("OLLAMA_MODEL",),
                "depth": (DEPTH_OPTIONS, {
                    "default": "detailed",
                    "tooltip": "Analysis depth",
                }),
            },
        }

    def identify_style(self, image, ollama_model: dict, depth: str = "detailed"):
        cfg = ollama_model
        img_b64 = tensor_to_base64(image)
        system = build_system_prompt(depth)

        _log.info("[OllamaNodes] Identifying art style (depth=%s)", depth)

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
            _log.warning("[OllamaNodes] Failed to parse JSON for style identification")
            return (raw,) + ("",) * 7

        return (
            parsed.get("primary_style", ""),
            parsed.get("style_movement", ""),
            parsed.get("techniques", ""),
            parsed.get("possible_influences", ""),
            parsed.get("medium", ""),
            parsed.get("style_tags", ""),
            parsed.get("similar_artists", ""),
            parsed.get("replication_prompt", ""),
        )
