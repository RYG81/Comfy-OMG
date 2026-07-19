"""
OllamaColorPalette — Color Palette Extractor
Extracts detailed color palettes from images.
"""
from __future__ import annotations
import json
import logging

from ...ollama_client import generate
from ...prompts.color_palette import build_system_prompt, USER_PROMPT
from ...utils.image_utils import tensor_to_base64
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)

PALETTE_TYPES = ["dominant", "harmonious", "contrast", "mood"]


class OllamaColorPalette:
    """Extracts color palettes from images with hex codes and descriptions."""

    CATEGORY = "Ollama-Magic-Nodes/Image"
    FUNCTION = "extract_palette"
    RETURN_TYPES = ("STRING",) * 7
    RETURN_NAMES = (
        "palette_hex", "palette_names", "palette_description",
        "primary_color", "accent_colors", "mood_description", "color_prompt_tags",
    )
    OUTPUT_TOOLTIPS = (
        "Hex codes comma-separated",
        "Descriptive color names",
        "Natural description of palette",
        "Most dominant color",
        "Accent/highlight colors",
        "Mood the palette evokes",
        "Color tags for prompts",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to extract palette from"}),
                "ollama_model": ("OLLAMA_MODEL",),
                "num_colors": ("INT", {
                    "default": 5,
                    "min": 3, "max": 12, "step": 1,
                    "tooltip": "Number of colors to extract",
                }),
                "palette_type": (PALETTE_TYPES, {
                    "default": "dominant",
                    "tooltip": "Type of palette to extract",
                }),
            },
        }

    def extract_palette(self, image, ollama_model: dict,
                       num_colors: int = 5, palette_type: str = "dominant"):
        cfg = ollama_model
        img_b64 = tensor_to_base64(image)
        system = build_system_prompt(num_colors, palette_type)

        _log.info("[OllamaNodes] Extracting color palette (%d colors, type=%s)", num_colors, palette_type)

        raw = generate(
            base_url=cfg["base_url"],
            model=cfg["model"],
            prompt=USER_PROMPT,
            system=system,
            temperature=cfg.get("temperature", 0.4),
            num_ctx=cfg.get("num_ctx", 4096),
            num_predict=1500,
            seed=cfg.get("seed", -1),
            keep_alive=cfg.get("keep_alive", "5m"),
            images=[img_b64],
            response_format="json",
        )

        parsed = extract_json_block(raw)
        if parsed is None:
            _log.warning("[OllamaNodes] Failed to parse JSON for color palette")
            return (raw,) + ("",) * 6

        return (
            parsed.get("palette_hex", ""),
            parsed.get("palette_names", ""),
            parsed.get("palette_description", ""),
            parsed.get("primary_color", ""),
            parsed.get("accent_colors", ""),
            parsed.get("mood_description", ""),
            parsed.get("color_prompt_tags", ""),
        )
